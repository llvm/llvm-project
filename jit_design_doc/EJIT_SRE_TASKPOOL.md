# EmbeddedJIT SRE Taskpool 编译调度机制

> 适用分支：`ejit_taskpool`
> 开关：`EJIT_SRE_TASKPOOL`（默认 **OFF**）
> 目标平台：`aarch64_be`（SRE，无 C++ 线程库）

---

## 1. 背景与设计约束

### 1.1 为什么需要 taskpool

`ejit_compile_or_get` 的原实现是一条直通路径：查 LRU cache → 未命中则同步编译 → 写入 LRU cache。
这在以下场景中有局限：

- **无去重**：多个调用方同时请求同一个 (funcIndex, cacheKey)，各自独立编译，浪费 CPU
- **无异步**：编译阻塞调用方，无法利用其他核并行
- **依赖 STL**：`std::unordered_map`、`std::list` 等容器在 SRE bare-metal 环境不可用

### 1.2 平台约束

目标 SRE 平台（aarch64_be）**没有 C++ 线程库**，因此禁用：

- `std::thread` / `std::async` / `std::future` / `std::promise`
- `std::mutex` / `std::shared_mutex` / `std::condition_variable`
- `<atomic>` / `<functional>`

以下 STL 容器通过**重载 `operator new`** 使用平台内存分配，可用：
- `<unordered_map>` / `<vector>` / `<string>`

SRE 平台提供三种基础并发原语：

| 原语 | 说明 | 本系统使用 |
|------|------|-----------|
| 原子变量 | SRE 原子读写/CAS（编译器 `__atomic_*` 内建或平台桩） | **主要使用** |
| 内存栅栏 | SRE acquire/release/full fence | 配合原子变量，保证内存可见性 |
| 核间信号量 | SRE 跨核通知/唤醒 | 可选项（将来用于 worker 唤醒优化，当前非核心） |

本系统以**原子变量 + 内存栅栏**为主要手段，实现轻量级无锁并发设计。在此基础上封装 `EJitAtomic`（底层抽象）和 `EJitRwLock`（读写锁），不再暴露底层原子指令。

平台另提供：
- SRE 队列原语 `QueueCreate` / `QueueWrite` / `QueueRead`（可选，通过宏切换）

### 1.3 核心设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 模型 | 纯异步：producer 入队 + 外部 poll worker 消费 | EJIT 不创建线程，worker 由外部平台驱动；不提供同步编译路径 |
| cache | 固定分桶 (32) × 每桶 `unordered_map` | 分桶限制 rehash 爆炸半径（rehash 只影响单桶读者），桶内弹性容量 |
| cache 读并发 | try-read（写标志置位 → 立即 fallback） | 读在热路径，不可阻塞等待 |
| cache 写并发 | spin-write（CAS 抢写标志 → spin 等读者退场） | 写不频繁，读者临界区极短（SRE 无抢占），spin 安全 |
| dedup | 扁平数组 `dedupState_[funcIndex]` + 5 态 CAS | `funcIndex` 直接当数组下标，O(1) 无分桶无扫描 |
| dedup 读并发 | lock-free（无原子锁直接读） | 单 worker 读者，Acquire-Release 保证可见性，尾端 CAS 兜底 |
| dedup 写并发 | try-write（CAS 抢槽，冲突 → 立即 fallback） | 多 producer + worker 可能冲突，不等待 |
| 队列 | Vyukov 无锁环形队列 (默认) / SRE 平台队列 | 默认可 host 测试，上板切换到平台原生队列 |
| 淘汰 | 逐实例 version 比对失效 + publish 覆盖 | toggle bump version → cache entry 逐实例比对不匹配 → 自然 miss |
| 分桶动机 | 限制 rehash 爆炸半径 | 单全局 `unordered_map` rehash 阻塞所有读者；分桶后 rehash 仅阻塞单桶 |
| 原子设施 | `EJitAtomic` wrapper → `EJitRwLock` 封装 | 原子底层集中封装，上层通过 RwLock 接口使用 |
| 模块划分 | Taskpool（调度） + JIT Compile Management（编译） | 关注点分离：调度不关心编译细节，编译不关心调度策略 |

---

## 2. 架构总览

整个系统分为两个模块：

| 模块 | 职责 | 关键组件 |
|------|------|---------|
| **Taskpool（调度层）** | 缓存管理、队列调度、开关控制、worker 驱动 | Cache, Dedup, Queue, SwitchController, Counters |
| **JIT Compile Management（编译层）** | IR 管理、编译流程控制、OrcJIT 引擎 | CompileDriver, OrcEngine, Optimizer, ModuleLoader, CodePool |

Taskpool 负责"这个编译请求谁来处理、结果如何缓存"，编译层负责"具体怎么编译"。Taskpool 通过回调接口调用编译层，不感知 IR/AOT/Opt 细节。

### 2.1 组件关系

```
EJitTaskPool                                EJitCompileDriver (编译层)
  ├── EJitRwLock             → 读写锁 (32个, 每桶独立) tryRead/write
  ├── EJitSwitchController   → 模式(Off/Async) + 每(dimType,instance)独立version
  ├── EJitTaskPoolCache      → 结果缓存 (32桶, 每桶 unordered_map)
  │                             cacheKey = hash(funcIndex, dims)
  │                             匹配: (cacheKey, dims, versions) 逐实例比对
  ├── EJitDedupTable         → 去重表 (扁平数组 dedupState_[n], O(1) funcIndex 索引)
  ├── EJitQueue              → 异步工作队列 (Vyukov ring / SRE 平台队列)
  └── EJitTaskPoolCounters   → 无锁原子统计计数器
```

### 2.2 数据流

```
compile_or_get(funcIndex, dims, numDims)
      │
      ├─ 0. 维度开关检查: for each (dimType, instanceId) in dims
      │     any disabled → InstanceDisabled, return fallback
      │
      ├─ 1. cacheKey = hash(funcIndex, dims)
      │
      ├─ 2. cache.lookup(funcIndex, dims, numDims) → hit → return {fnPtr, bucketIndex}
      │                                                (该桶 read token 外提)
      │                                                调用方: fnPtr(args)
      │                                                调用方: release_read(bucketIndex)
      │
      ├─ 3. Off mode → return fallback
      │
      ├─ 4. dedup.tryMarkPending(funcIndex) → 同 funcIndex 有 in-flight → fallback
      │
      └─ 5. queue.push({funcIndex, dims, versions[], fallback})
            → 立即返回 fallback
                    │
               外部 worker:
               pollOne() → runCompile → publish (该桶 write spin 等 readers→0)
```

### 2.3 Before vs After：`ejit_compile_or_get` 流程变化

```
                        BEFORE                                    AFTER
                  (ejit_dev_spec4)                        (ejit_taskpool)

ejit_compile_or_get                                   ejit_compile_or_get
  └─ EJit::getOrCompile                                 └─ EJit::getOrCompile
       └─ EJitCompileDriver::getOrCompile                    └─ EJitCompileDriver::getOrCompile
            │                                                     │
            ├─ LRU hash 查一次                                    └─ taskPool_->compileOrGet()
            │   hit → return                                          │
            │                                                         ├─ 维度开关检查
            └─ miss:                                                  │   any disabled → fallback
                 decode cacheKey                                      │
                 load bitcode                                         ├─ 维度开关检查
                 verify periods                                       │   any disabled → fallback
                 OrcJIT compile                                       ├─ cache.lookup(funcIndex, dims, numDims)
                   ↳ IR pipeline:                                     │   hit → {fnPtr, bucketIndex}
                     ① replace params                                ├─ dedup.tryMarkPending()
                     ② InstCombine                                   │   conflict → fallback
                     ③ StructFieldPass                               │
                     ④ L1/L2/L3 opts                                 ├─ queue.push({funcIndex, dims, versions})
                 lookup symbol                                        │   → return fallback
                 LRU cache.put()                                               │
                 return                                                    pollOne()
                                                                          → runCompile(worker核)
                                                                          → publish
```

核心差异：

| | Before | After |
|------|--------|-------|
| cache | LRU `unordered_map` + linked list | 32 桶 `unordered_map`，弹性容量 + rehash 隔离 |
| dedup | 无 | `EJitDedupTable`，同 key 只编一次 |
| 异步 | 无 | 入队 + 外部 poll worker，不创建线程 |
| 淘汰 | LRU 自动淘汰 | 逐实例 version 比对失效 + publish 覆盖 |
| 维度开关 | 无 | 每 `(dimType, instanceId)` 独立控制（无 IR 耦合） |
| 接口 | 内部 `cacheKey u64` 编码 | `funcIndex + dim pair 数组` 显式传入 |
| STL 依赖 | `<unordered_map>`, `<list>`, `<string>` (std::allocator) | `<unordered_map>` (重载 operator new) + `EJitAtomic` |

### 2.4 异步编译跑在哪个核？

EJIT **不分配核、不做 CPU affinity**。`pollOne` / `pollBudget` 是纯 C 函数——谁调用、在哪个栈/核上执行，EJIT 不感知。核绑定由外部平台完成：

```c
// 平台侧：创建 SRE task 并绑定到目标核
sre_task_create(&jit_worker_task,
                jit_worker_entry,
                SRE_CPU_AFFINITY_2);

// worker entry，运行在核 2 上
void jit_worker_entry(void) {
    while (1) {
        unsigned worked = ejit_taskpool_poll_one();
        if (!worked) sre_task_sleep_ms(1);
    }
}
```

```
核 0 (业务)                           核 2 (JIT worker)
ejit_compile_or_get()                 ejit_taskpool_poll_one()
  → miss                                → queue_.pop()
  → queue_.push() ────────────→        → runCompile()
  → 立即返回 fallback                   → cache_.publish()
```

---

## 3. 基础组件

### 3.1 EJitAtomic：原子操作 wrapper

**位置**：`llvm/include/llvm/ExecutionEngine/EJIT/EJitAtomic.h`

所有原子访问集中于此。taskpool 业务逻辑文件**不直接出现** `std::atomic` 或 `__atomic_*` 内建。

```cpp
template <typename T> class EJitAtomic {
public:
    T loadAcquire() const;           // __atomic_load_n(ACQUIRE)
    T loadRelaxed() const;
    void storeRelease(T v);          // __atomic_store_n(RELEASE)
    void storeRelaxed(T v);
    bool compareExchange(T &expected, T desired);  // strong CAS, ACQ_REL/ACQUIRE
    T fetchAdd(T v);                 // acq_rel
    T fetchSub(T v);
};

using EJitAtomicU32  = EJitAtomic<uint32_t>;
using EJitAtomicU64  = EJitAtomic<uint64_t>;
using EJitAtomicUPtr = EJitAtomic<uintptr_t>;
```

- 基于 `__atomic_*` 编译器内建，**不** `#include <atomic>`，`EJIT_FREESTANDING` 下可用
- 所有 memory order **显式写明**，便于后续替换为平台内建桩
- 不可拷贝/移动：每个实例标识一个固定内存位置

### 3.2 EJitRwLock：基于原子变量的读写锁

**位置**：`llvm/include/llvm/ExecutionEngine/EJIT/EJitRwLock.h`

`EJitRwLock` 在 `EJitAtomic` 之上封装，使用**两个独立原子变量**实现多读单写。

Cache 按桶粒度分配 RwLock：**每个桶 (32 个) 拥有独立的 `EJitRwLock` 实例**，共 32 个锁。写入桶 5 不影响桶 7 的读者。

```cpp
class EJitRwLock {
    EJitAtomicU32 writeFlag_;    // 0 = 空闲, 1 = 写者持有 / 等待写入
    EJitAtomicU32 readers_;      // 当前活跃读者计数
public:
    // 读侧（热路径，不可阻塞）
    // tryRead 成功后 readers_++ 保持，调用方使用完指针后必须主动调用 readRelease(bucketIndex)
    bool tryRead();              // 检查 writeFlag_，置位则立即返回 false。成功则 readers_++
    void readRelease();          // readers_--（调用方不再使用 fnPtr 时主动调用，与 tryRead 配对）

    // 写侧
    bool tryWrite();             // CAS 抢 writeFlag_，失败立即返回 false
    void write();                // CAS 抢 writeFlag_ → spin 等 readers_→0（已抢到后才等读者退场）
    void writeRelease();         // writeFlag_ = 0（与 write/tryWrite 配对）
};

// 32 个桶，每桶独立锁
EJitRwLock bucketLocks_[EJIT_SRE_TASKPOOL_BUCKETS];   // 默认 32
```

#### 3.2.1 tryRead（热路径读）

```cpp
bool tryRead() {
    if (writeFlag_.loadAcquire() != 0)
        return false;           // 写标志置位，立即回退到 fallback，不等待
    readers_.fetchAdd(1);       // 增加读者计数
    // 双重检查：fetchAdd 后写者可能已 CAS 抢到 writeFlag_
    if (writeFlag_.loadAcquire() != 0) {
        readers_.fetchSub(1);   // 写者已到，退出读者身份
        return false;
    }
    return true;                // 成功获取读权限
}
```

#### 3.2.2 write（spin 写入）

```cpp
void write() {
    // 1. CAS 抢写标志
    uint32_t expected = 0;
    while (!writeFlag_.compareExchange(expected, 1))
        expected = 0;           // spin 等写标志释放
    // 2. 已抢到写标志，等待所有读者退出
    while (readers_.loadAcquire() != 0)
        /* spin */;
    // 3. 此时：writeFlag_=1, readers_=0，独占写入
}
void writeRelease() {
    writeFlag_.storeRelease(0); // 释放写标志
}
```

#### 3.2.3 tryWrite（快速失败写入）

```cpp
bool tryWrite() {
    uint32_t expected = 0;
    if (!writeFlag_.compareExchange(expected, 1))
        return false;           // CAS 失败，立即返回，不等待
    if (readers_.loadAcquire() != 0) {
        writeFlag_.storeRelease(0);  // 有读者，回滚写标志
        return false;
    }
    return true;                // CAS 成功且无读者，立即持锁。调用方用完须 writeRelease()
}
```

#### 3.2.4 为什么读计数必须外提

RwLock 的核心设计约束来自一个简单的事实：**fnPtr 指向的代码可能被 worker 释放**。

```
错误做法（lookup 内部归还）：
  lookup() { tryRead → 读 fnPtr → readRelease → return fnPtr; }
                                                    ↑
                                         worker write() 此时可能写入并释放该 fnPtr
                                         调用方还握着这个地址！

正确做法（调用方主动归还，按桶粒度）：
  lookup() { tryRead(bucket) → 读 fnPtr → return {fnPtr, bucket}; }  // 不归还
  调用方: fnPtr(args);                                                 // 安全使用
  调用方: release_read(bucket);                                        // O(1) 归还
  worker write(bucket) { spin 等该桶 readers_→0 → 覆盖 + 释放旧代码; }
```

没有智能指针或 GC 的 C 环境下，无法自动判断 fnPtr 是否仍被使用。因此将"归还"操作暴露为调用方的显式责任：**谁获取了 fnPtr，谁就在使用完后调用 `release_read(bucketIndex)`**。

这带来的保证：`write(bucketIndex)` spin 到该桶 `readers_=0` 时，该桶所有拿到旧 fnPtr 的调用方均已归还，worker 可以安全释放被覆盖的旧代码。其他桶不受影响。

#### 3.2.5 各场景使用矩阵

| 组件 | 操作 | 接口 | 粒度 | 行为 |
|------|------|------|------|------|
| Cache 读 | tryRead | `tryRead(bucketIndex)` → 调用方用完 fnPtr 后 → `readRelease(bucketIndex)` | **每桶**独立锁 | 写标志置位 → 直接 fallback。成功则 readers_++ 保持，读计数外提 |
| Cache 写 | write | `write(bucketIndex)` → 覆盖旧 fnPtr → 释放旧代码 → `writeRelease(bucketIndex)` | **每桶**独立锁 | CAS 抢标志 → spin 等该桶 readers_→0 → 安全释放旧代码 → 写入 |
| Dedup 读 | lock-free | **不使用 RwLock** | — | 直接 `loadAcquire`，单 worker 读者，无需锁 |
| Dedup 写 | tryWrite | `tryWrite()` | slot 级 CAS | CAS 抢 slot 状态，冲突 → 立即 fallback |

Dedup 读不使用 RwLock 的原因见 §4.2.4 无锁读安全性分析。

### 3.3 EJitSreQueue：请求结构 + 无锁队列

**位置**：`llvm/include/llvm/ExecutionEngine/EJIT/EJitSreQueue.h`、`.cpp`

#### 3.3.1 EJitCompileRequest（队列元素）

固定布局 POD，无构造/析构、无 STL，可安全通过平台队列按值传递：

```cpp
struct EJitCompileRequest {
    uint32_t funcIndex;      // 函数标识
    uint32_t numDims;        // 维度对数量 (≤4)
    uintptr_t fallbackPtr;   // AOT fallback 函数指针
    ejit_dim_pair_t dims[4]; // 维度对数组 (numDims 有效)
    uint32_t versions[4];    // 入队时刻各实例 version 快照
    // cacheKey = hash(funcIndex, dims, numDims)  // worker dequeue 时自行计算
};
// aarch64: 16 + 4×8 + 4×4 = 64 字节
```

#### 3.3.2 EJitQueue（有界 MPSC 队列）

多生产者/单消费者。默认容量 1024，向上取整到 2 的幂。两个后端：

| 宏定义 | 后端 | 适用 |
|--------|------|------|
| (无) | Vyukov 无锁环形队列 | host 开发 / 测试 |
| `EJIT_SRE_TASKPOOL_PLATFORM_QUEUE` | SRE `QueueCreate/Write/Read` | 上板 |

**默认 Vyukov 环形队列原理**：

```
buffer_: [slot0] [slot1] ... [slot1023]
           ↑                  ↑
      dequeuePos          enqueuePos
slot = { EJitAtomicU32 sequence, EJitCompileRequest data }
```

**push（多生产者 CAS 抢占）**：

```
pos = enqueuePos
seq = slot[pos & mask].sequence (acquire)
if seq == pos:      → CAS(enqueuePos, pos, pos+1) 抢占 → 写 data → seq.storeRelease(pos+1)
if seq < pos:       → 队列满（生产者追上了消费者一整圈），返回 false
if seq > pos:       → 其他生产者已抢占，重读 enqueuePos
```

**pop（单消费者）**：

```
pos = dequeuePos
seq = slot[pos & mask].sequence (acquire)
if seq == pos+1:    → CAS(dequeuePos, pos, pos+1) 取得 → 读 data → seq.storeRelease(pos+mask+1)
if seq < pos+1:     → 队列空，返回 false
```

**SRE 平台后端**：定义宏后 push/pop 直接转发到 SRE 原语，SRE 符号 `QueueCreate`/`QueueWrite`/`QueueRead` 只声明不定义——缺符号在链接期暴露，不用 weak 兜底避免遮蔽平台实现。

---

## 4. Cache 与 Dedup

Cache 和 Dedup 使用不同的索引结构：Cache 分 32 桶（每桶 `unordered_map`），Dedup 扁平数组（`funcIndex` 直接索引）。

### cacheKey 计算

cacheKey 由 hash 算法生成，将 funcIndex 和 dim pair 数组混合：

```cpp
uint64_t hashCacheKey(uint32_t funcIndex, const ejit_dim_pair_t* dims, uint32_t numDims) {
    uint64_t key = funcIndex;
    for (uint32_t i = 0; i < numDims; i++) {
        key ^= ((uint64_t)dims[i].dimType << 32) | dims[i].instanceId;
        key *= 0x9e3779b97f4a7c15ULL;  // golden ratio
    }
    return key;
}
```

分桶公式：

```cpp
bucket = cacheKey % EJIT_SRE_TASKPOOL_BUCKETS   // 默认 32
```

### 4.1 EJitTaskPoolCache

32 个桶，每桶一个 `unordered_map<cacheKey, CacheEntry>` + 独立 `EJitRwLock`：

```
bucketLocks_[0]   → unordered_map<cacheKey, CacheEntry>    // 桶 0
bucketLocks_[1]   → unordered_map<cacheKey, CacheEntry>    // 桶 1
...
bucketLocks_[31]  → unordered_map<cacheKey, CacheEntry>    // 桶 31
```

每个函数最多关联 4 个生命周期维度，IR 中静态确定。cache entry 存储每个维度的 version 快照：

```cpp
struct EJitCacheEntry {
    uint32_t numDims;                      // 关联维度数 (≤4)
    struct {
        uint32_t dimType;                  // 维度类型编号
        uint32_t instanceId;               // 实例 ID
        uint32_t version;                  // 编译时刻快照
    } dims[4];
    uintptr_t fnPtr;                       // JIT 编译出的函数指针
};
// sizeof = 4 + 4×12 + 8 = 60 bytes

// bucket = cacheKey % 32
// 桶内: unordered_map<uint64_t, EJitCacheEntry>
```

**分桶的核心目的**：限制 `unordered_map` rehash 的爆炸半径。单全局 map rehash 会阻塞**所有**读者；分 32 个桶后，单桶 rehash 最多阻塞该桶的读者，其余 31 个桶不受影响。

**lookup**（try-read 语义，cacheKey 内部计算）：

```
lookup(funcIndex, dims, numDims) → {fnPtr, bucketIndex}：
  1. cacheKey = hash(funcIndex, dims, numDims)
  2. bucketIndex = cacheKey % 32
  3. bucketLocks_[bucketIndex].tryRead()  → 失败则立即返回 miss
  4. it = bucketMaps_[bucketIndex].find(cacheKey)
     → 未找到 → miss，readRelease()
  5. entry = it->second
     逐维度比对 version：
     for i in 0..entry.numDims:
         cur = switch_.getInstanceVersion(entry.dims[i].dimType, entry.dims[i].instanceId)
         if cur != entry.dims[i].version → miss，readRelease()
  6. 全部匹配 → 命中，返回 {fnPtr, bucketIndex}

命中时 lookup 不归还 read token。调用方用完 fnPtr 后：
  result = cache_.lookup(funcIndex, dims, numDims);
  result.fnPtr(args, ...);
  ejit_taskpool_release_read(result.bucketIndex);
```

**开关失效**：`set_instance_enabled(dim, id)` → version++ → 任何包含该实例的 cache entry 在步骤 4 比对失败 → 视为 miss。publish 时同 cacheKey 直接覆盖。零额外清理开销。

**为什么返回 bucketIndex**：`release_read(bucketIndex)` 直取 `bucketLocks_[bucketIndex].readRelease()`，O(1)。

**publish**（write 语义，单 worker 线程执行）：

1. `cacheKey = hash(funcIndex, dims, numDims)`
2. `bucketIndex = cacheKey % 32`
3. `bucketLocks_[bucketIndex].write()` —— spin 等该桶 readers_→0
4. 快照当前 version：`entry.dims[i].version = switch_.getInstanceVersion(...)`
5. `old = bucketMaps_[bucketIndex][cacheKey]` → 暂存 `old.fnPtr`
6. `bucketMaps_[bucketIndex][cacheKey] = entry` → 覆盖
7. 若 `old.fnPtr` 且 readers_=0 → 释放旧代码
8. `bucketLocks_[bucketIndex].writeRelease()`


### 4.2 EJitDedupTable

去重粒度是 **仅 `funcIndex`**。由于 `funcIndex` 本身是整数，**直接作为数组下标**——不需要分桶，不需要扫描：

```cpp
// 扁平数组，funcIndex 直接索引
EJitAtomicU32 dedupState_[MAX_FUNC_INDEX];     // 5 态：Empty/Claiming/Pending/Compiling/Publishing

// 辅助存储（仅 Pending/Compiling/Publishing 态有效）：
struct {
    ejit_dim_pair_t dims[4];
    uint32_t versions[4];
    uint32_t numDims;
    uintptr_t fallbackPtr;
} dedupPayload_[MAX_FUNC_INDEX];
```

**热点路径 O(1)**：

```
tryMarkPending(funcIndex):
  CAS(dedupState_[funcIndex], Empty, Claiming)  // 一次原子操作
  → 失败 → 读 state → AlreadyPending 或 DedupFull (不可能，数组与 funcIndex 一一对应)
  → 成功 → 写 dedupPayload_[funcIndex] → storeRelease(Pending)
```

**Worker 扫描**：遍历 `dedupState_[]` 找 Pending 条目 → `CAS(Pending→Compiling)` → 从 `dedupPayload_[funcIndex]` 读出编译参数。

**关键接口**：

```cpp
EJitDedupResult tryMarkPending(uint32_t funcIndex);
bool markCompiling(uint32_t funcIndex);
bool beginPublish(uint32_t funcIndex);
bool finishPublish(uint32_t funcIndex);
bool cancel(uint32_t funcIndex);
```

**覆盖三阶段**：

```
Empty ──(CAS)──→ Claiming ──(storeRelease)──→ Pending    ← 请求提交，等待 worker
                                                      ──→ Compiling   ← worker 编译中
                                                      ──→ Publishing  ← 写 cache 中
                                                      ──→ Empty       ← 完成释放
```

同一 `funcIndex` 在非 Empty 状态下，`tryMarkPending` 返回 `AlreadyPending`。

#### 4.2.4 Dedup 无锁读安全性分析

Dedup 的读者只有**单一 worker 线程**（`pollOne`/`pollBudget`）。由于采用扁平数组 `dedupState_[funcIndex]`，worker 可直接按索引读取，不需要扫描。

**worker 读 Dedup 的路径**：

```
worker 轮询 dedupState_[] 数组（或维护一个待处理 funcIndex 队列）：
  state = dedupState_[funcIndex].loadAcquire()
  
  if state == Pending:
     读 dedupPayload_[funcIndex].dims[]
     读 dedupPayload_[funcIndex].versions[]
     CAS(dedupState_[funcIndex], Pending, Compiling)  // 抢所有权
```

**安全性分析**（5 态 + Claiming 屏障，与之前一致）：

| 场景 | 分析 | 结论 |
|------|------|------|
| Producer 在 Claiming 态写 payload | `loadAcquire` 看到 Claiming，跳过 | ✅ |
| Producer 已 `storeRelease(Pending)` | Acquire-Release → worker 看到完整 payload | ✅ |
| Worker 读 payload 时 cancel 发生 | 尾端 CAS 失败，丢弃 | ✅ |
| 多 producer CAS 同一 `funcIndex` | 只有一个成功 | ✅ |

---

## 5. 调度器逻辑

### 5.1 EJitSwitchController

每个生命周期实例 `(dimType, instanceId)` 独立控制。**enabled 标志 + version 编码在同一 uint32_t 中**：

```
instanceVersions_[dimType][instanceId]:
  bit 0      = enabled (1=启用, 0=禁用)
  bits 31:1  = version (单调递增，每次 toggle +1)
```

```cpp
class EJitSwitchController {
    EJitAtomicU32 instanceState_[MAX_DIM_TYPES][MAX_INSTANCES];
    EJitAtomicU32 mode_;  // Off=0 / Async=1
public:
    // ===== 热路径：O(1) 检查 =====
    bool isInstanceEnabled(uint32_t dimType, uint32_t instanceId) {
        return instanceState_[dimType][instanceId].loadRelaxed() & 1;
    }

    // ===== 获取 version (bits 31:1) =====
    uint32_t getInstanceVersion(uint32_t dimType, uint32_t instanceId) {
        return instanceState_[dimType][instanceId].loadAcquire() >> 1;
    }

    // ===== 控制面：O(1) store =====
    void setInstanceEnabled(uint32_t dimType, uint32_t instanceId, bool enabled) {
        uint32_t old = instanceState_[dimType][instanceId].loadRelaxed();
        uint32_t version = (old >> 1) + 1;
        uint32_t newVal = (version << 1) | (enabled ? 1 : 0);
        instanceState_[dimType][instanceId].storeRelease(newVal);
    }
};
```

**关闭时的三层全懒失效**：

```
set_instance_enabled(0, 3, false)     // O(1) atomic store，立即返回
  ↓
第一层 Cache：新 lookup → 逐实例比对 version → 发现不匹配 → 自动 miss
第二层 Queue：worker dequeue 时逐实例比对 req.versions[] → 不匹配 → 丢弃
第三层 In-flight：runCompile gate 前逐实例比对 → 不匹配 → 丢弃，不 publish
```

**原理**：cache entry 存储编译时的 `dims[].version` 快照。toggle 仅 bump 一个实例的 version → 下一次 lookup 遍历 entry.dims[] 时发现该实例 version 不匹配 → miss。无需 XOR，零碰撞风险。

### 5.2 compileOrGet：统一入口

```cpp
CompileOrGetResult compileOrGet(funcIndex, dims, numDims, fallback) {
    // 0. 维度开关检查
    for each (dimType, instanceId) in dims:
        if (!switch_.isInstanceEnabled(dimType, instanceId))
            return {InstanceDisabled, fallback};

    // 1. cache hit（内部 hash → cacheKey → bucket → 逐实例比对 version）
    if (result = cache_.lookup(funcIndex, dims, numDims))
        return {CacheHit, result.fnPtr, result.bucketIndex};

    // 2. Off 模式
    if (switch_.getMode() == Off)
        return {DisabledFallback, fallback};

    // 3. dedup 去重（仅按 funcIndex。非 Empty → AlreadyPending）
    if (dedup_.tryMarkPending(funcIndex) == AlreadyPending)
        return {AlreadyPending, fallback};

    // 4. 异步入队 → 立即返回
    //    cacheKey 不入队——worker dequeue 后自行从 funcIndex+dims 计算
    req = {funcIndex, dims, numDims, fallback};
    for i in 0..numDims:
        req.versions[i] = switch_.getInstanceVersion(dims[i].dimType, dims[i].instanceId);
    if (!queue_.push(req)) {
        dedup_.clear(funcIndex);
        return {QueueFullFallback, fallback};
    }
    return {EnqueuedPending, fallback};
}
```

返回值状态全集：

| 状态 | fnPtr | 含义 |
|------|-------|------|
| CacheHit | JIT 函数指针 + bucketIndex | 命中缓存。调用方用完 fnPtr 后须 `release_read(bucketIndex)` |
| InstanceDisabled | fallback | 请求的某个生命周期实例被禁用 |
| OffMode | fallback | Taskpool 全局 Off 模式 |
| EnqueuedPending | fallback | 异步入队成功，等待 worker 编译 |
| AlreadyPending | fallback | 同 funcIndex 已有 in-flight |
| QueueFullFallback | fallback | 队列满，dedup 已回滚 |
| CompileFailed | fallback | 编译失败/实例 version 中途变更 |

### 5.3 runCompile：编译执行路径

由 worker（`pollOne`/`pollBudget`）调用。核心流程 + 每次的 CAS 检查：

```cpp
void *runCompile(req) {
    // cacheKey = hash(req.funcIndex, req.dims, req.numDims)  // 内部计算

    // 1. 检查点1：逐实例比对入队时刻 version vs 当前 version
    for i in 0..req.numDims:
        if req.versions[i] != switch_.getInstanceVersion(req.dims[i].dimType,
                                                          req.dims[i].instanceId)
            { dedup_.clear(req.funcIndex); → 丢弃 }

    // 2. Pending→Compiling (CAS)
    if (!dedup_.markCompiling(req.funcIndex)) → 丢弃

    // 3. 编译
    ok = compileFn_(ctx, req.funcIndex, req.dims, req.numDims, &fn);

    // 4. 检查点2：编译后再次逐实例比对
    for i in 0..req.numDims:
        if req.versions[i] != switch_.getInstanceVersion(req.dims[i].dimType,
                                                          req.dims[i].instanceId)
            { dedup_.cancel(req.funcIndex); → 丢弃 }

    if (!ok || !fn) {
        dedup_.clear(req.funcIndex); → 丢弃
    }

    // 5. Commit gate 1: Compiling→Publishing (CAS)
    if (!dedup_.beginPublish(req.funcIndex)) → 丢弃

    // 6. 写入 cache（内部快照当前 version）
    cache_.publish(req.funcIndex, req.dims, req.numDims, fn);

    // 7. Commit gate 2: Publishing→Empty (CAS)
    if (!dedup_.finishPublish(req.funcIndex)) {
        cache_.removeEntry(req.funcIndex, req.dims, req.numDims);
        → 丢弃
    }

    → fn (已发布到 cache)
}
```

两个 CAS gate 保证了 **实例开关竞态安全**（见 §5.4）。

### 5.4 实例开关失效机制

不再提供 `free_code` 逐个释放。改为通过 `set_instance_enabled` 控制：

```cpp
void setInstanceEnabled(uint32_t dimType, uint32_t instanceId, bool enabled) {
    // version++ → 所有引用该实例的 cache 条目中该 version 不匹配 → 自动失效
    instanceState_[dimType][instanceId].storeRelease( ... );
}
```

**失效路径**：

```
set_instance_enabled(0, 3, false)  // 关闭小区3
  → instanceVersion(0,3): 5 → 6
  → 任何 cache entry 中 dims[i]=(0,3): 存储的 version=5 ≠ 当前 6
  → lookup 逐实例比对时发现不匹配 → miss
  → in-flight compile 检查点发现 req.versions[i] 不匹配 → 丢弃
```

**与 runCompile 的竞态**：两个 CAS gate 同样保护。`set_instance_enabled` → version bump → worker 在 gate 1 前检测到 `versions[i]` 不匹配 → 丢弃结果。

**物理槽位回收**：版本号逻辑失效不删除 map entry。publish 时同 cacheKey 直接覆盖，旧 fnPtr 在 readers_=0 后释放。

---

## 6. 容量模型与约束

### 6.1 hash 分布与弹性容量

cacheKey 由 golden ratio hash 生成，`bucket = cacheKey % 32`。每桶内 `unordered_map`，弹性增长，无硬上限。总容量受平台内存约束。

### 6.2 rehash 隔离

单桶 `unordered_map` rehash 时，该桶 `write()` 持有写锁（publish 路径），或 `unordered_map` 内部触发 rehash（insert 时）。无论哪种，只阻塞该桶的读者，其余 31 个桶不受影响。

这就是分桶的核心价值——不是容量问题，是延迟隔离。

### 6.3 各层约束

| 边界 | 值 | 说明 |
|------|-----|------|
| 单桶容量 | 弹性 | `unordered_map`，受平台内存限制 |
| 单 funcIndex 在飞请求 | 1 | dedup 去重（O(1) 数组索引） |
| 逻辑失效 | 逐实例 version 不匹配 | toggle 后 lookup 自动 miss |

### 6.4 淘汰机制

| 途径 | 何时 | 操作 |
|------|------|------|
| 同 cacheKey 覆盖 | publish | `bucketMap[cacheKey] = newEntry`，旧 fnPtr 在 readers_=0 后释放 |
| 逻辑失效 | set_instance_enabled | 逐实例 version 不匹配 → lookup 自动 miss |

调大桶数减冲突：

```bash
-DEJIT_SRE_TASKPOOL_BUCKETS=64
```

---

## 7. C ABI

### 7.1 维度对结构体

```c
// 一个生命周期维度对：(维度类型, 实例ID)
// 例如：dimType=0("小区"), instanceId=3 → 小区3 的该特化版本
typedef struct {
    uint32_t dimType;       // 维度类型编号
    uint32_t instanceId;    // 该维度下的实例 ID
} ejit_dim_pair_t;
```

### 7.2 对外接口

6 个函数随 `libLLVMEJIT.a` 提供。仅在定义 `EJIT_SRE_TASKPOOL_PLATFORM_QUEUE` 时需外部提供 `QueueCreate`/`QueueWrite`/`QueueRead`。

```c
// ================= 核心编译接口 (AOT wrapper 调用) =================
// funcIndex: ejit_entry 函数索引
// dims: 维度对数组，描述此调用的特化维度 (如 [(0,3), (1,5)])
// numDims: 维度对数量
// outFn: 输出函数指针 (hit 或编译成功) / fallback
// outBucket: 输出桶号，调用方使用完 fnPtr 后须 release_read(outBucket)
ejit_status_t ejit_taskpool_compile_or_get(uint32_t funcIndex,
                                            const ejit_dim_pair_t* dims,
                                            uint32_t numDims,
                                            void** outFn,
                                            uint32_t* outBucket);

// ================= 开关控制 =================
// 控制某个生命周期实例的启用/禁用
// 禁用后：该实例相关的现有缓存视为失效，新请求直接 fallback
void ejit_taskpool_set_instance_enabled(uint32_t dimType, uint32_t instanceId,
                                        uint32_t enabled);

// ================= 读计数释放 =================
// 调用方使用完 compile_or_get 返回的 fnPtr 后调用
// outBucket 由 compile_or_get 返回，O(1) 定位桶锁
void ejit_taskpool_release_read(uint32_t bucketIndex);

// ================= Worker 驱动 =================
// 从队列消费 1 个请求，当前栈编译。返回 1(干活)/0(空)
unsigned ejit_taskpool_poll_one(void);

// 同上，一次最多消费 maxItems 个。返回实际数量
unsigned ejit_taskpool_poll_budget(unsigned maxItems);

// ================= 监控 =================
unsigned ejit_taskpool_pending_count(void);
ejit_status_t ejit_taskpool_get_stats(ejit_taskpool_stats_t *out);
```

**角色总结**：

```
开关控制:
ejit_taskpool_set_instance_enabled(dimType, instanceId, enabled)

生产者侧 (业务 / AOT wrapper)        消费者侧 (外部 worker 驱动)
ejit_taskpool_compile_or_get(       ejit_taskpool_poll_one
  funcIndex, dims, numDims,        ejit_taskpool_poll_budget
  &fnPtr, &bucket)                   (均在调用者栈/核上执行)
  → 命中 → 调用 fnPtr(...)
           release_read(bucket)
  → miss → 入队 → 返回 fallback

释放:
ejit_taskpool_release_read(bucketIndex)
```

**统计结构体**：

```c
typedef struct {
    uint64_t cacheHits, asyncCompiles, asyncEnqueues;
    uint64_t alreadyPending, queueFull;
    uint64_t compileFailed, publishFailed, instanceDisabled;
    uint32_t readyEntries, pendingEntries, queueApproxSize;
    uint32_t reserved;
} ejit_taskpool_stats_t;
```

新增 status code（additive，旧值不变）：`EJIT_ERR_QUEUE_FULL`、`EJIT_ERR_INSTANCE_DISABLED`、`EJIT_PENDING`。

---

## 8. Trace 与调试

关键路径埋了默认空展开的 trace 宏：

```cpp
#ifndef EJIT_TASKPOOL_TRACE
#define EJIT_TASKPOOL_TRACE(...) do {} while (0)
#endif
```

上板时可用 `-D'EJIT_TASKPOOL_TRACE(...)=SRE_printf(__VA_ARGS__)'` 重定义。参数限整数/指针/C 字符串。埋点覆盖：

| 函数 | 埋点 |
|------|------|
| `compileOrGet` | enter / cacheHit / dedup / queuePush / instanceDisabled |
| `runCompile` | begin / compiled / published |
| `pollOne` | empty / dequeued |
| `setInstanceEnabled` | enter / versionBump |

---

## 9. 构建与测试

### 9.1 构建

```bash
./build.sh release aarch64_be --freestanding --sre-taskpool
```

| 开关 | 说明 | 默认 |
|------|------|------|
| `--sre-taskpool` | 开关 taskpool | OFF |
| `--sre-taskpool-buckets=<n>` | cache 桶数 | 32 |
| `--sre-taskpool-queue-capacity=<n>` | 队列容量 (pow2) | 1024 |

对应 CMake option：`EJIT_SRE_TASKPOOL`、`EJIT_SRE_TASKPOOL_BUCKETS`、`EJIT_SRE_TASKPOOL_QUEUE_CAPACITY`。

### 9.2 测试

```bash
cmake --build build-ejit-sre-taskpool --target check-ejit-taskpool -j8
```

`EJITTaskPoolTests` 编译 `EJitTaskPool.cpp` + `EJitSreQueue.cpp`（带 `EJIT_SRE_TASKPOOL_TESTING`），不依赖 `EJITTests`，host 可跑。使用 mock compiler + mock ring queue，**不使用真实线程**——并发交错全部通过显式 `pollOne`/`pollBudget` + 测试钩子模拟。覆盖用例：

- atomic wrapper、RwLock 每桶隔离、SwitchController (enabled+version 编码)
- queue（容量/FIFO/满返回）
- dedup（funcIndex 直接索引 O(1)、5 态 transitions、cancel 阻断 finishPublish）
- cache（hash 分布、逐实例 version 比对失效）
- async 路径、实例开关竞态（编译中 toggle、publish 窗口内 version 变更）
- stats 计数、request flat POD 断言

---

## 10. 核间语义 hardening（本轮）

本轮目标是把 taskpool 从“单进程异步骨架”推进到“可支撑核间共享的并发语义”，重点不是引入 C++ 线程库，而是加固共享状态发布协议。

### 10.1 核间锁与屏障

- 新增 `EJitIpcBucketLock` + `EJitSharedBarrier` 封装。
- `lock(bucketId)` / `unlock(bucketId)` / `tryLock(bucketId)` 三接口统一可用。
- host/unit-test 默认使用 `EJitAtomicU32` spin lock；真实平台实现只做符号声明接入，不提供 weak fallback。
- 锁仅用于短临界区状态更新：cache bucket 元数据、dedup slot 状态迁移、局部统计。

禁止在持锁状态执行重操作：

- compile callback
- ORC/JITLink
- code pool allocate/seal
- SRE_printf/trace 输出
- 可能阻塞的队列读写

### 10.2 发布协议（release/acquire）

- queue producer 先写完整 `EJitCompileRequest`，再 release 发布 cell sequence。
- queue consumer acquire 读取 sequence，随后读取 request 内容。
- dedup slot 使用 `Empty -> Claiming -> Pending -> Compiling -> Publishing -> Empty`：
    - identity 字段先写
    - `storeRelease(Pending)` 才发布给读者
    - 读者以 acquire 读 state 后再匹配 identity
- cache publish 先写 `fnPtr/version/identity`，最后 `storeRelease(Ready)`。
- 调用方 acquire 看到 `Ready` 后读取 `fnPtr`。

### 10.3 activate/deactivate 与版本语义

- `EJitSwitchController` 增加 `activate(mode)` / `deactivate()`，均单调 bump version。
- `compileOrGet` 在 Off/Disabled 时不入队，直接 fallback。
- worker 在 dequeue 后、compile 后、publish 窗口中都会检查 active/version；失配时丢弃并回滚，不发布旧代结果。
- 旧 version 结果不会覆盖新 version cache。

### 10.4 FreeCode 语义

- `freeCode` 仍为 v1 logical free：只更新 taskpool 共享状态，不物理回收 code pool。
- 若 `freeCode` 与 publish 并发，worker 必须在 publish gate/finish gate 失败后回滚，最终不留下 READY。

### 10.5 大端共享结构约束

- 共享结构使用固定宽度整数与明确对齐。
- 不使用 bitfield。
- 不按字节解析整数，不把 native layout 持久化为跨端文件协议。

