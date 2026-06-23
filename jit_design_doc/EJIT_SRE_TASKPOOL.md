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

平台另需提供:

- **task 创建/销毁原语**:由 SRE 平台提供 task 创建/销毁能力。本系统封装为 `EJitSreTask`(§3.4)抽象层,业务代码不直接接触平台 task API。host 测试用 `std::thread` 实现,SRE 上板使用平台原语;两份实现链接时择一。

### 1.3 核心设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 模型 | 纯异步：producer 入队 + 单 worker 消费 | 单 worker 跑在独立 SRE task 上，由 `EJitSreTask` 抽象创建；不提供同步编译路径 |
| worker 数量 | **固定 1 个** | 队列采用 Vyukov MPSC 无锁实现，单消费者(SC)是无锁正确性的前提；多 worker 需要把 queue 升级到 MPMC，改造面大，当前不需要 |
| worker 平台依赖 | `EJitSreTask` 抽象层封装 | host 测试用 `std::thread`，SRE 用平台 task 原语，链接时择一；与 `EJitAtomic` 同样的"头文件抽象 + 平台实现"模式 |
| cache | 固定分桶 (32) × 每桶 `unordered_map` | 分桶限制 rehash 爆炸半径（rehash 只影响单桶读者），桶内弹性容量 |
| cache 读并发 | try-read（写标志置位 → 立即 fallback） | 读在热路径，不可阻塞等待 |
| cache 写并发 | spin-write（CAS 抢写标志 → spin 等读者退场） | 写不频繁，spin 开销可接受 |
| 队列 | Vyukov 无锁环形队列(自实现，MPSC) | 基础组件。SRE 平台未提供 queue 原语，使用自实现版本；多 producer + 单 consumer，单 worker 假设由此而来 |
| dedup | 扁平数组 `inFlight_[funcIndex]` + 1 bit 占位 | 基础组件。`funcIndex` 直接当数组下标，O(1) 无分桶无扫描；payload 全在 queue，dedup 不持有任何编译参数 |
| dedup 写并发 | CAS 抢占位 (0→1)，冲突 → 立即 fallback | 多 producer 可能冲突，不等待。worker 完成后回零 |
| taskQueue | `EJitTaskQueue` 组合 queue + dedup | 业务组件。向上提供去重的任务提交/消费统一接口，内部协调 queue 与 dedup 的生命周期 |
| 淘汰 | 逐实例 version 比对失效 + publish 覆盖 | toggle bump version → cache entry 逐实例比对不匹配 → 自然 miss |
| 分桶动机 | 限制 rehash 爆炸半径 | 单全局 `unordered_map` rehash 阻塞所有读者；分桶后 rehash 仅阻塞单桶 |
| 原子设施 | `EJitAtomic` wrapper → `EJitRwLock` 封装 | 原子底层集中封装，上层通过 RwLock 接口使用 |
| 模块划分 | 基础组件(Atomic/RwLock/Queue/SreTask/Dedup) + 业务组件(Cache/TaskQueue) + 调度器(Switch/Worker) + JIT Compile Management（编译） | 关注点分离：基础组件不感知编译语义；调度不关心编译细节，编译不关心调度策略 |

---

## 2. 架构总览

整个系统分为两个模块：

| 模块 | 职责 | 关键组件 |
|------|------|---------|
| **Taskpool（调度层）** | 缓存管理、任务队列调度、开关控制、worker 驱动 | Cache, TaskQueue, SwitchController, Worker, Counters |
| **JIT Compile Management（编译层）** | IR 管理、编译流程控制、OrcJIT 引擎 | CompileDriver, OrcEngine, Optimizer, ModuleLoader, CodePool |

Taskpool 负责"这个编译请求谁来处理、结果如何缓存"，编译层负责"具体怎么编译"。Taskpool 通过回调接口调用编译层，不感知 IR/AOT/Opt 细节。

### 2.1 组件关系

```
EJitTaskPool                                       EJitCompileDriver (编译层)
  │
  ├── 业务组件 (有状态)
  │     ├── EJitSwitchController   → 模式(Off/Async) + 每(dimType,instance)独立version
  │     ├── EJitTaskPoolCache      → 结果缓存 (32桶, 每桶 unordered_map + 独立 RwLock)
  │     │                             cacheKey = hash(funcIndex, dims)
  │     │                             匹配: (cacheKey, dims, versions) 逐实例比对
  │     ├── EJitTaskQueue          → 去重的任务提交管理 (组合 Queue + Dedup, §4.2)
  │     │                              tryEnqueue: dedup 占位 → queue push (失败自动回滚)
  │     │                              tryDequeue / release: queue pop / dedup 释放
  │     ├── EJitWorker             → 调度循环 (单 worker, 跑在独立 SRE task 上)
  │     │                             轮询 taskQueue.tryDequeue → runCompile, 软停止
  │     └── EJitTaskPoolCounters   → 无锁原子统计计数器
  │
  ├── 基础组件 (平台抽象, 不感知编译语义)
  │     ├── EJitAtomic             → 原子操作 wrapper (__atomic_* 封装)
  │     ├── EJitRwLock             → 读写锁 (基于 EJitAtomic, Cache 每桶一个)
  │     ├── EJitSreQueue           → 无锁 MPSC 环形队列 (Vyukov 自实现, §3.3)
  │     ├── EJitDedupTable         → 按整数 key 的 CAS 占位表 (§3.5)
  │     └── EJitSreTask            → 平台 task 抽象 (host: std::thread / SRE: 平台原语)
  │
  └── 编译边界
        compileFn_: CompileCallback (函数指针 + ctx 指针)
        runCompile() ──compileFn_(ctx, req, &fn)──→ EJitCompileDriver
                                                       (taskpool 不感知 IR/Orc/Opt)
```

**关键边界**:

- **`EJitTaskPool` 与 `EJitCompileDriver`** 通过 `CompileCallback` 函数指针解耦,taskpool 头文件不引用编译层任何类型
- **`EJitTaskQueue` 与基础组件**:TaskQueue 内部持有 `EJitQueue` + `EJitDedupTable` 实例,组合两者提供去重入队/出队/释放的统一接口；Queue 和 Dedup 作为基础组件不感知编译语义
- **`EJitWorker` 与 `EJitSreTask`**:Worker 持有 task 句柄,task 入口指向 worker 实例方法;SreTask 替换实现(host vs SRE)对 worker 透明
- **`EJitWorker` 与 `EJitTaskQueue`**:单 worker 通过 TaskQueue 的 tryDequeue 消费,单消费者是 Vyukov MPSC 队列的 SC 假设依赖,**多 worker 会破坏无锁正确性**(§1.3 决策)

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
      ├─ 4. taskQueue.tryEnqueue({funcIndex, dims, versions[], fallback})
      │     → 内部: dedup 占位 + queue push (失败自动回滚) → 立即返回 fallback
      │
      └─ EJitWorker (内部 task,§5.5):
           pollOne() → taskQueue.tryDequeue() → runCompile → publish (该桶 write spin 等 readers→0)
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
                     ① replace params                                ├─ taskQueue_.tryEnqueue()
                     ② InstCombine                                   │   dedup占位 → queue push
                     ③ StructFieldPass                               │   → return fallback
                     ④ L1/L2/L3 opts
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
| dedup | 无 | `EJitDedupTable`(基础组件)，1 bit 占位防止同 funcIndex 重复编译 |
| taskQueue | 无 | `EJitTaskQueue`(业务组件)，组合 queue + dedup 提供去重的任务提交管理 |
| 异步 | 无 | 入队 + 内部 EJitWorker 自动驱动，不暴露外部 poll 接口 |
| 淘汰 | LRU 自动淘汰 | 逐实例 version 比对失效 + publish 覆盖 |
| 维度开关 | 无 | 每 `(dimType, instanceId)` 独立控制（无 IR 耦合） |
| 接口 | 内部 `cacheKey u64` 编码 | `funcIndex + dim pair 数组` 显式传入 |
| STL 依赖 | `<unordered_map>`, `<list>`, `<string>` (std::allocator) | `<unordered_map>` (重载 operator new) + `EJitAtomic` |

### 2.4 异步编译跑在哪个核？

EJIT 内部封装一个 **`EJitWorker`** 模块,启动时通过 `EJitSreTask` 抽象层创建一个独立 task 承载编译循环。**核绑定由 `EJitSreTask` 的 SRE 实现决定**——业务代码、worker 模块本身、taskpool 业务组件均不感知具体运行在哪个核。

```cpp
// EJitTaskPool 内部启动 worker(由 Runtime 初始化时调用)
EJitTaskPool pool;
pool.setCompiler(&driver::compile, &driver);
pool.startWorker();          // 内部: EJitWorker → EJitSreTask::create

// EJitSreTask::create 的 SRE 实现负责具体核绑定:
//   bool EJitSreTask::create(EJitSreTask &out, EntryFn entry, void *ctx, ...) {
//       sre_task_create(&out.handle_, entry, ctx, SRE_CPU_AFFINITY_2);
//   }
// host 实现则用 std::thread,核绑定由 OS 调度器决定。

// worker 内部循环 (在 EJitWorker::run 中,由 SreTask 入口调用):
//   while (!stopRequested()) {
//       if (!pool_.pollOne()) yield();   // 空闲让出
//   }
```

**调用流向**(单 worker):

```
核 0 (业务)                               核 N (JIT worker, 由 SreTask 决定)
ejit_compile_or_get()                     EJitWorker::run() 循环中
  → miss → taskQueue_.tryEnqueue()          → pool.pollOne()
  → 立即返回 fallback                       → taskQueue_.tryDequeue()
                                            → runCompile()
                                               compileFn_(ctx, req, &fn)
                                            → cache_.publish()
                                            → taskQueue_.release()
```

`pollOne` / `pollBudget` 仍然作为 C ABI 保留,host 测试可以**不启动 worker**,直接在测试代码里调 `pollOne` 模拟时序;集成场景下由内部 `EJitWorker` 自动驱动。

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

using EJitAtomicU8   = EJitAtomic<uint8_t>;
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

**配对约束**：调用方对同一 `bucketIndex` 的 `release_read` 必须与 `lookup` **成功命中次数严格 1:1 配对**。同线程在第一次 release 之前可再次 lookup 同桶——若命中，`readers_` 累加为 2，须配对两次 release（嵌套的后续 `tryRead` 可能因写者介入而失败，此为正常行为）。这一约束直接对应 RwLock 的 `readers_` 计数语义：每次 `tryRead` 成功隐含一次 `fetchAdd(1)`。

#### 3.2.5 各场景使用矩阵

| 组件 | 操作 | 接口 | 粒度 | 行为 |
|------|------|------|------|------|
| Cache 读 | tryRead | `tryRead(bucketIndex)` → 调用方用完 fnPtr 后 → `readRelease(bucketIndex)` | **每桶**独立锁 | 写标志置位 → 直接 fallback。成功则 readers_++ 保持，读计数外提 |
| Cache 写 | write | `write(bucketIndex)` → 覆盖旧 fnPtr → 释放旧代码 → `writeRelease(bucketIndex)` | **每桶**独立锁 | CAS 抢标志 → spin 等该桶 readers_→0 → 安全释放旧代码 → 写入 |
| TaskQueue 入队 | 组合操作 | `tryEnqueue` → 内部 dedup CAS + queue push | — | dedup 占位失败 → AlreadyPending；push 失败 → 自动回滚 dedup |
| TaskQueue 出队/释放 | 转发 | `tryDequeue` → queue.pop / `release` → dedup.clear | — | consumer 仅通过 TaskQueue 操作，不直接接触基础组件 |
| Dedup (基础组件) 读 | lock-free | **不使用 RwLock** | — | 直接 `loadAcquire`，仅读 1 bit 占位标志 |
| Dedup (基础组件) 写 | CAS | `tryMarkPending` / `clear` | 单字 CAS | CAS 抢 0→1 占位；storeRelease(0) 回零 |

Dedup 不持有 payload（编译参数全部存放于 queue），因此读者只需要看 1 bit 占位标志，不存在 payload 撕裂问题，无需 RwLock，也无需中间屏障态。业务代码不应直接操作 Dedup，应通过 `EJitTaskQueue` 统一入口。

### 3.3 EJitSreQueue：请求结构 + 无锁队列

**位置**：`llvm/include/llvm/ExecutionEngine/EJIT/EJitSreQueue.h`、`.cpp`

#### 3.3.1 EJitCompileRequest（队列元素）

固定布局 POD，无构造/析构、无 STL，可按值传递：

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

多生产者/单消费者(MPSC)。**单消费者(SC)是 Vyukov 无锁正确性的硬约束**——因此 §1.3 决策固定 worker 数量为 1。默认容量 1024，向上取整到 2 的幂。

SRE 平台未提供 queue 原语，本系统使用**自实现的 Vyukov 无锁环形队列**作为唯一后端,host 测试与 SRE 上板共用同一份实现。

**Vyukov 环形队列原理**：

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

pop 端的 CAS 看似支持多消费者,**但 Vyukov 队列的 sequence 编码方案在多消费者并发时会出现"幽灵空"伪信号**(消费者 A 抢到 slot 后还没写 sequence,消费者 B 看到旧 sequence 误判为空,提前返回)。本系统通过 §1.3 worker 数量决策避开这个陷阱。如果将来确实需要多 worker,需要把队列升级到 MPMC 实现(例如 Michael-Scott 队列或带 sequence 双计数器的 MPMC ring)。

### 3.4 EJitSreTask：平台 task 抽象

**位置**：`llvm/include/llvm/ExecutionEngine/EJIT/EJitSreTask.h`、`EJitSreTask_host.cpp`、`EJitSreTask_sre.cpp`

`EJitSreTask` 是平台 task 创建/销毁原语的抽象层,**与 `EJitAtomic` 同级**——头文件定义接口,平台实现分别在 `_host.cpp`(用 `std::thread`)和 `_sre.cpp`(用 SRE 平台原语)中提供,链接时择一。

```cpp
class EJitSreTask {
public:
    using EntryFn = void (*)(void *ctx);

    // 创建并启动 task。entry 在新 task 上下文以 ctx 为参数调用。
    // name 用于平台调试(SRE 任务名 / pthread_setname),可为 nullptr。
    // 返回 true 表示创建成功;失败时 out 保持未初始化。
    static bool create(EJitSreTask &out, EntryFn entry, void *ctx,
                       const char *name = nullptr);

    // 请求软停止并等待 task 退出。task 内部应通过 stopRequested() 自检。
    // 调用后该实例不可再用。幂等。
    static void destroy(EJitSreTask &task);

    // task 内部查询:外部是否已请求停止
    bool stopRequested() const;

private:
    void *handle_ = nullptr;       // host: std::thread*; SRE: 平台句柄
    EJitAtomicU32 stopFlag_;
};
```

**约束**:

- **不支持强停止**:SRE 平台 task 强终止行为各异且容易导致编译中途资源泄漏。仅支持软停止(set flag → task 在循环顶端自检退出)。
- **不暴露核绑定接口**:核绑定是平台属性,在 SRE 实现的 `create` 内部根据上下文决定;host 实现交给 OS 调度器。`EJitSreTask` 接口对核绑定**保持透明**。
- **不可拷贝/移动**:每个实例对应一个固定 task 句柄。

### 3.5 EJitDedupTable：CAS 占位表

**位置**：`llvm/include/llvm/ExecutionEngine/EJIT/EJitDedupTable.h`

`EJitDedupTable` 是**通用基础组件**——按整数 key 做 CAS 占位/释放,不感知编译语义。去重粒度是 **仅 `funcIndex`**，由于 `funcIndex` 本身是整数，**直接作为数组下标**——不需要分桶，不需要扫描，不需要状态机：

```cpp
// 扁平数组：1 bit 占位，0 = 空闲，1 = 已有 in-flight
EJitAtomicU32 inFlight_[MAX_FUNC_INDEX];
```

#### 3.5.1 为什么是 1 bit 而不是状态机

dedup 的唯一职责是**"防止同一 key 同时被标记为 in-flight"**。在当前架构下：

- **payload 全部由 queue 持有**：`EJitCompileRequest` 完整结构体入队（§3.3.1），dedup 不存任何编译参数
- **跨核可观察由 queue 提供**：任何核扫 queue 的 ring buffer 即可看到所有待处理工作
- **失效由上层调度保证**：version bump 后 worker 在检查点丢弃过时结果（§5.3），不需要 dedup 参与

producer 和 consumer 对 dedup 的全部诉求是一个二元判断："这个 key 当前有没有人在做？" 1 bit 占位完全表达了这个语义。

#### 3.5.2 接口

```cpp
class EJitDedupTable {
    EJitAtomicU32 inFlight_[MAX_KEYS];   // 0 = 空闲, 1 = 占位中
public:
    // 尝试占位。CAS(0 → 1)
    //   返回 true：占位成功，调用方负责后续操作或 clear 回滚
    //   返回 false：已有 in-flight，调用方走 AlreadyPending
    bool tryMarkPending(uint32_t key);

    // 释放占位。storeRelease(0)
    //   场景：处理完成 / 失配丢弃 / 后续操作失败回滚
    void clear(uint32_t key);
};
```

热点路径仅一次 `compareExchange(0, 1)`，O(1) 无扫描。

#### 3.5.3 协作时序

```
producer:
  if (!dedup.tryMarkPending(key))        // CAS 0 → 1
      return AlreadyPending;
  if (!后续操作()) {
      dedup.clear(key);                   // 回滚占位
      return Failure;
  }
  return Success;

consumer (worker):
  取出工作;
  if (失配)           { dedup.clear(key); return; }
  处理();
  if (失配 || 失败)   { dedup.clear(key); return; }
  publish();
  dedup.clear(key);                        // 收尾
```

整个生命周期内 dedup 只有两种操作：CAS 占位、storeRelease 释放。无中间状态。

#### 3.5.4 与 queue 的职责分工

| 职责 | dedup | queue |
|------|-------|-------|
| 防止同 key 重复处理 | ✅ | — |
| 持有完整 payload | — | ✅ |
| 跨核观测待处理工作 | — | ✅ |
| 表达"谁在做、做到哪一步" | — | — |

第三行的"做到哪一步"既不在 dedup 也不在 queue——这是 consumer 自己的局部状态，不对外暴露。Dedup 保持 1 bit 精简，不为外部可观察性膨胀。

---

## 4. 结果缓存

Cache 使用分桶索引结构：32 桶（每桶 `unordered_map`）。

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


### 4.2 EJitTaskQueue：去重的任务提交管理

**位置**：`llvm/include/llvm/ExecutionEngine/EJIT/EJitTaskQueue.h`

`EJitTaskQueue` 是**业务组件**——它组合两个基础组件（`EJitQueue` + `EJitDedupTable`），向上提供去重的任务提交/消费统一接口。TaskQueue 是 producer（`compileOrGet`）和 consumer（`EJitWorker`）操作队列的唯一入口。

```cpp
class EJitTaskQueue {
    EJitQueue        queue_;   // MPSC 无锁环形队列 (§3.3)
    EJitDedupTable   dedup_;   // CAS 占位表 (§3.5)
public:
    // ===== Producer 侧 =====
    // 去重入队：内部先 dedup.tryMarkPending → 成功则 queue.push
    //   占位失败 → AlreadyPending
    //   push 失败 → 自动回滚 dedup.clear，返回 QueueFull
    //   push 成功 → Enqueued
    EnqueueResult tryEnqueue(const EJitCompileRequest &req);

    // ===== Consumer 侧 =====
    // 出队：直接转发 queue.pop。返回 true 表示取到工作。
    bool tryDequeue(EJitCompileRequest &out);

    // 释放占位：dedup.clear。worker 编译完成/失配丢弃时调用。
    void release(uint32_t funcIndex);
};
```

**设计要点**:

- **去重与入队原子化**：`tryEnqueue` 内部保证 dedup 占位 + queue push 为一个不可分割的操作序列——push 失败时自动回滚 dedup，调用方无需手动协调两者的生命周期
- **出队与释放分离**：`tryDequeue` 只从 queue 取数据，不操作 dedup；`release` 由 consumer 在合适的时机显式调用（编译完成 / 失配丢弃）。这保留了 consumer 在"取出工作"到"确认完成"之间自由执行重操作（编译）的能力
- **对基础组件透明封装**：TaskQueue 不改变 Queue 和 Dedup 的并发语义——Queue 仍为 MPSC、Dedup 仍为 CAS 占位。TaskQueue 只是把两者的调用编排成两个高层接口

**协作时序**：

```
producer (compileOrGet):
  result = taskQueue_.tryEnqueue(req);
  // 内部: dedup.tryMarkPending → queue.push (失败自动 clear)
  switch (result):
    Enqueued        → return fallback (等待 worker 处理)
    AlreadyPending  → return fallback (同 funcIndex 已在编)
    QueueFull       → return fallback (队列满，dedup 已回滚)

consumer (worker / runCompile):
  if (!taskQueue_.tryDequeue(req)) return;    // 队列空
  if (version 失配)        { taskQueue_.release(req.fi); return; }
  ok = compileFn_(req);
  if (version 失配 || !ok) { taskQueue_.release(req.fi); return; }
  cache.publish(req, fn);
  taskQueue_.release(req.fi);                  // 收尾
```

**与直接操作 Queue + Dedup 的对比**：

| | 直接操作 | 通过 TaskQueue |
|------|--------|------|
| producer 入队 | 手动 dedup.tryMarkPending → queue.push → 失败手动 clear | `tryEnqueue` 一步完成 |
| consumer 操作 | 手动 queue.pop + 多处 dedup.clear | `tryDequeue` + 统一 `release` |
| 回滚逻辑 | 分散在 compileOrGet 和 runCompile 多处 | 封装在 tryEnqueue / release 内 |
| Queue/Dedup 类型依赖 | 调用方直接 #include 两个基础组件头文件 | 调用方只依赖 TaskQueue 一个头文件 |

---

## 5. 调度器逻辑

### 5.1 EJitSwitchController

每个生命周期实例 `(dimType, instanceId)` 独立控制。**enabled 标志和 version 使用两个独立的原子变量**，避免在同一条原子操作中混合两个语义：

```
enabled_[dimType][instanceId]:  EJitAtomicU8   (0=禁用, 1=启用)
version_[dimType][instanceId]:  EJitAtomicU32  (单调递增，仅 enabled 状态变化时 +1)
```

**容量规格**：

| 常量 | 值 | 说明 |
|------|-----|------|
| `MAX_DIM_TYPES` | 8 | 最大维度类型数，编译期常量 |
| `MAX_INSTANCES` | 256 | 每维度最大实例数，instanceId ∈ [0, 255] |
| enabled 数组 | 8 × 256 × 1 = **2 KiB** | `EJitAtomicU8` |
| version 数组 | 8 × 256 × 4 = **8 KiB** | `EJitAtomicU32` |
| 总内存占用 | **10 KiB** | 二维静态数组，零运行时分配 |

```cpp
static_assert(MAX_DIM_TYPES == 8 && MAX_INSTANCES == 256);

class EJitSwitchController {
    EJitAtomicU8  enabled_[MAX_DIM_TYPES][MAX_INSTANCES];  // 2 KiB
    EJitAtomicU32 version_[MAX_DIM_TYPES][MAX_INSTANCES];  // 8 KiB
    EJitAtomicU32 mode_;  // Off=0 / Async=1
public:
    // ===== 热路径：直接读取，无位移 =====
    bool isInstanceEnabled(uint32_t dimType, uint32_t instanceId) {
        return enabled_[dimType][instanceId].loadRelaxed();
    }
    uint32_t getInstanceVersion(uint32_t dimType, uint32_t instanceId) {
        return version_[dimType][instanceId].loadAcquire();
    }

    // ===== 控制面：一次 CAS(enabled)，无循环 =====
    // 仅当 enabled 实际发生变化时，version 递增。
    // CAS 失败 = 已是目标状态 → no-op，version 保持不变。
    void setEnabled(uint32_t dimType, uint32_t instanceId, bool wantOn) {
        uint8_t expected = wantOn ? 0 : 1;   // 期望当前是相反状态
        uint8_t desired  = wantOn ? 1 : 0;
        if (enabled_[dimType][instanceId].compareExchange(expected, desired))
            version_[dimType][instanceId].fetchAdd(1);
    }
};
```

**热路径优化**：`isInstanceEnabled` 和 `getInstanceVersion` 都是纯 load，无需 `>> 1` 或 `& 1` 位移——因为 enabled 和 version 各自独立在专用原子变量中。

**为什么不用合并 U32（CAS 循环）？**

合并方案中 enabled 和 version 共享一个 U32 字，`setEnabled` 必须用 CAS 循环：load → 拆 version+enabled → 算新值 → CAS → 失败则重试。而拆分方案中 `setEnabled` 仅一次 CAS(enabled, old→new) 且从不循环——因为 enabled 从 0→1 (或 1→0) 只要成功一次就达到目标，中间没有需要重算的依赖。version 只在 CAS 成功后才 `fetchAdd`，是一个独立的原子操作，不参与 CAS 的条件判断。

**两个原子变量之间的窗口**：

```
disable 路径:
  CAS(enabled, 1→0) 成功  ← enabled 先变
  极小窗口...
  fetchAdd(version, 1)     ← version 后变

窗口内: enabled=0, version=旧值
```

分析两个方向：

- **窗口内有 producer 进来**：`isInstanceEnabled` 看到 enabled=0 → 直接返回 InstanceDisabled，不经过 version 比对。安全。
- **窗口内并发 enable/disable**：另一个线程 CAS(enabled, 0→1) 会成功或失败，version 最终各自 fetchAdd 一次，总增量正确。可能出现一次多余失效（version 连续涨了两次而中间没有实际编译产物），但不会造成错误命中。

**连续 toggle 的 version 单调性保证**：

`version_[][]` 通过 `fetchAdd(1)` 递增——这是一个 acq_rel 原子操作，**永远不回退**。考虑最坏情况：N 个线程并发对同一 `(dimType, instanceId)` 调用 `setEnabled`（连续 "关→开→关→开…"）：

- 每次状态实际翻转（CAS 成功）恰好对应一次 `fetchAdd(1)`
- 已是目标态的 setEnabled CAS 失败 → no-op，**不消耗 version 增量**
- N 次成功翻转 → version 累计 +N，严格单调

因此任何已入队请求快照的 `req.versions[i]` 一旦被一次成功 toggle "甩在身后"，就**永远无法再次匹配** current version。worker 检查点 1/2 的失配判定不会因为"先关后开恢复到原状态"而误判——状态可能恢复，但 version 不会。同样地，cache entry 中存储的 `dims[i].version` 一旦被 toggle，对应实例的 lookup 比对就**单调地、永久地** miss，直到下一次 publish 写入新 version 快照覆盖该 entry。

**为什么 version 用 32-bit？**

`EJitAtomicU32` version 回绕周期 ~2.1×10⁹ 次 toggle，实际不可能触发。enabled 用 U8 是因为它仅表达布尔值，1 字节已足够，且 `__atomic_compare_exchange_1` 原生支持。

**关闭时的三层全懒失效**：

```
setEnabled(0, 3, false)              // CAS(enabled 1→0) + fetchAdd(version)
  ↓
第一层 Cache：新 lookup → 逐实例比对 version → 发现不匹配 → 自动 miss
第二层 Queue：worker dequeue 时逐实例比对 req.versions[] → 不匹配 → 丢弃
第三层 In-flight：runCompile gate 前逐实例比对 → 不匹配 → 丢弃，不 publish
```

**原理**：cache entry 存储编译时的 `dims[].version` 快照。toggle 仅 bump 一个实例的 version → 下一次 lookup 遍历 entry.dims[] 时发现该实例 version 不匹配 → miss。

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

    // 3. 去重入队 → 立即返回
    //    taskQueue 内部: dedup 占位 + queue push (失败自动回滚)
    req = {funcIndex, dims, numDims, fallback};
    for i in 0..numDims:
        req.versions[i] = switch_.getInstanceVersion(dims[i].dimType, dims[i].instanceId);
    result = taskQueue_.tryEnqueue(req);
    if (result == Enqueued)
        return {EnqueuedPending, fallback};
    // AlreadyPending / QueueFull 均已由 taskQueue 内部处理 (dedup 自动回滚)
    return {result == AlreadyPending ? AlreadyPending : QueueFullFallback, fallback};
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

由 worker（`pollOne`/`pollBudget`）调用。核心流程 + 两次 version 检查：

```cpp
void *runCompile() {
    // 0. 从 TaskQueue 取工作
    EJitCompileRequest req;
    if (!taskQueue_.tryDequeue(req)) return nullptr;
    // cacheKey = hash(req.funcIndex, req.dims, req.numDims)  // 内部计算

    // 1. 检查点1：逐实例比对入队时刻 version vs 当前 version
    for i in 0..req.numDims:
        if req.versions[i] != switch_.getInstanceVersion(req.dims[i].dimType,
                                                          req.dims[i].instanceId)
            { taskQueue_.release(req.funcIndex); → 丢弃 }

    // 2. 编译
    ok = compileFn_(ctx, req.funcIndex, req.dims, req.numDims, &fn);

    // 3. 检查点2：编译后再次逐实例比对
    for i in 0..req.numDims:
        if req.versions[i] != switch_.getInstanceVersion(req.dims[i].dimType,
                                                          req.dims[i].instanceId)
            { taskQueue_.release(req.funcIndex); → 丢弃 }

    if (!ok || !fn) {
        taskQueue_.release(req.funcIndex); → 丢弃
    }

    // 4. 写入 cache（内部快照当前 version；同 cacheKey 直接覆盖旧 fnPtr）
    cache_.publish(req.funcIndex, req.dims, req.numDims, fn);

    // 5. 释放 TaskQueue 占位（dedup.clear）
    taskQueue_.release(req.funcIndex);

    → fn (已发布到 cache)
}
```

两个检查点保证了 **实例开关竞态安全**（见 §5.4）。TaskQueue 不参与 version 校验，`release` 只承担去重占位释放；版本失效完全由 SwitchController + 这两个检查点完成。

**检查点 2 与 publish 之间的窗口**：检查点 2 通过后、publish 步骤 4 快照 version 之前，仍可能再次 toggle。这一窗口是**良性**的：

- publish 在 §4.1 步骤 4 内部**重新调用** `switch_.getInstanceVersion(...)` 写入 entry，**不复用** `req.versions[]`。所以 cache 中 entry 持有的始终是 publish 那一刻的 current version 快照。
- 若 publish 之后立即 toggle 再 bump version，下一次 lookup 对该 entry 比对失败 → miss。entry 被无害"饿死"在 map 里，等下一次 publish 同 cacheKey 时覆盖。
- 若 publish 写入的 version 已经"过时"（极小窗口内又被 bump），表现为**这一次发布的 fnPtr 立刻失效**——和"toggle 在 publish 完成后 1ns 到达"等价，不引入新的安全问题。

### 5.4 实例开关失效机制

逻辑失效不依赖逐 entry 释放，统一通过 `set_instance_enabled` 触发：

```cpp
void setInstanceEnabled(uint32_t dimType, uint32_t instanceId, bool enabled) {
    // version++ → 所有引用该实例的 cache 条目中该 version 不匹配 → 自动失效
    instanceState_[dimType][instanceId].storeRelease( ... );
}
```

`freeCode` 不参与失效路径，仅在两个场景被调用（与 dedup/SwitchController 解耦）：

1. **模块 destroy**：整体清理 cache 时释放所有 fnPtr
2. **publish 覆盖**：cache 同 cacheKey 写入新 fnPtr 时，旧 fnPtr 在该桶 `readers_=0` 后释放

**失效路径**：

```
set_instance_enabled(0, 3, false)  // 关闭小区3
  → instanceVersion(0,3): 5 → 6
  → 任何 cache entry 中 dims[i]=(0,3): 存储的 version=5 ≠ 当前 6
  → lookup 逐实例比对时发现不匹配 → miss
  → in-flight compile 检查点发现 req.versions[i] 不匹配 → 丢弃
```

**与 runCompile 的竞态**：runCompile 的两个 version 检查点同样保护。`set_instance_enabled` → version bump → worker 在检查点 1 或检查点 2 检测到 `versions[i]` 不匹配 → `taskQueue_.release` 后丢弃结果，不写 cache。

**物理槽位回收**：版本号逻辑失效不删除 map entry。publish 时同 cacheKey 直接覆盖，旧 fnPtr 在 readers_=0 后释放。

### 5.5 EJitWorker：调度循环模块

`EJitWorker` 把"消费 TaskQueue + 调用 runCompile"封装成独立模块,跑在由 `EJitSreTask` 创建的独立 task 上。本系统采用**单 worker 模型**(§1.3 决策),依赖此假设维持 Vyukov 队列的 SC 正确性。

**职责**:

- 在初始化时启动 worker task(`EJitSreTask::create`)
- 提供 task 入口函数,在循环里轮询 `taskQueue_.tryDequeue()` → 空闲让出
- 在销毁时请求软停止并等待 task 退出(`EJitSreTask::destroy`)
- 维护 worker 局部统计(已处理数、空轮询数、运行标志)

**接口**:

```cpp
class EJitWorker {
public:
    explicit EJitWorker(EJitTaskPool &pool, const char *name = "ejit-worker");
    ~EJitWorker();   // 内部调用 stop()

    // 启动 worker task。失败返回 false。
    // 已经在运行时再次调用是 no-op。
    bool start();

    // 请求软停止并等待退出。幂等。
    void stop();

    // 监控
    bool   isRunning() const;
    uint64_t processedCount() const;   // 已成功处理的请求数
    uint64_t spinCount() const;        // 空轮询次数(队列为空)

private:
    static void taskEntry(void *ctx);  // SreTask 入口,转发到 run()
    void run();                        // 实际循环

    EJitTaskPool &pool_;
    const char  *name_;
    EJitSreTask  task_;
    EJitAtomicU64 processed_;
    EJitAtomicU64 spins_;
    EJitAtomicU32 running_;
};
```

**循环逻辑**:

```cpp
void EJitWorker::run() {
    running_.storeRelease(1);
    while (!task_.stopRequested()) {
        if (pool_.pollOne()) {          // 内部: taskQueue_.tryDequeue → runCompile
            processed_.fetchAdd(1);
        } else {
            spins_.fetchAdd(1);
            // 空闲让出。具体策略由实现选择:
            //   - 简单版: 让 CPU(yield/pause)
            //   - 优化版: 核间信号量等待(§1.2 提到的可选项)
        }
    }
    running_.storeRelease(0);
}
```

**生命周期与 EJitTaskPool 的耦合**:

- `EJitTaskPool` 持有 `EJitWorker *worker_ = nullptr`(可选)
- Runtime 初始化时调用 `pool.startWorker()` → 内部构造 `EJitWorker` 并 `start()`
- Runtime 销毁时调用 `pool.stopWorker()` → 析构 `EJitWorker`(内部 `stop()`)
- **Worker 不暴露给 C ABI**——和 Cache/TaskQueue 一样,是 taskpool 内部组件

**单 worker 假设的传递路径**:

```
EJitWorker::run() 中只此一处 pollOne(生产构建中是唯一调用方)
       ↓
EJitTaskPool::pollOne() → taskQueue_.tryDequeue() → runCompile
       ↓
EJitQueue::pop()  ← Vyukov MPSC 的 SC 端
```

只要 `startWorker` 在整个 taskpool 生命周期内**最多调用一次成功**(再次调用返回 no-op),且生产构建不暴露任何外部 pop 入口,就不会出现两个并发 pop 路径。`pollOne` / `pollBudget` 仅在测试构建(§7.2 的 testing-only 接口)下可用,与内部 worker 不同时启用。

**与 §10 核间语义的关系**:worker 跑在独立核时,worker 与 producer 的所有共享状态(queue cell、dedup 占位、cache entry、SwitchController version)都通过 `EJitAtomic` 的 acquire/release 配对发布,见 §10.2。

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
| 单 funcIndex 在飞请求 | 1 | TaskQueue 去重，内部通过 Dedup 1 bit 占位（O(1) 数组索引） |
| 维度类型上限 | 8 | `MAX_DIM_TYPES`，编译期常量 |
| 单维度实例上限 | 256 | `MAX_INSTANCES`，instanceId ∈ [0, 255] |
| SwitchController 内存 | 10 KiB | enabled: 8×256×1B + version: 8×256×4B，二维静态数组 |
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

生产构建提供 4 个 C 函数(随 `libLLVMEJIT.a`)。worker 由 taskpool 内部 `EJitWorker` 模块管理(§5.5),不在 C ABI 暴露。

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

// ================= 监控 =================
unsigned ejit_taskpool_pending_count(void);
ejit_status_t ejit_taskpool_get_stats(ejit_taskpool_stats_t *out);
```

**测试专用接口** —— 仅在定义 `EJIT_SRE_TASKPOOL_TESTING` 宏时编译,**不进入生产 `libLLVMEJIT.a`**:

```c
#ifdef EJIT_SRE_TASKPOOL_TESTING
// 直接驱动 queue 消费一次。host 测试中用于显式控制 pop 时序,
// 替代真实的 EJitWorker 循环。**不应在内部 worker 已启动时调用**——
// 测试代码自己负责保证生命周期不重叠(测试模式要么用真实 worker,要么用
// 这两个函数,二选一,详见 §9.2 的两类测试模式划分)。
unsigned ejit_taskpool_poll_one(void);
unsigned ejit_taskpool_poll_budget(unsigned maxItems);
#endif
```

**角色总结**：

```
开关控制:
ejit_taskpool_set_instance_enabled(dimType, instanceId, enabled)

生产者侧 (业务 / AOT wrapper)        消费者侧 (内部 EJitWorker, 自动驱动)
ejit_taskpool_compile_or_get(       内部 EJitSreTask 创建的 task 中循环:
  funcIndex, dims, numDims,           pool.pollOne()
  &fnPtr, &bucket)                  (核绑定由 SRE 实现决定)
  → 命中 → 调用 fnPtr(...)
           release_read(bucket)
  → miss → 入队 → 返回 fallback

释放:                                测试构建额外提供:
ejit_taskpool_release_read(...)        ejit_taskpool_poll_one      [TESTING-ONLY]
                                       ejit_taskpool_poll_budget   [TESTING-ONLY]
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
| `compileOrGet` | enter / cacheHit / taskQueueEnqueue / instanceDisabled |
| `EJitTaskQueue::tryEnqueue` | dedup / queuePush / dedupRollback |
| `runCompile` | begin / compiled / published |
| `pollOne` | empty / dequeued |
| `setInstanceEnabled` | enter / versionBump |
| `EJitWorker::start/stop` | started / stopRequested / exited |
| `EJitSreTask::create/destroy` | created / destroyed |

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

异步 wrapper 尚在稳定阶段，PASS3 暂时提供编译期开关：

| Clang 参数 | 生成的 wrapper | 默认 |
|-----------|---------------|------|
| 无 | 同步调用 `ejit_compile_or_get` | 是 |
| `-mllvm -ejit-wrapper-async` | 异步调用 `ejit_taskpool_compile_or_get` | 否 |

两种 wrapper 都使用注册阶段分配的 dense `funcIndex`，不会退回旧的函数名
hash。异步路径稳定后应删除此临时开关并固定生成 taskpool wrapper。

**`EJitSreTask` 实现选择**:链接时根据构建目标择一,与上述 CMake option 正交:

| 构建目标 | SreTask 实现 | 说明 |
|---------|------------|------|
| host (Linux/macOS 测试) | `EJitSreTask_host.cpp` | 用 `std::thread` 实现 task,核绑定交给 OS 调度器 |
| SRE 上板 (`--freestanding`) | `EJitSreTask_sre.cpp` | 调用 SRE 平台 task 原语,核绑定由 SRE 实现内部决定 |

构建脚本根据是否传入 `--freestanding` 自动选择对应 `.cpp` 编入 `libLLVMEJIT.a`。

### 9.2 测试

```bash
cmake --build build-ejit-sre-taskpool --target check-ejit-taskpool -j8
```

`EJITTaskPoolTests` 编译 `EJitTaskPool.cpp` + `EJitSreQueue.cpp` + `EJitSreTask_host.cpp` + `EJitWorker.cpp`(带 `EJIT_SRE_TASKPOOL_TESTING`),不依赖 `EJITTests`,host 可跑。两类测试模式:

- **不启动 worker 模式**(默认):测试代码用 mock compiler + 显式 `pollOne`/`pollBudget` 模拟时序,**不使用真实线程**,并发交错由测试钩子精确控制
- **启动 worker 模式**:用 `EJitSreTask_host.cpp`(`std::thread`)启动真实 worker,验证 `EJitWorker` 的启停、循环、计数;mock compiler 注入可控延迟

覆盖用例:

- atomic wrapper、RwLock 每桶隔离、SwitchController (enabled+version 编码)
- queue(基础组件：容量/FIFO/满返回)
- dedup(基础组件：funcIndex 直接索引 O(1)、CAS 占位/释放)
- taskQueue(业务组件：去重入队 tryEnqueue、自动回滚、tryDequeue/release)
- cache(hash 分布、逐实例 version 比对失效)
- async 路径、实例开关竞态(编译中 toggle、publish 窗口内 version 变更)
- worker 启停幂等、stop 后 task 退出、stopRequested 自检、processed/spin 计数
- stats 计数、request flat POD 断言

---

## 10. 核间语义 hardening（本轮）

本轮目标是把 taskpool 从“单进程异步骨架”推进到“可支撑核间共享的并发语义”，重点不是引入 C++ 线程库，而是加固共享状态发布协议。

### 10.1 核间锁与屏障

- 新增 `EJitIpcBucketLock` + `EJitSharedBarrier` 封装。
- `lock(bucketId)` / `unlock(bucketId)` / `tryLock(bucketId)` 三接口统一可用。
- host/unit-test 默认使用 `EJitAtomicU32` spin lock；真实平台实现只做符号声明接入，不提供 weak fallback。
- 锁仅用于短临界区状态更新：cache bucket 元数据、dedup 占位 CAS、局部统计。

禁止在持锁状态执行重操作：

- compile callback
- ORC/JITLink
- code pool allocate/seal
- SRE_printf/trace 输出
- 可能阻塞的队列读写
- `EJitWorker` 循环本身(worker 路径不进入持锁区,以避免 worker 与持锁的业务核相互等待)

### 10.2 发布协议（release/acquire）

- queue producer 先写完整 `EJitCompileRequest`，再 release 发布 cell sequence。
- queue consumer acquire 读取 sequence，随后读取 request 内容。
- dedup 仅维护 1 bit 占位（0/1）：
    - producer `compareExchange(0, 1)` 抢占位
    - worker / producer 完成或回滚时 `storeRelease(0)` 释放
    - 不持有 payload，不需要中间屏障态
- cache publish 先写 `fnPtr/version/identity`，最后 `storeRelease(Ready)`。
- 调用方 acquire 看到 `Ready` 后读取 `fnPtr`。

### 10.3 activate/deactivate 与版本语义

- `EJitSwitchController` 增加 `activate(mode)` / `deactivate()`，均单调 bump version。
- `compileOrGet` 在 Off/Disabled 时不入队，直接 fallback。
- worker 在 dequeue 后和 compile 后两个 version 检查点丢弃过时结果；失配时 `taskQueue_.release` + 不写 cache。
- 旧 version 结果不会覆盖新 version cache。

### 10.4 FreeCode 语义

- `freeCode` 不参与失效路径，仅在以下两个场景触发：
    - 模块 destroy：清理 cache 中所有 fnPtr
    - cache publish 覆盖：同 cacheKey 写入新 fnPtr 时，旧 fnPtr 在该桶 `readers_=0` 后释放
- 失效语义由 SwitchController 的 version bump + worker 检查点共同保证，不通过 `freeCode` 实现。
- `freeCode` 与 publish 互斥由 cache 桶锁（`EJitRwLock.write`）保证：写者 spin 到 `readers_=0` 后才释放旧 fnPtr，确保无人持有该指针。

### 10.5 大端共享结构约束

- 共享结构使用固定宽度整数与明确对齐。
- 不使用 bitfield。
- 不按字节解析整数，不把 native layout 持久化为跨端文件协议。

---

## 11. 跨核共享单例 worker（`EJIT_SRE_SHARED_TASKPOOL`）

> 开关：`EJIT_SRE_SHARED_TASKPOOL`（默认 **OFF**，要求 `EJIT_SRE_TASKPOOL=ON`）
> 默认 OFF 时：默认构建与既有 `EJIT_SRE_TASKPOOL=ON` 路径、ABI、产物**逐字节不变**。

第 1–10 节的 taskpool 是**每个 `EJit` 运行时实例各自持有一份** `EJitTaskPool`（`EJitCompileDriver` 内 `std::unique_ptr<EJitTaskPool>`）。在多核共享地址空间场景下这会产生**多个 worker、多份队列/缓存**，互相看不到对方的编译结果。本节把调度状态收敛为**单份跨核共享 POD 状态 + 单一 worker owner**：

```
Core 0 ─┐
Core 1 ─┼─> EJitSharedTaskPoolState（共享 section）─> 唯一 worker owner ─> LLVM JIT
Core N ─┘
```

### 11.1 为什么原先会创建多个 worker

- 每个 `EJit` 构造时 `EJitCompileDriver` 都 `make_unique<EJitTaskPool>`，各自含 queue/dedup/cache/SwitchController/worker。
- Async 模式下每个实例都 `startWorker()`，于是 N 个核 = N 个 worker、N 份缓存，编译结果不共享、去重不跨核。

共享方案：所有核绑定**同一份** `EJitSharedTaskPoolState`，通过 CAS 选出唯一 owner，只有 owner 创建 worker；其余核作为 producer 共享 queue/dedup/SwitchController/cache。

### 11.2 共享 vs 私有字段清单

**共享（POD，放入 `EJIT_SHARED_SECTION`，全部 `EJitAtomic` 访问）—— `EJitSharedTaskPoolState`：**

| 组 | 字段 |
|----|------|
| 头部 | `magic` / `abiVersion` / `structSize`（按值比较，端序无关） |
| owner/init | `initState`(状态机) / `ownerCoreId` / `generation` / `lastInitError` / `initAttempts` / `codeSharingEnabled` / `workerTaskId` / `registrationFingerprint`(核间注册一致性摘要) |
| SwitchController | `enabled[8][256]` / `version[8][256]` / `mode` |
| dedup | `inFlight[MAX_FUNC_INDEX]`（存 generation，0=空闲） |
| MPSC 队列 | `ring[QUEUE_SLOTS]`（Vyukov cell）/ `enqueuePos` / `dequeuePos`（分处独立 cache line） |
| 统计 | `EJitSharedCounters`（全 `EJitAtomicU64`） |
| 结果缓存 | `buckets[32]`，每桶固定 `slots[16]`（POD，无 `unordered_map`）+ 内联两字 RwLock |

**绝不共享（owner 核私有）：** `EJit`、`EJitCompileDriver`、`LLVMContext`、ORC/JITLink、`std::string/vector/map`、含虚函数/`unique_ptr` 的 C++ 对象、编译临时状态、compile/release/worker/idle 回调函数指针（指向 owner 私有对象）。`EJitFuncRegistry`/`EJitLifecycleRegistry` 仍为核私有 STL（不入共享区），仅其**指纹摘要**入共享状态。

ABI 约束（`static_assert`）：`is_standard_layout` + `is_trivially_destructible` + **`is_trivially_default_constructible`** + `alignof==64` + `offsetof(magic)==0`。**关键（本轮）**：`EJitAtomic` 改为**平凡默认构造**（`EJitAtomic() = default`），使整个 blob 平凡默认构造。因此全局 `gEJitSharedTaskPoolState` 落在 **`.bss`**（加载器零填），**不产生** `_GLOBAL__sub_I_*` / `.init_array` / 启动 `memset`——多核分别跑 init_array 的平台上，后启动的核**绝不会重零已运行的共享 queue/cache/owner 状态**。只有赢得 `Uninitialized→Initializing` CAS 的 owner 核才经 `initSharedStorage()` 逐字段初始化。blob 成为 implicit-lifetime 类型，共享存储上存在真实对象生命期，无 UB。依然**不** assert `trivially_copyable`（`EJitAtomic` 删除拷贝，永不 memcpy）。回归测试 `SharedStateRequiresNoDynamicInitialization` + 对 `EJitCompileDriver.cpp.o` 的 llvm-nm/readelf 检查双重保障无动态构造。

### 11.3 共享内存是否要求同虚拟地址

**是。** 两个前提要求所有核把该 blob 映射到**相同虚拟地址**：

1. `EJitAtomic` 直接对字段地址做 `__atomic_*`；跨核语义要求各核看到同一物理字且地址一致。
2. 缓存里的 `fnPtr` 是绝对代码地址；非 owner 核要执行它，必须在自己地址空间内同地址可见可执行（见 §11.6）。

### 11.4 owner election 状态机

`initState`（`EJitAtomicU32`）取值之一，**绝不用单个含糊 bool**：

```
Uninitialized ──CAS成功──> Initializing ──(建好全部共享状态+起worker)──> Ready
      │                          │                                         │
      │(CAS失败,旁观)            │(起worker失败,记录lastInitError)         │(owner shutdown:
      ▼                          ▼                                          软停+join后)
   旁观其它核                  Failed  <───────────────────────────────  Stopping ─> Uninitialized
```

- 首个把 `Uninitialized` CAS 成 `Initializing` 的核成为 owner：`initSharedStorage` 逐字段写 → bump `generation` → 写 `ownerCoreId/codeSharingEnabled/header` → 起唯一 worker → **最后 `storeRelease(Ready)`**（发布序：所有内容先就绪，Ready 最后发布）。
- 其它核 `acquire` 观察：`Ready` 校验 `magic/version/size` 后绑定（`AttachedReady`，**绝不创建第二个 worker**）；`Failed`/`Stopping` → 干净 fallback，**不无限等待**；`Initializing` → 有限自旋后返回 `InitInProgress`（pending，不死锁）。
- 重复 `init()` 幂等：owner 再次 `init()` 观察到 `Ready` 即 `AttachedReady`，不重选不重建。
- worker 起动失败传播为 `OwnerFailed`/`Failed` + `lastInitError=WorkerStartFailed`，`ejit_init` 失败并销毁实例，**绝不把 init 失败伪装成 JIT 成功**。
- 核 ID 注入式：`EJitCoreId::current()`。host 用每线程可设值（`setCurrentForTest`）在单进程内**无真实线程**地确定性模拟多核；真实平台绑定**只声明、不提供 weak fallback** 的 `ejit_sre_current_core_id`（缺失即链接错误）。

### 11.5 queue 发布协议（跨核）

Vyukov MPSC，单消费者=唯一 owner worker（§1.3 SC 前提不变）：

- producer：`cell.sequence acquire == pos` → CAS `enqueuePos` 抢槽 → **先写完整 `EJitCompileRequest`** → `cell.sequence.storeRelease(pos+1)` 发布；队列满（`seq<pos`）→ 干净 fallback 并**回滚 dedup 占位**。
- consumer：`cell.sequence acquire == pos+1` → CAS `dequeuePos` → 读 request → `storeRelease(pos+mask+1)` 释放槽。
- 半写不可见：consumer 只有看到 release 后的 sequence 才读 request；producer 看不到他核半写的 cell。
- **`req.generation`（v2 ABI，真实字段）**：`EJitCompileRequest` 新增 `uint32_t generation`，enqueue 时 `acquire` 当前 `state.generation` 写入。consumer 在编译前（checkpoint 0）与编译后（checkpoint 2）都校验 `req.generation == state.generation`，不匹配则丢弃、回滚 dedup、不编译不 publish。`cachePublish` 锁内再次校验 `req.generation == 当前 generation`（**用 req.generation，不以当前 generation 顶替**），并以 `req.generation` 写槽。owner re-init bump generation 后，旧请求/旧编译结果都进不了新 cache。

### 11.6 dedup / cache 跨核协议与 fnPtr 可共享前提

- **dedup（generation-aware，v2 ABI）**：扁平槽 `inFlight[funcIndex]` 存**占位的 generation**（0=空闲），不再是 1 bit。`dedupMark(fi,gen)=CAS(0→gen)`；`dedupClear(fi,gen)=CAS(gen→0)`——只清掉仍属于本 generation 的槽。**关键**：若平台 `SRE_TaskDelete` 不等价 join，旧 generation 的 stale worker 完成后调用 `dedupClear(fi, oldGen)`，此时槽可能已被新 generation 的 producer 重新占为 `newGen`，`CAS(oldGen→0)` 失败 → **不会误清新 generation 的 in-flight 标记**，避免重复编译。queue 满回滚同样按本 gen `CAS`。
- cache：固定槽 POD 表替代 `unordered_map`（共享内存不能放 STL）。
  - publish（仅 owner worker）：桶写锁 spin 到 `readers==0` → **锁内重校验逐实例 version**（commit gate，失配 `VersionMismatch` 不写）→ 同 identity 原地覆盖 / 空槽 / 桶满淘汰 → `storeRelease(state=Ready)` → **锁外**通过 owner 私有 release 回调释放旧/被淘汰 fnPtr（回调可能重入 code pool/ORC/分配器，绝不在临界区内运行）。
  - lookup：桶 `tryRead` → 扫 `Ready` 且 `generation`/identity/version 匹配 → 命中持读 token（调用方用完 `release_read`）。
- **fnPtr 跨核可共享前提（构建开关 `EJIT_SRE_SHARED_CODE_POINTERS`，默认 OFF）：** 仅当平台保证①同一地址空间、②code pool 在所有核映射到**相同 VA**、③code pool 生命周期覆盖所有读者、④I/D-cache 跨核执行一致时才显式置 ON（`EJitCompileDriver` 据此调 `setCodeSharingEnabled`，**绝不自动猜测平台能力**）。即使地址相同，`enable_ex` 修改的执行权限也可能只属于调用核的 stage-1 映射，因此非 owner 核命中共享 fnPtr 后，必须先通过 `PrepareCodeCallback` 在**当前核**安装执行权限；成功后在 cache slot 的 64-bit `executableCoreMask` 中记录该核，后续同核命中不重复调用。准备失败时返回 `readyButNotShareable`（fallback，**不重新入队**），绝不返回不可执行指针。当前平台适配完整支持 legacy 2MiB seal：按 2MiB 对齐 fnPtr 后调用 `enable_ex(1, poolBase)`；4K seal 模式因裸 fnPtr 不携带完整代码范围而 clean reject，后续需要在 cache slot 增加代码范围后才能安全共享。owner 核读自己的 fnPtr 不重复准备。`codeSharingEnabled` 和 `executePrepareFailed` 在诊断中输出。
- 大端/内存序：仅固定宽度标量按值访问，无 bitfield、无字节解析；所有发布/消费成对 acquire/release（§10.2）。共享状态 ABI version 因 request/dedup 变化升到 **v2**。

### 11.7 worker 启动时序、栈与任务生命周期

- **启动抢跑修复（worker 等待 Ready + 让出 CPU）**：owner `init()` 顺序为 `CAS(→Initializing)` → 建好全部共享状态 → `workerStart` → `storeRelease(Ready)`（成功）/`Failed`（失败）。在 `SRE_TaskCreate` 立即调度、且 worker 优先级较高的环境里，worker 可能在 owner 发布 Ready 之前就跑起来，观察到 `Initializing`：**必须主动让出调度（调用注入的 idle/yield hook），绝不提前退出，绝不读半初始化的 queue/cache**。状态机 `workerPollOnce()`：`Ready→消费`、`Initializing→WaitForReady（yield）`、`Failed/Stopping/Uninitialized→Exit`。
- **不再用 spin budget 退出**：生产 worker 是生命周期完整的任务——`runWorkerLoop` 永远**不会**因为 owner 稍慢就退出（删除了上轮的 `workerStartupSpinBudget_`），只在终态（Failed/Stopping/Uninitialized）退出。测试通过**可注入 idle callback / step machine** 受控驱动转态与退出，不靠真实线程。
- **idle/yield 抽象**：Read-but-empty 与 Initializing-wait 两种空闲都调用同一 idle hook；生产构建注入 `EJitSreTask::yield()`（freestanding=`SRE_TaskDelay(1)`，host=`std::this_thread::yield()`），**共享 taskpool 核心不直接依赖 `SRE_TaskDelay`**。禁止在持有 bucket lock / queue slot / dedup 临界态时 yield（`pollOne` 返回后才 idle）。诊断计数 `workerIdleYields`/`workerConsumeLoops`/`workerWaitedForReady` 证明 worker 让出且真正进入 Ready 消费阶段。同理，竞选中的 peer `init()` 观察到 `Initializing` 也调 idle hook 让出，不饥饿 owner。`workerStart` 失败仍发布 `Failed`。
- **worker 栈（单一事实源）**：`EJIT_SRE_TASKPOOL_WORKER_STACK_SIZE`（默认 `1048576`，**仅默认值**）。CMake 在 `EJIT_SRE_TASKPOOL` 打开时即定义该宏；`EJitSreTask_sre.cpp` 的 `init.uwStackSize` **直接使用同一个宏**（删除旧 `EJIT_SRE_TASK_STACK_SIZE`，消除双事实源；源码仅为非 CMake 编译保留保守 fallback）。`static_assert` 校验非零、16 字节对齐（AArch64 SP）、可放入 32 位 `uwStackSize`；任务创建日志打印最终栈大小。LLVM optimize/codegen/ORC/JITLink 跑在 worker 栈，平台**必须实测栈峰值**并调整（不要默认设 5 MiB）。
- worker 起动经注入式 hook（owner 私有），`TaskCreate` 失败干净返回 → init 失败。
- **stop=软停 + join**：`ownerShutdown` 先 `storeRelease(Stopping)`（worker 循环顶检查到 `Stopping` 退出）→ 调 worker stop → 才把状态归 `Uninitialized`、bump generation。`EJitSreTask::destroy` 的 `SRE_TaskDelete` **契约要求阻塞到 worker 真正退出（真 join）**——若平台删除非 join，generation-aware dedup（§11.6）兜底防止污染新 generation，但仍要求真 join 以杜绝 worker 回调触碰已销毁的 driver。`EJitCompileDriver` 析构先 `ownerShutdown` 再析构 owner 私有 ORC/driver；非 owner 实例析构对 worker **无副作用**（`ownerShutdown` 检查 `isOwner_`）。

### 11.8 新增只读诊断 API

`EJitSharedTaskPool::getDiagnostics(EJitSharedDiagnostics&)`：`initState` / `ownerCoreId` / `generation` / `lastInitError` / `initAttempts` / `codeSharingEnabled` / `workerTaskId` / `queueDepth` / `pendingCount` / `cacheReadyCount` / 全部统计计数（enqueue/cacheHit/compile/publish/…）。taskpool C ABI（`ejit_taskpool_compile_or_get` / `set_instance_enabled` / `release_read` / `pending_count` / `get_stats` / 测试用 `poll_one`/`poll_budget`）在 shared 构建经 `activeTaskPool()` 路由到共享池；统计经 `getDiagnostics` 映射。诊断/trace 宏默认完全不展开、参数不求值，可外部重定义为 `SRE_printf`；**不**用 `raw_ostream/std::string` 组织平台日志。

### 11.9 构建开关（最小集合）

| CMake / build.sh | 含义 | 默认 |
|------|------|------|
| `EJIT_SRE_SHARED_TASKPOOL` / `--sre-shared-taskpool` | 跨核共享 taskpool 总开关（要求 `EJIT_SRE_TASKPOOL`） | OFF |
| `EJIT_SRE_TASKPOOL_WORKER_STACK_SIZE` / `--sre-taskpool-worker-stack-size=` | worker 栈字节数（**单一事实源**，`EJIT_SRE_TASKPOOL` 打开即定义；`EJitSreTask_sre.cpp` 直接消费） | 1048576 |
| `EJIT_SRE_SHARED_CODE_POINTERS` / `--sre-shared-code-pointers` | 允许非 owner 核读共享 cache fnPtr（需平台同 VA + cache coherent；错误开启可致非法执行） | OFF |
| `EJIT_SRE_SHARED_TASKPOOL_CACHE_SLOTS` | 每桶固定缓存槽数 | 16 |
| `EJIT_SHARED_SECTION_ATTR` → `EJIT_SHARED_SECTION` | 共享 section 放置属性（host 空；平台覆盖） | 空 |
| `EJIT_SRE_SHARED_TASKPOOL_PLATFORM`（自动） | shared+`EJIT_FREESTANDING` 时由 CMake 自动定义：`EJitCoreId::current()` 绑定**只声明、无 weak fallback** 的平台符号 `ejit_sre_current_core_id`（缺失即链接错误）；host（非 freestanding）保留可设的模拟核 ID | 随 freestanding |

**平台必须提供的符号（shared+freestanding）：** `extern "C" uint32_t ejit_sre_current_core_id();`（强符号，参与 owner 判断，不能让所有真实核默认 0）；以及本地 `EJitSreTask_sre.cpp` 适配的 `SRE_TaskCreate/SRE_TaskDelete/SRE_TaskDelay`（`SRE_TaskDelete` 须真 join）。**本轮新增前提校验**：当 `EJIT_SRE_SHARED_TASKPOOL=ON` 且 `EJIT_FREESTANDING=ON` 而 `EJIT_SHARED_SECTION_ATTR` 为空时，CMake **直接 `FATAL_ERROR`**（空 section 意味着状态落在每镜像私有 `.bss`，根本不跨核共享，不允许静默通过）；构建摘要打印最终 section 属性。沿用既有固定值（buckets=32、queue=1024、maxFuncIndex=4096）。默认 OFF 时新代码不编译；`EJit/EJitCompileDriver/EJitRuntime` 的绑定全部 `#ifdef EJIT_SRE_SHARED_TASKPOOL`。

**核间注册一致性（registration fingerprint）：** `EJitFuncRegistry`/`EJitLifecycleRegistry` 仍是核私有 STL（不入共享区）。每个核对自身 `(name→funcIndex)` 与 `(name→dimType)` 映射计算确定性 FNV-1a 摘要（func 部分 XOR 累加、与 map 遍历顺序无关；lifecycle 部分按 slot 顺序）。owner 将摘要发布到 `state.registrationFingerprint`；peer attach 时比对自身摘要，**不一致返回 `FingerprintMismatch` clean fail，绝不提交请求**（避免 owner 与 producer 的 funcIndex/dimType 映射不同而静默跨索引）。

### 11.10 测试与尚需真实平台验证项

host `EJITSharedTaskPoolTests`（单进程、注入式核 ID 确定性模拟多核，无真实线程）覆盖：单一 owner、多核仅一个 worker、半初始化不可见、owner 失败传播 Failed、多 producer 共享队列、跨核同 key 去重、queue 满回滚 dedup、generation 切换丢弃旧状态、编译中 deactivate 阻止 publish、publish 可见性 + read token 释放、跨核取回同一 fnPtr（`codeSharingEnabled`）+ 关闭时 clean reject、publish 覆盖释放旧 code、大端字段语义、ABI layout/`static_assert`、实例禁用不入队。**本轮新增**：worker 在 Initializing 启动并等待 Ready（真实状态机 `workerPollOnce` + 真实 `runWorkerLoop` 同步启动/消费/Stopping 退出）、worker start 失败发布 Failed、平台 core-id 构建选择（host 不定义 PLATFORM、核 ID 参与 owner）、配置栈大小校验、code sharing OFF 非 owner 不取指针且不重入队 / ON 取同一指针、stale generation 请求被丢弃、编译中 generation 改变丢弃结果、**stale worker 不能清新 generation 的 dedup**、peer 析构不停 owner worker、owner shutdown 先 stop+join 再回 Uninitialized。**本轮（round-3）再增**：`SharedStateRequiresNoDynamicInitialization`（blob 平凡默认构造，无 .init_array）、真实 `runWorkerLoop` 在 Initializing 启动同一 worker yield → 存活到 Ready → 消费 → 空队 yield → Stopping 退出（`RealWorkerEntrySurvivesInitializingAndConsumes`，证明不因 spin budget 提前消失）、注册 fingerprint 一致 attach / 不一致 `FingerprintMismatch` clean fail。`std::thread/mutex/condition_variable`/STL 共享字段静态扫描在共享核源文件中为空。构建产物检查：`llvm-nm`/`llvm-readelf` 确认 `EJitCompileDriver.cpp.o` 无共享状态动态构造器/`.init_array`。

**本轮明确记录的平台前提：** ① 各核必须看到**同一共享 section、同 VA 或正确共享映射**（否则 fnPtr/原子地址语义不成立）；② `SRE_TaskDelete` 必须是**真 join**（软停+等退），否则靠 generation-aware dedup 兜底但仍可能 worker 回调触碰已销毁 driver；③ `EJIT_SRE_SHARED_CODE_POINTERS=ON` 需同 VA + seal 完成 + I/D cache 一致。

**尚需真实平台（aarch64_be 多核）验证：** ① 真实 `ejit_sre_current_core_id` 与跨核 owner election；② 共享 section 同 VA 映射 + cache coherent；③ `EJIT_SRE_SHARED_CODE_POINTERS=ON` 下跨核执行同一 fnPtr 的 I/D-cache 一致性与 seal 时序；④ host 单进程多 `EJit` 实例共享一份全局状态时，owner 用自身 ORC 为所有 producer 编译——真实平台为单一共享注册镜像，此语义才成立；⑤ 平台 `SRE_TaskCreate` 栈尺寸与 worker 长编译任务的匹配（实测栈峰值）；⑥ 平台 `SRE_TaskDelete` 必须是真 join。
