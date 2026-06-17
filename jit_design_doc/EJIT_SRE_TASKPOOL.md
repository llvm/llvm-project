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

目标 SRE 平台（aarch64_be）**没有 C++ 线程库**，因此全程禁用：

- `std::thread` / `std::async` / `std::future` / `std::promise`
- `std::mutex` / `std::shared_mutex` / `std::condition_variable`
- `<atomic>` / `<unordered_map>` / `<vector>` / `<string>` / `<functional>`

平台允许：
- `__atomic_*` 编译器内建（封装为 `EJitAtomic`）
- SRE 队列原语 `QueueCreate` / `QueueWrite` / `QueueRead`（可选，通过宏切换）

### 1.3 核心设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 异步模型 | producer + explicit poll worker | EJIT 不创建线程，worker 由外部平台驱动 |
| cache | 固定分桶 (32×8) | 零堆分配，可预测内存，替代 `unordered_map` |
| dedup | 固定分桶 (32×8) + 5 态状态机 | 同 key 只编译一次，CAS 状态机保证并发安全 |
| 队列 | Vyukov 无锁环形队列 (默认) / SRE 平台队列 | 默认可 host 测试，上板切换到平台原生队列 |
| 淘汰 | 无自动淘汰 | 固定数组做 LRU 不现实，桶满拒绝 + 显式 freeCode |
| 原子设施 | `EJitAtomic` wrapper | 集中封装 `__atomic_*`，后续可整体替换为平台内建桩 |

---

## 2. 架构总览

### 2.1 组件关系

```
EJitTaskPool
  ├── EJitSwitchController   → 使能位 + 模式(Off/Sync/Async) + 单调 version
  ├── EJitTaskPoolCache      → 固定分桶结果缓存 (32桶×8slot, 查/发布/释放)
  ├── EJitDedupTable         → 固定分桶去重表 (32桶×8slot, 5态状态机)
  ├── EJitQueue              → 异步工作队列 (Vyukov ring / SRE 平台队列)
  └── EJitTaskPoolCounters   → 无锁原子统计计数器
```

### 2.2 数据流

```
compile_or_get ──→ cache.lookup ──→ hit → return
                      │
                      ↓ miss
                  switch enabled?
                      │
                      ↓ yes
                  dedup.tryMarkPending
                      │
            ┌─────────┴─────────┐
            ↓ Sync               ↓ Async
       runCompile(当前栈)    queue.push(req)
       → cache.publish       → 立即返回 fallback
       → return JIT ptr          │
                           外部 worker:
                           pollOne() → runCompile → publish
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
            │                                                         ├─ cache.lookup()
            └─ miss:                                                  │   hit → return
                 decode cacheKey                                      │
                 load bitcode                                         ├─ dedup.tryMarkPending()
                 verify periods                                       │   AlreadyPending / DedupFull
                 OrcJIT compile                                       │   → return fallback
                   ↳ IR pipeline:                                     │
                     ① replace params                                ├─ Sync: runCompile(当前栈)
                     ② InstCombine                                   │   → cache.publish()
                     ③ StructFieldPass                               │   → return JIT ptr
                     ④ L1/L2/L3 opts                                 │
                 lookup symbol                                        └─ Async:
                 LRU cache.put()                                          queue.push()
                 return                                                   → return fallback
                                                                               │
                                                                          pollOne()
                                                                          → runCompile(worker核)
                                                                          → publish
```

核心差异：

| | Before | After |
|------|--------|-------|
| cache | LRU `unordered_map` + linked list | 固定分桶 `EJitTaskPoolCache`，32×8 |
| dedup | 无 | `EJitDedupTable`，同 key 只编一次 |
| 异步 | 无 | 入队 + 外部 poll worker，不创建线程 |
| 淘汰 | LRU 自动淘汰 | 无，桶满拒绝 + 显式 freeCode |
| 版本控制 | 无 | `SwitchController.version` |
| STL 依赖 | `<unordered_map>`, `<list>`, `<string>` | 无，仅 `EJitAtomic` |

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
  → miss + async                        → queue_.pop()
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

### 3.2 EJitSreQueue：请求结构 + 无锁队列

**位置**：`llvm/include/llvm/ExecutionEngine/EJIT/EJitSreQueue.h`、`.cpp`

#### 3.2.1 EJitCompileRequest（队列元素）

固定布局 POD，无构造/析构、无 STL，可安全通过平台队列按值传递：

```cpp
struct EJitCompileRequest {
    uint32_t funcIndex;      // 函数标识 (cacheKey 高 32 位)
    uint32_t version;        // SwitchController 代际 (入队时刻快照)
    uint64_t cacheKey;       // 完整特化 key (funcIdx | dims)
    uintptr_t fallbackPtr;   // AOT fallback 函数指针
    uintptr_t userData;      // 调用方透传 cookie (taskpool 核心不使用)
};
// aarch64: sizeof = 32 字节，alignof ≤ 8
```

#### 3.2.2 EJitQueue（有界 MPSC 队列）

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

## 4. Cache 与 Dedup：固定分桶结构

cache 和 dedup 使用完全相同的分桶结构。核心公式：

```cpp
bucket = funcIndex % EJIT_SRE_TASKPOOL_BUCKETS   // 默认 32
slot   = bucket * EJIT_SRE_TASKPOOL_BUCKET_SLOTS   // 每桶默认 8
```

### 4.1 EJitTaskPoolCache

```
entries_[256]: 扁平 C 数组，连续内存，零堆分配

  bucket 0               bucket 1                    bucket 31
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬     ┌──┬──┬──┬──┬──┬──┬──┬──┐
│s0│s1│s2│s3│s4│s5│s6│s7│s0│s1│ ...              ...   │s0│s1│s2│s3│s4│s5│s6│s7│
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴     ...         └──┴──┴──┴──┴──┴──┴──┴──┘
```

```cpp
struct EJitCacheEntry {
    EJitAtomicU32 state;      // Empty / Ready / Failed / Cancelled
    EJitAtomicU32 funcIndex;
    EJitAtomicU32 version;
    EJitAtomicU64 cacheKey;
    EJitAtomicUPtr fnPtr;     // JIT 编译出的函数指针
};
```

**lookup**：`funcIndex % 32` 定桶 → 桶内最多线性扫描 8 slot → 匹配 `(funcIndex, cacheKey, version)` 三元组。`state` 为 Ready 且身份全匹配才命中。

**publish**（写者只有单消费者——sync 调用栈 或 poll worker，二者不同时执行）：

1. 第一遍：桶内已有同 `(funcIndex, cacheKey)` 的 slot → 原地更新 fnPtr + version，不消耗新槽
2. 第二遍：找 Empty / Failed / Cancelled 状态的槽位 → 写入身份 + pointer → `state.storeRelease(Ready)` 最后发布
3. 8 个 slot 全是其他 key 的 Ready → 返回 false，绝不静默覆盖

**与 `std::unordered_map` 对比**：

| | unordered_map | 固定分桶 |
|------|--------------|---------|
| 内存 | 堆分配，per-node 独立 new | 编译期确定，连续 C 数组 |
| 查找 | hash → bucket → 链表 | `funcIndex%32` → 最多扫 8 slot |
| 上限 | 无（受内存限） | 硬上限 256 |
| 淘汰 | 手动/LRU | 仅同 key 复用 + 显式 freeCode |
| 依赖 | `<unordered_map>`, `<list>` | 仅 `EJitAtomic` |

### 4.2 EJitDedupTable

与 cache 完全相同的数据布局（32×8），不同的是每个 slot 的状态机：

```
Empty ──(CAS)──→ Claiming ──(release-store)──→ Pending ──(CAS)──→ Compiling ──(CAS)──→ Publishing ──(CAS)──→ Empty
                    ↑                               │                   │                   │
                    │                freeCode.cancel │    freeCode.cancel│    freeCode.cancel│
                    │                version 丢弃     │    version 丢弃    │                   │
                    └────────────────────────────────┴───────────────────┴───────────────────┘
```

5 个状态的语义：

| 状态 | 含义 | 谁拥有 |
|------|------|--------|
| Empty | 空闲 | — |
| Claiming | 正在写 identity（瞬态，不参与匹配） | producer 私有 |
| Pending | identity 已发布，等待编译 | producer → worker 交接 |
| Compiling | worker 取得所有权，编译中 | worker |
| Publishing | 编译完成，正在写 cache（瞬态） | worker |

**Claiming 的意义**：旧实现 CAS `Empty→Pending` 后再写 identity。多生产者下，B 可能看到 state 已非 Empty 但 funcIndex/cacheKey 还是 0，误判为新 key 又占了一个槽。新协议把"占槽"和"可被匹配"拆成两步：Claiming 是 producer 私有瞬态，`state.storeRelease(Pending)` 一次性发布 identity。读者 `state.loadAcquire()` 要么看不到（Claiming），要么看到完整 identity（Pending/Compiling/Publishing 三态之一）。

**关键接口**：

```cpp
EJitDedupResult tryMarkPending(funcIndex, cacheKey, version);  // 去重+占槽
bool markCompiling(funcIndex, cacheKey, version);              // Pending→Compiling
bool beginPublish(funcIndex, cacheKey, version);               // Compiling→Publishing
bool finishPublish(funcIndex, cacheKey, version);              // Publishing→Empty
bool cancel(funcIndex, cacheKey);                               // 强制回 Empty（freeCode）
void clear(funcIndex, cacheKey, version);                       // 回滚指定版本
```

slot 匹配比较三元组 `(funcIndex, cacheKey, version)`——不同函数哈希到同一桶各占独立 slot，互不误判。

---

## 5. 调度器逻辑

### 5.1 EJitSwitchController

```cpp
class EJitSwitchController {
    EJitAtomicU32 enabled_;   // 0/1
    EJitAtomicU32 mode_;      // Off=0 / Sync=1 / Async=2
    EJitAtomicU32 version_;   // 单调递增
};
```

- **version**：`bumpVersion()` 单调递增。入队请求携带当时 version；worker 在 runCompile 开始及编译后检查 version，不匹配则丢弃结果、清理 dedup、不 publish。cache lookup 也带 version，旧条目不命中（但物理槽位仍被占着）。
- **mode**：Sync 在调用栈直接编译；Async 入队后立即返回 fallback。Off 模式下只是不进 dedup/队列，仍可通过 `ejit_taskpool_sync_compile` 强制编译。

### 5.2 compileOrGet：统一入口

```cpp
CompileOrGetResult compileOrGet(funcIndex, cacheKey, fallback) {
    version = switch_.getVersion();

    // 1. cache hit (Ready 且 version 匹配)
    if (p = cache_.lookup(funcIndex, cacheKey, version))
        return {CacheHit, p};

    // 2. disabled / Off
    if (!switch_.isEnabled() || mode == Off)
        return {DisabledFallback, fallback};

    // 3. dedup 去重/占槽
    result = dedup_.tryMarkPending(funcIndex, cacheKey, version);
    if (result == AlreadyPending) return {AlreadyPending, fallback};
    if (result == DedupFull)     return {DedupFullFallback, fallback};

    // 4a. Sync: 当前栈编译 → publish → return
    if (mode == Sync)
        return runCompile(req, fromWorker=false);

    // 4b. Async: 入队 → 立即返回; 满则回滚 dedup
    if (!queue_.push(req)) {
        dedup_.clear(funcIndex, cacheKey, version);
        return {QueueFullFallback, fallback};
    }
    return {EnqueuedPending, fallback};
}
```

返回值状态全集：

| 状态 | fnPtr | 含义 |
|------|-------|------|
| CacheHit | JIT 函数指针 | 命中缓存 |
| SyncCompiled | JIT 函数指针 | 同步编译成功并已发布 |
| DisabledFallback | fallback | Off 模式 |
| EnqueuedPending | fallback | 异步入队成功，等待 worker 编译 |
| AlreadyPending | fallback | 同 key 已有人在编 |
| QueueFullFallback | fallback | 队列满，dedup 已回滚 |
| DedupFullFallback | fallback | dedup 桶满 |
| CacheFullFallback | fallback | 编译成功但 cache 桶满写不进 |
| CompileFailed | fallback | 编译失败/被取消/version 作废 |

### 5.3 runCompile：编译执行路径

sync 和 async 共用。核心流程 + 每次的 CAS 检查：

```cpp
void *runCompile(req, fromWorker) {
    // 1. version 检查：入队到执行期间 version 可能已变
    if (req.version != switch_.getVersion()) {
        dedup_.clear(...); → CompileFailed
    }

    // 2. Pending→Compiling (CAS)。freeCode 可能已 cancel，导致失败
    if (!dedup_.markCompiling(...)) → CompileFailed

    // 3. 实际编译（OrcJIT 引擎，同步在调用栈）
    ok = compileFn_(ctx, req, &fn);

    // 4. 编译后再次 version 检查
    if (req.version != switch_.getVersion()) {
        dedup_.cancel(...); → CompileFailed
    }

    if (!ok || !fn) {
        dedup_.clear(...); → CompileFailed
    }

    // 5. Commit gate 1: Compiling→Publishing (CAS)
    //    若 freeCode 在编译期间 cancel 过，此 CAS 失败
    if (!dedup_.beginPublish(...)) → CompileFailed (不写 cache)

    // 6. 写入 cache (可能返回 false = 桶满)
    published = cache_.publish(funcIndex, cacheKey, version, fn);
    if (!published) {
        dedup_.clear(...); → CacheFullFallback  // 不卡 pending
    }

    // 7. Commit gate 2: Publishing→Empty (CAS)
    //    若 freeCode 在 publish 窗口内把 slot 强制 Empty，
    //    此 CAS 失败 → 回滚刚写入的 cache 条目
    if (!dedup_.finishPublish(...)) {
        cache_.freeCode(funcIndex, cacheKey);  // 回滚
        → CompileFailed
    }

    → SyncCompiled (fromWorker=false) 或 async publish (fromWorker=true)
}
```

两个 CAS gate 保证了 **FreeCode 竞态安全**（见 §5.4）。

### 5.4 FreeCode：逻辑释放

```cpp
bool freeCode(funcIndex, cacheKey) {
    dedup_.cancel(funcIndex, cacheKey);    // 先把 in-flight slot 强制 Empty
    cache_.freeCode(funcIndex, cacheKey);  // 再清 cache Ready 条目
}
```

v1 是 **logical free**——只改状态位，不动物理 code pool。保证被取消的 in-flight compile 不会最终 publish：

```
     worker                            freeCode
        │                                  │
   beginPublish (Compiling→Publishing)     │         ← gate 1 先过了
        │                           cancel(Publishing→Empty) ← 冲掉
        │                                  │
   cache.publish() ← 瞬态可见               │
        │                                  │
   finishPublish (Publishing→Empty) → 失败 │         ← gate 2 拦住了
        │                                  │
   cache.freeCode() 回滚                    │
```

两个 CAS gate 把发布拆成 **Publishing 中间态 + 两道门**：

1. **gate 1** (`beginPublish`)：freeCode 在编译期间 cancel → CAS `Compiling→Publishing` 失败 → worker 直接丢弃
2. **gate 2** (`finishPublish`)：freeCode 在 publish 窗口内 cancel → CAS `Publishing→Empty` 失败 → worker 回滚 cache 条目

存在极短瞬态窗口（worker 已写入 cache、尚未回滚），期间并发读者可能命中该条目。v1 是 logical free、物理 code 仍有效，命中返回的是已下线的可执行代码——此行为已文档化且可接受。

---

## 6. 容量模型与约束

### 6.1 不是简单的 "256 个函数"

同一个 `ejit_entry` 函数的全部特化版本共享同一个 `funcIndex`，落在同一个桶：

```cpp
bucket = funcIndex % 32
cacheKey = (funcIndex << 32) | dim[0] | (dim[1]<<8) | (dim[2]<<16) | (dim[3]<<24)

funcIndex=5 的 jit_entry(cellIdx, trpIdx):
  cell=0,trp=0 → bucket 5  }
  cell=0,trp=1 → bucket 5  }  全挤在桶 5 的 8 个 slot
  ...
  cell=0,trp=7 → bucket 5  }
  cell=1,trp=0 → bucket 5  →  第 9 个 → CacheFullFallback！
```

### 6.2 各层约束

| 边界 | 值 | 受什么影响 |
|------|-----|-----------|
| 总槽位 | 256 | 32 × 8 |
| 单函数最大特化数 | ≤8 | 独占一桶时=8，碰撞时被稀释 |
| 碰桶时单函数上限 | <8 | 同桶其他函数占用后剩余 |
| 单 (funcIndex, cacheKey) | 1 | 同 key publish 原地覆盖 |
| 单函数同时在飞请求 | ≤8 | dedup 表同样 32×8 |

### 6.3 哈希碰撞

```
funcA: funcIndex=5  → 桶 5       funcB: funcIndex=37 → 桶 5  (碰撞)
如果 funcB 先占满 8 slot，funcA 一个都写不进 → 全部 CacheFullFallback
```

### 6.4 淘汰机制

**没有自动淘汰**。释放槽位仅三种途径：

| 途径 | 何时 | 操作 |
|------|------|------|
| 同 key 复用 | publish 第一遍扫到同 key | 原地覆盖 fnPtr |
| 显式 freeCode | 外部调用 | Ready→Empty |
| 回收到可回收槽 | publish 第二遍扫到 Empty/Failed/Cancelled | 覆盖写入 |

version bump 只让旧条目**逻辑不可见**，**不释放物理槽位**。需要调大容量：

```bash
-DEJIT_SRE_TASKPOOL_BUCKETS=64       # 减碰撞
-DEJIT_SRE_TASKPOOL_BUCKET_SLOTS=16  # 增每函数上限
```

当 `BUCKETS ≥ 函数总数` 且分布均匀时，每个函数真正独占一桶。

---

## 7. C ABI

6 个函数随 `libLLVMEJIT.a` 提供，不需要外部实现。仅在定义 `EJIT_SRE_TASKPOOL_PLATFORM_QUEUE` 时需外部提供 `QueueCreate`/`QueueWrite`/`QueueRead`。

```c
// 预热：强制同步编译，忽略 Off/Sync/Async 模式
ejit_status_t ejit_taskpool_sync_compile(uint32_t funcIndex,
                                         uint64_t cacheKey, void **outFn);

// 逻辑释放：取消 in-flight + 清 cache 条目（不动 code pool）
ejit_status_t ejit_taskpool_free_code(uint32_t funcIndex, uint64_t cacheKey);

// 驱动异步编译：从队列消费 1 个请求，当前栈编译。返回 1(干活)/0(空)
unsigned ejit_taskpool_poll_one(void);

// 同上，一次最多消费 maxItems 个。返回实际数量
unsigned ejit_taskpool_poll_budget(unsigned maxItems);

// 监控：in-flight slot 数（快照）
unsigned ejit_taskpool_pending_count(void);

// 监控：完整统计快照
ejit_status_t ejit_taskpool_get_stats(ejit_taskpool_stats_t *out);
```

**角色总结**：

```
监控面:  ejit_taskpool_get_stats / ejit_taskpool_pending_count

生产者侧 (业务)              消费者侧 (外部 worker 驱动)
ejit_compile_or_get (内部)   ejit_taskpool_poll_one       ← 单次消费
  → 命中 → 返回              ejit_taskpool_poll_budget    ← 批量消费
  → miss+async → 入队 → 返回   (均在调用者栈/核上执行)

预热/释放:
ejit_taskpool_sync_compile    ← 强制栈上编译
ejit_taskpool_free_code       ← 逻辑释放
```

**统计结构体**（独立于旧 LRU `ejit_stats_t`）：

```c
typedef struct {
    uint64_t cacheHits, syncCompiles, asyncCompiles, asyncEnqueues;
    uint64_t alreadyPending, queueFull, dedupFull;
    uint64_t compileFailed, publishFailed, freeCodeCalls;
    uint32_t readyEntries, pendingEntries, queueApproxSize;
    uint32_t reserved;
} ejit_taskpool_stats_t;
```

新增 status code（additive，旧值不变）：`EJIT_ERR_QUEUE_FULL`、`EJIT_ERR_DEDUP_FULL`、`EJIT_ERR_DISABLED`、`EJIT_PENDING`；cache 桶满复用 `EJIT_ERR_CACHE_FULL`。

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
| `compileOrGet` | enter / cacheHit / dedup / queuePush |
| `runCompile` | begin / compiled / published |
| `pollOne` | empty / dequeued |
| `freeCode` | enter |

---

## 9. 构建与测试

### 9.1 构建

```bash
./build.sh release aarch64_be --freestanding --sre-taskpool
```

| 开关 | 说明 | 默认 |
|------|------|------|
| `--sre-taskpool` | 开关 taskpool | OFF |
| `--sre-taskpool-buckets=<n>` | dedup/cache 桶数 | 32 |
| `--sre-taskpool-bucket-slots=<n>` | 每桶 slot 数 | 8 |
| `--sre-taskpool-queue-capacity=<n>` | 队列容量 (pow2) | 1024 |

对应 CMake option：`EJIT_SRE_TASKPOOL`、`EJIT_SRE_TASKPOOL_BUCKETS`、`EJIT_SRE_TASKPOOL_BUCKET_SLOTS`、`EJIT_SRE_TASKPOOL_QUEUE_CAPACITY`。

### 9.2 测试

```bash
cmake --build build-ejit-sre-taskpool --target check-ejit-taskpool -j8
```

`EJITTaskPoolTests` 编译 `EJitTaskPool.cpp` + `EJitSreQueue.cpp`（带 `EJIT_SRE_TASKPOOL_TESTING`），不依赖 `EJITTests`，host 可跑。使用 mock compiler + mock ring queue，**不使用真实线程**——并发交错全部通过显式 `pollOne`/`pollBudget` + 测试钩子模拟。覆盖 32 个用例：

- atomic wrapper、SwitchController、queue（容量/FIFO/满返回）
- dedup（Claiming 不被当成重复、5 态 transitions、cancel 阻断 finishPublish）
- cache（publish bucket full → `CacheFullFallback`、freeCode）
- sync/async 路径、FreeCode 竞态（编译中取消、publish 窗口内回滚）
- stats 计数、request flat POD 断言
