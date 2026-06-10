# EmbeddedJIT 运行时库设计文档 — 基于 OrcJIT + JITLink

**版本**: 1.1
**日期**: 2026-04-29
**关联**: SPEC4.md, PLAN4.md, PASS1–6 设计文档
**类型**: 运行时库 (libejit.a)
**核心框架**: LLVM OrcJIT + JITLink

---

## 1. 架构概述

### 1.1 设计目标

EmbeddedJIT 运行时库基于 LLVM OrcJIT + JITLink 构建，支持同步 (Sync) 和异步 (Async) 两种编译模式，并针对嵌入式环境的资源受限特点（Flash 存储、低 RAM）进行定制。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EmbeddedJIT 运行时库 (libejit.a)                      │
│                                                                             │
│  ┌──────────────────────┐    ┌──────────────────────┐                       │
│  │   C 语言 API 层        │    │   C++ 内部 API 层      │                       │
│  │   (EJitRuntime.h)     │    │   (EJit.h)            │                       │
│  │   ejit_init/shutdown  │    │   EJit class          │                       │
│  │   ejit_activate/      │    │   EJitCache           │                       │
│  │     deactivate        │    │                       │                       │
│  │   ejit_compile_or_get │    │                       │                       │
│  └──────────┬───────────┘    └───────────┬───────────┘                       │
│             │                            │                                    │
│  ┌──────────┴────────────────────────────┴──────────────────────────────┐   │
│  │                      EJitRuntimeCore (核心状态管理)                     │   │
│  │                                                                       │   │
│  │  ┌───────────────┐  ┌───────────────────────────┐                       │   │
│  │  │ PeriodArray    │  │ RuntimeState              │                       │   │
│  │  │ Registry       │  │ (activate/deactivate)     │                       │   │
│  │  └───────────────┘  └───────────────────────────┘                       │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                    │                                          │
│  ┌─────────────────────────────────┴────────────────────────────────────┐   │
│  │                    EJitOrcEngine (OrcJIT + JITLink 封装)               │   │
│  │                                                                       │   │
│  │  ┌─────────────┐  ┌──────────────────┐  ┌────────────────────────┐  │   │
│  │  │ LLJIT-based │  │ IRTransformLayer │  │ EJitJITLinkMemoryMgr   │  │   │
│  │  │ Engine      │  │ (EJitStructField │  │ (嵌入式内存管理)         │  │   │
│  │  │             │  │  IR Transform    │  │                        │  │   │
│  │  └─────────────┘  └──────────────────┘  └────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                          │
│  ┌─────────────────────────────────┴────────────────────────────────────┐   │
│  │                     同步/异步编译隔离层                                │   │
│  │                                                                       │   │
│  │  ┌─────────────────────────┐    ┌───────────────────────────────┐    │   │
│  │  │ EJitSyncCompiler        │    │ EJitAsyncCompiler              │    │   │
│  │  │ (调用线程直接编译)        │    │ (后台线程 + 请求队列 + 隔离引擎) │    │   │
│  │  └─────────────────────────┘    └───────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                          │
│  ┌─────────────────────────────────┴────────────────────────────────────┐   │
│  │                      EJitCache (Code Cache)                           │   │
│  │  ┌──────────────────────┐  ┌────────────────────────────────────┐    │   │
│  │  uint64_t Cache Key    │  │ LRU 淘汰 + 大小限制                │    │   │
│  │  funcIdx|dim[3..0]   │  │ (maxCacheSize, maxEntries)         │    │   │
│  │  └──────────────────────┘  └────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 选择 OrcJIT + JITLink 的理由

| 考量 | MCJIT (旧) | OrcJIT + JITLink (新) |
|------|-----------|---------------------|
| 内存管理 | 固定 SectionMemoryManager | JITLinkMemoryManager 完全可覆盖 |
| 并发支持 | 无内置线程池 | ExecutionSession + NumCompileThreads |
| 模块隔离 | 弱 | JITDylib + ResourceTracker 精确控制 |
| 嵌入式适配 | 困难 | 自定义 allocator + BasicLayout |
| LLVM 演进 | Legacy (维护模式) | 当前主力 |
| IR 变换 | 无内置 | IRTransformLayer 天然支持 |

---

## 2. 核心组件设计

### 2.1 EJitOrcEngine — LLJIT 封装

EJitOrcEngine 封装 LLJIT 实例，管理 JIT 编译生命周期。

```cpp
// llvm/lib/ExecutionEngine/EJIT/EJitOrcEngine.h

namespace llvm::ejit {

class EJitOrcEngine {
    LLVMContext Ctx;                       // 拥有者 LLVMContext (用于 bitcode 加载)
    std::unique_ptr<orc::LLJIT> J;         // OrcJIT 实例
    orc::JITDylib* MainJD;                // 主 JITDylib 引用
    orc::ResourceTrackerSP DefaultRT;      // 默认 ResourceTracker (用于模块管理)

    EJitJITLinkMemoryManager* MemMgr;      // 我们的嵌入式内存管理器

    // JIT 编译上下文 — 每次编译前由 SyncCompiler/AsyncCompiler 设置
    // 同步模式下在调用线程设置；异步模式下在后台线程设置，无竞态
    SpecializationContext* ActiveCtx = nullptr;
    OptimizationLevel OptLevel = OptimizationLevel::Level2;

public:
    // 工厂方法: 创建 EJitOrcEngine
    static Expected<std::unique_ptr<EJitOrcEngine>>
    Create(const EJitConfig& config);

    // 错误信息 (创建失败时)
    static std::string getLastError() { return lastError_; }
    static std::string lastError_;

    // 加载 Bitcode Module 并注册到 JITDylib
    Error loadBitcodeModule(StringRef bitcodeData,
                            StringRef funcName);

    // 查询已编译符号
    Expected<void*> lookup(StringRef funcName);

    // 移除模块 (释放内存)
    Error removeModule(orc::ResourceTrackerSP RT);

    // 获取内存统计
    size_t getCurrentCodeSize() const;
    size_t getTotalAllocatedMemory() const;
};

} // namespace llvm::ejit
```

### 2.1.1 LLJIT 创建流程

```cpp
Expected<std::unique_ptr<EJitOrcEngine>>
EJitOrcEngine::Create(const EJitConfig& config) {
    auto engine = std::make_unique<EJitOrcEngine>();

    // 注意: 使用 Expected<> 返回错误，不在嵌入式场景调用 cantFail/abort

    // 步骤 1: 创建嵌入式 JITLinkMemoryManager
    auto memMgr = std::make_unique<EJitJITLinkMemoryManager>(
        config.maxCodeMemory,      // 最大代码内存 (默认 512KB)
        config.pageSize             // 页大小 (ARM/AArch64 通常 4KB)
    );
    engine->MemMgr = memMgr.get();

    // 步骤 2: 配置 JITTargetMachineBuilder
    auto JTMBOrErr = JITTargetMachineBuilder::detectHost();
    if (!JTMBOrErr) {
        return JTMBOrErr.takeError();
    }
    auto JTMB = std::move(*JTMBOrErr);

    // 步骤 3: 构建 LLJIT
    //   - 使用自定义 ObjectLinkingLayerCreator 注入嵌入式内存管理器
    //   - 设置 compile threads 为 0 (同步) 或 1+ (异步)
    //   - IRTransformLayer 注入 EJitStructFieldPass
    orc::LLJITBuilder Builder;

    Builder.setJITTargetMachineBuilder(std::move(JTMB));

    // 注入嵌入式内存管理器
    Builder.setObjectLinkingLayerCreator(
        [&](orc::ExecutionSession& ES)
            -> Expected<std::unique_ptr<orc::ObjectLayer>> {
            return std::make_unique<orc::ObjectLinkingLayer>(
                ES, *engine->MemMgr);
        });

    // 编译线程配置
    if (config.compileMode == CompileMode::Sync) {
        // 同步模式: 0 编译线程 = 在调用线程上编译
        Builder.setNumCompileThreads(0);
    } else {
        // 异步模式: 1 个编译线程 (嵌入式场景资源受限)
        Builder.setNumCompileThreads(1);
    }

    // 创建 LLJIT 实例
    auto JOrErr = Builder.create();
    if (!JOrErr) {
        return JOrErr.takeError();
    }
    engine->J = std::move(*JOrErr);
    engine->MainJD = &engine->J->getMainJITDylib();
    engine->DefaultRT = engine->MainJD->getDefaultResourceTracker();

    // 步骤 4: 注册 IRTransformLayer 回调
    // 完整的 JIT Pipeline: 参数替换 → InstCombine → Inline → PASS6 → 标准优化 (详见 §2.4)
    // 注意: IRTransformLayer::TransformFunction 签名为
    //   Expected<ThreadSafeModule>(ThreadSafeModule, MaterializationResponsibility&)
    // withModuleDo 的回调签名为 Expected<Error>(Module&)，原地修改 Module
    engine->J->getIRTransformLayer().setTransform(
        [engine](orc::ThreadSafeModule TSM,
                 orc::MaterializationResponsibility& R)
            -> Expected<orc::ThreadSafeModule> {
            Error Err = TSM.withModuleDo([engine](Module& M) -> Error {
                if (!engine->ActiveCtx)
                    return Error::success();

                // Step 1: 参数预处理 (替换 ejit_period_arr_ind 为常量)
                preReplacePeriodIndices(M, engine->ActiveCtx);

                // Step 2: InstCombine (折叠常量链)
                runInstCombine(M);

                // Step 3: Inline (展开跨函数 GEP 链)
                runInline(M);

                // Step 4: EJitStructFieldPass (may_const load → 常量)
                EJitStructFieldPass SFPass;
                SFPass.setSpecializationContext(engine->ActiveCtx);
                ModuleAnalysisManager MAM;
                SFPass.run(M, MAM);

                // Step 5: 标准优化 (按 OptLevel 编排)
                runOptimizationPipeline(M, engine->OptLevel);

                return Error::success();
            });
            if (Err)
                return std::move(Err);
            return std::move(TSM);
        });

    // 步骤 5: 注册 Process Symbols JITDylib
    // 使 JIT 代码可调用 AOT 编译的外部函数
    if (auto Err = engine->J->getProcessSymbolsJITDylib()) {
        // 外部符号在此 JITDylib 中解析
    }

    return engine;
}
```

---

## 2.2 EJitJITLinkMemoryManager — 嵌入式内存管理器

### 2.2.1 设计目标

| 约束 | 规格 |
|------|------|
| 最大代码区 | 512KB (可配置) |
| 页大小 | 4KB (ARM Cortex-A / AArch64) |
| 分配策略 | 固定大小的 slab 分配器 |
| 回收 | 在 deallocate 时调用 `munmap` 等效操作 |
| 并发安全 | 需要 mutex (异步编译时后台线程访问) |

### 2.2.2 实现

```cpp
// llvm/lib/ExecutionEngine/EJIT/EJitJITLinkMemoryManager.h

namespace llvm::ejit {

class EJitJITLinkMemoryManager : public jitlink::JITLinkMemoryManager {
public:
    EJitJITLinkMemoryManager(size_t maxTotalSize, uint64_t pageSize = 4096);

    // 分配: 解析 LinkGraph → 分配 Code + Data 段 → 返回
    void allocate(const jitlink::JITDylib* JD,
                  jitlink::LinkGraph& G,
                  OnAllocatedFunction OnAllocated) override;

    // 释放: 归还已 finalize 的分配
    void deallocate(std::vector<FinalizedAlloc> Allocs,
                    OnDeallocatedFunction OnDeallocated) override;

    // 统计信息
    size_t getCurrentUsage() const;
    size_t getMaxUsage() const;
    size_t getAllocationCount() const;

private:
    // 嵌入式 slab 分配器
    // 预分配一块连续内存 (如 512KB)，在其中分配固定大小段

    struct SlabRegion {
        void* baseAddr;              // slab 基地址
        size_t totalSize;            // 总大小
        size_t usedSize;             // 已使用大小
        std::mutex allocMutex;       // 分配锁
    };

    SlabRegion codeSlab_;            // 代码 slab (RX)
    SlabRegion dataSlab_;            // 数据 slab (RW)

    struct Allocation {
        FinalizedAlloc alloc;
        size_t size;
        uint64_t slabOffset;
    };
    std::vector<Allocation> activeAllocs_;

    // 使用 JITLink 的 BasicLayout 辅助划分段
    struct SegmentAlloc {
        orc::ExecutorAddr addr;
        size_t workingSize;
        size_t targetSize;
    };
    std::vector<SegmentAlloc> allocateSegments(LinkGraph& G);
    void applySegments(LinkGraph& G, const std::vector<SegmentAlloc>& segs);
};
```

```cpp
// allocate 实现
void EJitJITLinkMemoryManager::allocate(
    const jitlink::JITDylib* JD,
    jitlink::LinkGraph& G,
    OnAllocatedFunction OnAllocated) {

    // 步骤 1: 使用 BasicLayout 分析段布局
    BasicLayout BL(G);

    // 步骤 2: 为每个 AllocGroup 分配内存
    // AllocGroup 按 (MemProt, MemLifetime) 分类
    //   类型: ReadOnly (RO), ReadWrite (RW), ReadExec (RX)
    //   Lifetime: Standard, Finalize

    for (auto& KV : BL.segments()) {
        auto& AG = KV.first;       // AllocGroup (描述保护/生命周期)
        auto& Segs = KV.second;    // Segment 列表

        for (auto& Seg : Segs) {
            size_t allocSize = Seg.ContentSize + Seg.ZeroFillSize;

            // 选择 slab: code → RX, data → RW
            SlabRegion* slab = nullptr;
            if (AG.getMemProt() == orc::MemProt::Read ||
                AG.getMemProt() == orc::MemProt::ReadWrite) {
                slab = &dataSlab_;
            } else if (AG.getMemProt() == orc::MemProt::ReadExec) {
                slab = &codeSlab_;
            }

            if (!slab) {
                OnAllocated(nullptr);
                return;
            }

            // 从 slab 分配
            std::lock_guard<std::mutex> lock(slab->allocMutex);

            if (slab->usedSize + allocSize > slab->totalSize) {
                // OOM: 触发 Code Cache 淘汰
                OnAllocated(make_error<StringError>(
                    "EJIT: Code memory exhausted",
                    inconvertibleErrorCode()));
                return;
            }

            uintptr_t allocAddr = (uintptr_t)slab->baseAddr + slab->usedSize;
            slab->usedSize += allocSize;

            // 设置内存保护
            applyMemoryProtection(allocAddr, allocSize, AG.getMemProt());

            Seg.Addr = orc::ExecutorAddr(allocAddr);
            Seg.WorkingMem = MutableArrayRef<char>(
                (char*)allocAddr, Seg.ContentSize);
        }
    }

    // 步骤 3: 应用布局到 LinkGraph
    BL.apply();

    // 步骤 4: 创建 InFlightAlloc 并传给 callback
    auto FA = std::make_unique<EJitInFlightAlloc>(...);
    OnAllocated(std::move(FA));
}
```

### 2.2.3 嵌入式优化: 固定 slab 分配

```
嵌入式 Code Cache 内存布局 (2MB 示例):

┌──────────────────────────────────────────────────────────────┐
│                      Code Slab (RX) — 2MB                   │
│  ┌──────────┬──────────┬──────────┬───────────────────────┐  │
│  │ func_1   │ func_2   │ func_3   │ ... (free)            │  │
│  │ (48KB)   │ (32KB)   │ (56KB)   │                       │  │
│  └──────────┴──────────┴──────────┴───────────────────────┘  │
├──────────────────────────────────────────────────────────────┤
│                      Data Slab (RW) — 128KB                   │
│  ┌──────────┬──────────┬───────────────────────────────────┐ │
│  │ data_1   │ data_2   │ ... (free)                        │ │
│  └──────────┴──────────┴───────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘

分配策略:
- Code Slab: 按顺序分配 (bump allocator), 不支持单独释放
  → deallocate 只在整个 slab 重置时生效 (clearCache)
- Data Slab: 同上

LRU 淘汰时:
- 当 slab 使用率超过阈值 → evictLRU()
- 重置整个 slab → 重新编译活跃函数
- 或者: 使用更好的 allocator 支持碎片回收 (后续优化)
```

---

## 2.3 同步/异步编译隔离

### 2.3.1 隔离架构

```
                         ┌─────────────┐
                         │ 调用线程      │
                         │ (用户代码)    │
                         └──────┬──────┘
                                │
                    ejit_compile_or_get()
                                │
                    ┌───────────┴───────────┐
                    │  EJitCompileDriver    │
                    │  (编译调度器)           │
                    └───────────┬───────────┘
                                │
              ┌─────────────────┴─────────────────┐
              │                                   │
     ┌────────┴────────┐                 ┌───────┴──────────┐
     │ Sync Path       │                 │ Async Path        │
     │                 │                 │                   │
     │ 调用线程执行:     │                 │ 提交到:            │
     │   loadModule    │                 │   AsyncCompiler   │
     │   runPasses     │                 │   (后台线程)        │
     │   codeGen       │                 │       │           │
     │   cache.put     │                 │   loadModule      │
     │   return pfn    │                 │   runPasses       │
     │                 │                 │   codeGen         │
     │  阻塞直到完成     │                 │   cache.put       │
     │                 │                 │                   │
     │  返回: pfn/NULL │                 │  返回: NULL       │
     └─────────────────┘                 └───────────────────┘
```

### 2.3.2 同步编译器

```cpp
// 同步编译器: 在调用线程上执行完整编译流程
class EJitSyncCompiler {
public:
    struct Result {
        void* funcPtr;               // 特化函数指针 (NULL 表示失败)
        size_t compileTimeMs;        // 编译耗时
        size_t codeSize;             // 代码大小
    };

    Result compile(EJitOrcEngine& engine,
                   const std::string& bitcodeData,
                   const SpecializationContext& ctx) {
        Result result = {nullptr, 0, 0};
        auto startTime = std::chrono::steady_clock::now();

        // Step 1: 设置编译上下文 (供 IRTransformLayer 回调读取)
        engine.ActiveCtx = &ctx;
        engine.OptLevel = ctx.optLevel;

        // Step 2: 加载 bitcode module 到 LLJIT (使用 uint64_t cacheKey)
        if (auto Err = engine.loadBitcodeModule(bitcodeData, ctx.cacheKey, ctx.fnName)) {
            engine.ActiveCtx = nullptr;
            return result;
        }

        // Step 3: lookup 触发 materialization
        auto addr = engine.lookup(ctx.cacheKey, ctx.fnName);
        engine.ActiveCtx = nullptr;  // 清理上下文
        if (!addr) {
            return result;
        }

        result.funcPtr = addr->toPtr<void*>();
        result.codeSize = engine.getModuleCodeSize(ctx.fnName);

        auto endTime = std::chrono::steady_clock::now();
        result.compileTimeMs = std::chrono::duration_cast<
            std::chrono::milliseconds>(endTime - startTime).count();

        return result;
    }
};
```

### 2.3.3 异步编译器

```cpp
// 异步编译器: 后台线程 + 请求队列 + 隔离引擎实例
class EJitAsyncCompiler {
public:
    EJitAsyncCompiler(EJitConfig& config, EJitCache& cache,
                       EJitRuntimeState& runtimeState);
    ~EJitAsyncCompiler();

    // 启动后台线程
    void start();

    // 停止后台线程 (等待当前编译完成)
    void stop();

    // 提交异步编译请求 (非阻塞)
    // 若相同 cacheKey 已有编译在进行中，直接忽略 (dedup)
    void submitRequest(CompileRequest req);

private:
    // 后台线程主循环
    void workerLoop();

    // 执行单次编译 (在后台线程中)
    void compileOne(const CompileRequest& req);

    // === 线程安全隔离 ===

    // 后台线程拥有独立的 LLVM 资源:
    std::unique_ptr<EJitOrcEngine> workerEngine_;  // 独立 LLJIT 实例
    // 注意: workerEngine_ 有自己的 LLVMContext, TargetMachine
    // 与用户调用线程完全隔离，避免 LLVMContext 竞态

    // 共享的只读数据 (线程安全):
    EJitConfig& config_;           // 编译配置 (const 访问)
    EJitCache& cache_;            // Code Cache (内部有 mutex)
    EJitRuntimeState& runtimeState_; // 时间窗激活状态 (内部有 mutex)

    // 线程控制
    std::thread workerThread_;
    std::queue<CompileRequest> requestQueue_;
    std::mutex queueMutex_;
    std::condition_variable queueCV_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stopping_{false};

    // 正在编译的请求集合 (按 uint64_t cacheKey)
    // 防止同一 cacheKey 重复提交编译请求
    std::set<uint32_t> requestsInFlight_;
    std::mutex inFlightMutex_;
};

// 编译请求 (v1.7: cacheKey 在 ctx 内, 去掉冗余字段)
struct CompileRequest {
    std::string funcName;
    std::string bitcodeData;
    SpecializationContext ctx;         // 含 uint64_t cacheKey
    uint64_t timestamp;
};
```

```cpp
void EJitAsyncCompiler::submitRequest(CompileRequest req) {
    {
        std::lock_guard<std::mutex> lock(inFlightMutex_);
        // 去重: 相同 cacheKey (uint64_t) 已有编译在进行中，跳过
        if (requestsInFlight_.count(req.ctx.cacheKey)) {
            return;
        }
        requestsInFlight_.insert(req.ctx.cacheKey);
    }

    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        requestQueue_.push(std::move(req));
    }
    queueCV_.notify_one();
}

void EJitAsyncCompiler::workerLoop() {
    while (!stopping_.load()) {
        CompileRequest req;

        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            queueCV_.wait(lock, [this] {
                return !requestQueue_.empty() || stopping_.load();
            });

            if (stopping_.load()) break;
            if (requestQueue_.empty()) continue;

            req = std::move(requestQueue_.front());
            requestQueue_.pop();
        }

        // 在后台线程编译 (使用隔离的 Engine 实例)
        compileOne(req);
    }

    // 处理完剩余请求
    while (true) {
        CompileRequest req;
        {
            std::lock_guard<std::mutex> lock(queueMutex_);
            if (requestQueue_.empty()) break;
            req = std::move(requestQueue_.front());
            requestQueue_.pop();
        }
        compileOne(req);
    }
}

void EJitAsyncCompiler::compileOne(const CompileRequest& req) {
    // 异步安全：编译前重新检查时间窗激活状态
    // getOrCompile() 中的 isPeriodActive 检查与 compileOne() 实际执行之间
    // 存在时间窗，用户线程可能在此期间调用了 deactivate()。
    // 若时间窗已失效，跳过编译并清理 in-flight 记录。
    if (!runtimeState_->isPeriodActive(req.ctx)) {
        std::lock_guard<std::mutex> lock(inFlightMutex_);
        requestsInFlight_.erase(req.ctx.cacheKey);
        return;
    }

    // 内存序保证：在读取 may_const 字段值之前获取同步屏障
    // 确保用户线程在 activate() 中对全局变量的写入对编译线程可见
    // （与 RuntimeState::activate 中的 release 屏障配对）
    std::atomic_thread_fence(std::memory_order_acquire);

    // 设置编译上下文 (供 IRTransformLayer 回调读取)
    workerEngine_->ActiveCtx = &req.ctx;
    workerEngine_->OptLevel = req.ctx.optLevel;

    EJitSyncCompiler syncCompiler;
    auto result = syncCompiler.compile(*workerEngine_,
                                       req.bitcodeData,
                                       req.ctx,
                                       req.ctx.cacheKey);

    workerEngine_->ActiveCtx = nullptr;  // 清理上下文

    if (result.funcPtr) {
        // 编译成功: 存入 Cache (内部 mutex 保护)
        cache_.put(req.ctx.cacheKey, result.funcPtr, result.codeSize);
    }
    // 编译失败: 不存缓存, 后续调用 retry 或 fallback

    // 编译完成 (成功或失败), 从 in-flight 集合移除
    {
        std::lock_guard<std::mutex> lock(inFlightMutex_);
        requestsInFlight_.erase(req.ctx.cacheKey);
    }
}
```

### 2.3.4 隔离级别总结

| 资源 | 同步模式 | 异步模式 |
|------|---------|---------|
| LLVMContext | 调用线程拥有 (单实例) | 后台线程拥有独立实例 |
| LLJIT (ExecutionSession) | 调用线程使用 | 后台线程拥有独立 LLJIT |
| TargetMachine | 调用线程使用 | 后台线程独立创建 |
| MemoryManager (JITLink) | 调用线程使用 | **独立 slab**（同步/异步引擎各自拥有 JITLinkMemoryManager，避免 bump-allocator 空间竞争） |
| Code Cache | 调用线程访问 | 共享 (mutex 保护) |
| PeriodArrayRegistry | 调用线程更新 (activate/deactivate) | 只读快照 |

---

## 2.4 IRTransformLayer — EJitStructFieldPass 集成

### 2.4.1 Transform 回调

Transform 回调在 §2.1.1 `EJitOrcEngine::Create()` 中注册为内联 lambda，直接捕获 `engine` 指针读取 `ActiveCtx` 和 `OptLevel`。不需要插件类：

```cpp
// §2.1.1 中的注册代码 (简化引用)
engine->J->getIRTransformLayer().setTransform(
    [engine](orc::ThreadSafeModule TSM, ...) -> Expected<orc::ThreadSafeModule> {
        Error Err = TSM.withModuleDo([engine](Module& M) -> Error {
            if (!engine->ActiveCtx)
                return Error::success();

            preReplacePeriodIndices(M, engine->ActiveCtx);
            runInstCombine(M);
            runInline(M);

            EJitStructFieldPass SFPass;
            SFPass.setSpecializationContext(engine->ActiveCtx);
            ModuleAnalysisManager MAM;
            SFPass.run(M, MAM);

            runOptimizationPipeline(M, engine->OptLevel);
            return Error::success();
        });
        if (Err)
            return std::move(Err);
        return std::move(TSM);
    });
```

`preReplacePeriodIndices`、`runInstCombine`、`runInline`、`runOptimizationPipeline` 为同一编译单元内的静态辅助函数（详见 §2.4.2, §2.4.3）。
```

### 2.4.2 JIT Pipeline 各阶段实现

**`preReplacePeriodIndices`** — 将 `ejit_period_arr_ind` 参数替换为运行时常量值：

```cpp
void preReplacePeriodIndices(Module& M, SpecializationContext* ctx) {
    Function* F = M.getFunction(ctx->fnName);
    if (!F) return;

    for (int i = 0; i < ctx->period_count; ++i) {
        // 从函数 metadata 查找 periodName 对应的参数索引
        int argIdx = findPeriodArrIndArg(F, ctx->dimensions[i].periodName);
        if (argIdx < 0) continue;

        Argument* arg = F->getArg(argIdx);
        Constant* constVal = ConstantInt::get(arg->getType(),
                                               ctx->dimensions[i].cellIdx);
        arg->replaceAllUsesWith(constVal);
    }
}
```

**`runInstCombine`** — 在参数替换后传播常量，折叠初始分支：

```cpp
void runInstCombine(Module& M) {
    FunctionPassManager FPM;
    FPM.addPass(InstCombinePass());
    FunctionAnalysisManager FAM;
    // 为每个函数运行 InstCombine
    for (Function& F : M) {
        if (!F.isDeclaration())
            FPM.run(F, FAM);
    }
}
```

**`runInline`** — 内联 callee，使跨函数的 may_const load 暴露给 PASS6：

```cpp
void runInline(Module& M) {
    // 使用 LLVM 默认 Inline 阈值
    ModuleAnalysisManager MAM;
    ModulePassManager MPM;
    MPM.addPass(InlinerPass());
    MPM.run(M, MAM);
}
```

### 2.4.3 优化 Pipeline (L1/L2/L3) — PASS6 之后

```cpp
void runOptimizationPipeline(Module& M, OptimizationLevel level) {
    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;

    PassBuilder PB;

    // 注册 analysis managers
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    ModulePassManager MPM;

    // L1: 基础优化 (全等级执行)
    MPM.addPass(SCCPPass());            // 稀疏条件常量传播
    MPM.addPass(ADCEPass());            // 激进死代码消除
    MPM.addPass(SimplifyCFGPass());     // CFG 简化 (分支折叠)

    // L2: 中等优化 (二次内联)
    // 注意：首次 Inline 已在 PASS6 之前完成（见 §2.4.1 Transform 函数）
    // 此处的 Inline 用于内联 PASS6 常量替换后新暴露的调用机会
    if (level >= OptimizationLevel::Level2) {
        MPM.addPass(InlinerPass());     // 二次函数内联
        MPM.addPass(SimplifyCFGPass()); // 内联后再次简化 CFG
    }

    // L3: 激进优化
    if (level >= OptimizationLevel::Level3) {
        MPM.addPass(LoopUnrollPass(    // 循环展开
            LoopUnrollOptions()
                .setPartial(false)      // 不全展开
                .setPeeling(false)      // 不 peeling
                .setRuntime(false)      // 不运行时展开
                .setUpperBound(true)    // 使用上限
                .setThreshold(50)       // 展开阈值
        ));
        MPM.addPass(SimplifyCFGPass()); // 展开后 CFG 简化
    }

    // 运行
    MPM.run(M, MAM);
}
```

---

## 2.5 EJitCompileDriver — 编译调度器

编译调度器统一同步/异步编译路径，调用方不感知内部实现。

```cpp
// 编译调度器
class EJitCompileDriver {
public:
    EJitCompileDriver(EJitConfig& config,
                      EJitCache& cache,
                      PeriodArrayRegistry& periodReg);

    // 统一入口: 获取或编译特化函数
    // 返回 NULL 表示需要 fallback
    void* getOrCompile(uint32_t funcIdx,
                       uint64_t cacheKey,
                       int count);

private:
    EJitConfig& config_;
    EJitCache& cache_;
    PeriodArrayRegistry& periodReg_;

    // 编译引擎 (同步/异步共用或独立)
    std::unique_ptr<EJitSyncCompiler> syncCompiler_;
    std::unique_ptr<EJitAsyncCompiler> asyncCompiler_;

    // Bitcode 数据缓存 (funcName → bitcode bytes)
    // AOT 嵌入的 bitcode 在 ejit_init 时加载到此缓存
    std::unordered_map<std::string, std::string> bitcodeCache_;

    // 构建 Cache Key (v1.8: uint64_t)
    // funcIdx(32b) | dim[0](8b) | dim[1](8b) | dim[2](8b) | dim[3](8b)
    uint64_t buildCacheKey(uint32_t funcIdx, uint64_t cacheKey);

    // 构建 SpecializationContext
    SpecializationContext buildContext(const std::string& funcName,
                                       uint64_t cacheKey);
};
```

```cpp
void* EJitCompileDriver::getOrCompile(uint32_t funcIdx,
                                       uint64_t cacheKey,
                                       int count) {
    // Step 1: 构建 Cache key (funcIdx from wrapper, 无字符串开销)
    uint32_t funcIdx = hashFuncName(funcName)  // deterministic, zero map lookup;
    uint64_t cacheKey = EJitCache::buildCacheKey(funcIdx, dims, count);

    // Step 2: 查 Cache
    if (void* cached = cache_.getOrNull(cacheKey)) {
        return cached;  // 命中 → 直接返回
    }

    // Step 3: 验证时间窗状态
    SpecializationContext ctx = buildContext(funcName, dims, count);
    if (!isPeriodActive(ctx)) {
        return nullptr; // 未激活 → fallback
    }

    // Step 4: 获取 Bitcode
    auto it = bitcodeCache_.find(funcName);
    if (it == bitcodeCache_.end()) {
        return nullptr; // 无 bitcode → fallback
    }

    // Step 5: 编译 (同步/异步)
    if (config_.compileMode == CompileMode::Sync) {
        auto result = syncCompiler_->compile(
            *syncEngine_, it->second, ctx, cacheKey);
        if (result.funcPtr) {
            cache_.put(cacheKey, result.funcPtr, result.codeSize);
        }
        return result.funcPtr;  // 同步: 立即返回结果
    } else {
        // 异步: 提交编译请求 → 立即返回 NULL
        CompileRequest req;
        req.funcName = funcName;
        req.ctx.cacheKey = cacheKey;
        req.bitcodeData = it->second;
        req.ctx = std::move(ctx);
        asyncCompiler_->submitRequest(std::move(req));
        return nullptr;  // 异步: 下次调用再查 Cache
    }
}
```

---

## 2.6 EJitCache — Code Cache

```cpp
// Code Cache 管理器 — iterator-embedded LRU (单 hash 查找完成 LRU bump)
class EJitCache {
public:
    using LruList = std::list<uint64_t>;

    struct Entry {
        void* funcPtr;
        size_t codeSize;
        LruList::iterator lruIt;          // embedded → O(1) splice/erase
        SmallVector<std::string, 4> periodDeps;
    };

    EJitCache(size_t maxEntries = 4096,
              size_t maxTotalSize = 32 * 1024 * 1024,
              size_t maxSingleFuncSize = 512 * 1024);

    // 查询: unique_lock (splice 修改链表指针，不可 shared)
    void* getOrNull(uint64_t cacheKey);

    // 存入
    bool put(uint64_t cacheKey, void* funcPtr, size_t codeSize,
             ArrayRef<std::string> periodDeps = {});

    // 时间窗失效 → 清理依赖缓存 (periodIndex_ 索引)
    void invalidateByPeriod(const std::string& periodName, uint8_t cellIdx);

    void clear();
    Stats getStats() const;

    static uint64_t buildCacheKey(uint32_t funcIdx,
        const std::pair<std::string, uint8_t>* dims, unsigned count);

private:
    void evictLRU();

    mutable MutexType mutex_;               // BareMetalMutex or shared_mutex
    std::unordered_map<uint64_t, Entry> cache_;
    LruList lruList_;                       // uint64_t key, LRU order
    std::unordered_map<std::string, std::set<uint64_t>> periodIndex_;

    size_t maxEntries_;
    size_t maxTotalSize_;
    size_t maxSingleFuncSize_;
    size_t currentTotalSize_ = 0;
};
```

---

## 2.7 运行时初始化

### 2.7.1 EJitRegistrationStore — 全局注册暂存区

`EJitRegistrationStore` 是一个进程级全局单例，解决 `ejit_auto_register` (constructor 时执行) 与 `ejit_init` (用户 main 中调用) 之间的数据传递问题。

**问题**: `ejit_auto_register()` 通过 `@llvm.global_ctors` (优先级 65535) 在 `main()` 之前执行，调用 `ejit_register_period_array()`、`ejit_register_bitcode()` 等注册函数。但此时 `ejit_init()` 尚未调用，正式的 `PeriodArrayRegistry`、`BitcodeTracker` 等数据结构还不存在。若注册函数直接操作这些未创建的对象，会导致空指针或数据丢失。

**方案**: 注册函数统一写入全局单例 `EJitRegistrationStore`；`ejit_init()` 启动时从单例 `consume()` 取出所有暂存数据，填充正式 Registry，然后清空单例。

```cpp
// llvm/lib/ExecutionEngine/EJIT/EJitRegistrationStore.h

namespace llvm::ejit {

// 全局注册数据暂存区 (进程级单例)
// 生命周期:
//   constructor 阶段: ejit_auto_register() → 注册函数 → 写入 Store
//   main() 阶段:      ejit_init() → consume() → 填充正式 Registry → Store 清空
class EJitRegistrationStore {
public:
    static EJitRegistrationStore& instance() {
        static EJitRegistrationStore store;
        return store;
    }

    // --- 写入接口 (constructor 阶段, 单线程) ---

    void registerBitcode(const std::string& funcName,
                         const uint8_t* data, size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        bitcodes_.push_back({funcName, data, size});
    }

    void registerPeriodArray(const std::string& periodName,
                             const std::string& varName,
                             void* baseAddr, uint64_t arraySize) {
        std::lock_guard<std::mutex> lock(mutex_);
        periodArrays_.push_back({periodName, varName, baseAddr, arraySize});
    }

    void registerStaticVar(const std::string& varName,
                           void* varAddr) {
        std::lock_guard<std::mutex> lock(mutex_);
        staticVars_.push_back({varName, varAddr});
    }

    // --- 消费接口 (ejit_init 调用) ---

    struct StoredData {
        std::vector<BitcodeEntry> bitcodes;
        std::vector<PeriodArrayEntry> periodArrays;
        std::vector<StaticVarEntry> staticVars;
    };

    // 取出所有暂存数据并清空内部容器
    // 仅在 ejit_init 中调用一次
    StoredData consume() {
        std::lock_guard<std::mutex> lock(mutex_);
        StoredData data;
        data.bitcodes = std::move(bitcodes_);
        data.periodArrays = std::move(periodArrays_);
        data.staticVars = std::move(staticVars_);
        return data;
    }

private:
    EJitRegistrationStore() = default;

    struct BitcodeEntry {
        std::string funcName;
        const uint8_t* data;    // 生命周期: 指向 .ejit.bitcode ELF section，由 OS loader 加载，
                                // 进程生命周期内有效，不可释放。若未来支持动态卸载共享库，
                                // 需在 dlclose 前清空对应 entries 并确保无进行中的 JIT 编译。
        size_t size;
    };
    struct PeriodArrayEntry {
        std::string periodName;
        std::string varName;
        void* baseAddr;
        uint64_t arraySize;
    };
    struct StaticVarEntry {
        std::string varName;
        void* varAddr;
    };

    std::mutex mutex_;  // 防御性: constructor 阶段单线程, 保证接口安全
    std::vector<BitcodeEntry> bitcodes_;
    std::vector<PeriodArrayEntry> periodArrays_;
    std::vector<StaticVarEntry> staticVars_;
};

} // namespace llvm::ejit
```

### 2.7.2 ejit_init

**全局单例类型定义**：

```cpp
// 进程级 EJIT 单例，由 ejit_init 创建，ejit_shutdown 释放
struct EJitInstance {
    std::unique_ptr<PeriodArrayRegistry> periodReg;
    std::unique_ptr<BitcodeTracker> bitcodeTracker;
    std::unique_ptr<EJitRuntimeState> runtimeState;
    std::unique_ptr<EJitCache> cache;
    std::unique_ptr<EJitOrcEngine> syncEngine;
    std::unique_ptr<EJitOrcEngine> asyncEngine;
    std::unique_ptr<EJitAsyncCompiler> asyncCompiler;
    std::unique_ptr<EJitCompileDriver> driver;
    EJitConfig config;
};

static EJitInstance gEJitInstance;
```

```cpp
ejit_status_t ejit_init(const ejit_config_t* config) {
    // Step 1: 解析配置
    EJitConfig cfg = config ? parseConfig(config) : EJitConfig::defaults();

    // Step 2: 从全局暂存区消费 constructor 阶段的注册数据
    auto storedData = EJitRegistrationStore::instance().consume();

    // Step 3: 创建核心组件并使用暂存数据填充
    //   3a. PeriodArrayRegistry
    auto periodReg = std::make_unique<PeriodArrayRegistry>();
    for (auto& entry : storedData.periodArrays) {
        periodReg->registerArray(entry.periodName, entry.varName,
                                  entry.baseAddr, entry.arraySize);
    }
    for (auto& entry : storedData.staticVars) {
        periodReg->registerStatic(entry.varName, entry.varAddr);
    }

    //   3b. Bitcode 缓存 (funcName → bitcode 映射)
    auto bitcodeTracker = std::make_unique<BitcodeTracker>();
    for (auto& entry : storedData.bitcodes) {
        bitcodeTracker->registerBitcode(entry.funcName,
                                         entry.data, entry.size);
    }

    //   3c. RuntimeState (activate/deactivate 状态管理)
    auto runtimeState = std::make_unique<EJitRuntimeState>();

    //   3d. Code Cache
    auto cache = std::make_unique<EJitCache>(
        cfg.maxCacheSize, cfg.maxCacheEntries, cfg.maxSingleFunctionSize);

    // Step 4: 创建 OrcJIT 引擎 (同步模式)
    auto syncEngineOrErr = EJitOrcEngine::Create(cfg);
    if (!syncEngineOrErr) {
        logError("EJit: failed to create sync engine");
        return EJIT_ERR_COMPILE_FAILED;
    }
    auto syncEngine = std::move(*syncEngineOrErr);
    syncEngine->setPeriodRegistry(periodReg.get());
    syncEngine->setRuntimeState(runtimeState.get());

    // Step 5: 创建异步引擎 (如果配置为异步)
    std::unique_ptr<EJitOrcEngine> asyncEngine;
    std::unique_ptr<EJitAsyncCompiler> asyncCompiler;
    if (cfg.compileMode == CompileMode::Async) {
        auto asyncEngineOrErr = EJitOrcEngine::Create(cfg);
        if (!asyncEngineOrErr) {
            logError("EJit: failed to create async engine");
            return EJIT_ERR_COMPILE_FAILED;
        }
        asyncEngine = std::move(*asyncEngineOrErr);
        asyncEngine->setPeriodRegistry(periodReg.get());
        asyncEngine->setRuntimeState(runtimeState.get());

        asyncCompiler = std::make_unique<EJitAsyncCompiler>(
            cfg, *cache, *runtimeState);
        asyncCompiler->setEngine(asyncEngine.get());
    }

    // Step 6: 创建编译调度器
    auto driver = std::make_unique<EJitCompileDriver>(
        cfg, *cache, *periodReg, *bitcodeTracker);
    driver->setSyncEngine(syncEngine.get());
    if (asyncCompiler) {
        driver->setAsyncCompiler(asyncCompiler.get());
    }

    // Step 7: 验证注册数据完整性
    if (storedData.periodArrays.empty() && storedData.staticVars.empty()) {
        // 警告: 无 EmbeddedJIT 数据 — 可能是无 ejit 代码的普通程序
        // 不视为错误，后续 ejit_compile_or_get 全部走 fallback
        logWarning("EJit: no registration data consumed from store");
    }

    // Step 8: 启动异步编译器 (如果配置)
    if (asyncCompiler) {
        asyncCompiler->start();
    }

    // Step 9: 存储全局单例
    gEJitInstance = EJitInstance{
        std::move(periodReg),
        std::move(bitcodeTracker),
        std::move(runtimeState),
        std::move(cache),
        std::move(syncEngine),
        std::move(asyncEngine),
        std::move(asyncCompiler),
        std::move(driver),
        cfg
    };

    return EJIT_OK;
}
```

### 2.7.3 Auto Register 与 ejit_init 的时序

```
程序加载
    │
    ├── 操作系统加载 .ejit.bitcode section → 内存
    │
    ├── llvm.global_ctors 执行 (优先级 65535)
    │   └── ejit_auto_register():
    │       ├── ejit_register_bitcode("funcName", ptr, size)
    │       │   → 写入 EJitRegistrationStore::instance().bitcodes_
    │       ├── ejit_register_period_array("cell", ...)
    │       │   → 写入 EJitRegistrationStore::instance().periodArrays_
    │       ├── ejit_register_static_var(...)
    │       │   → 写入 EJitRegistrationStore::instance().staticVars_
    │       └── ...
    │
    ├── main()
    │   └── ejit_init(NULL)  (用户调用)
    │       ├── EJitRegistrationStore::instance().consume()
    │       │   → 取出所有暂存数据，清空 Store
    │       ├── 用暂存数据填充 PeriodArrayRegistry
    │       ├── 用暂存数据填充 BitcodeTracker
    │       ├── 创建 Engine, Cache, CompileDriver
    │       ├── 验证注册数据完整性
    │       └── 启动异步编译器 (如果配置)
    │
    └── 业务代码...
        └── ejit_activate("cell", 3)
        └── process_task(3)  ← 首次调用触发 JIT
```

---

## 3. 线程安全模型

### 3.1 线程角色

```
┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│   用户线程 (Main)      │     │   后台编译线程          │     │   可能的多用户线程      │
│   调用:                │     │   执行:                │     │   (未来扩展)          │
│   - ejit_activate     │     │   - loadBitcode       │     │                      │
│   - ejit_deactivate   │     │   - IRTransformPass   │     │                      │
│   - process_task()    │     │   - optimize          │     │                      │
│     → Wrapper         │     │   - codeGen           │     │                      │
│       → getOrCompile  │     │   - cache.put         │     │                      │
│                       │     │                       │     │                      │
│   RuntimeState        │     │   独立的 LLVMContext   │     │                      │
│   (R/W, mutex保护)     │     │   独立的 LLJIT         │     │                      │
│                       │     │   共享 MemoryMgr       │     │                      │
│   Cache (查询)         │     │   (mutex 保护)         │     │                      │
│                       │     │                       │     │                      │
│   仅在主线程:           │     │   Cache (写入)         │     │                      │
│   - activate/deactiv. │     │                       │     │                      │
└──────────────────────┘     └──────────────────────┘     └──────────────────────┘
```

### 3.2 锁策略

| 数据结构 | 锁类型 | 读端 | 写端 |
|---------|--------|------|------|
| EJitCache | shared_mutex (or BareMetalMutex) | getOrNull (独占 — splice write) | put/evict (独占) |
| RuntimeState | mutex | 任意 | 仅主线程 (activate/deactivate) |
| PeriodArrayRegistry | 无锁 (只读) | 任意 (init 后只读) | 仅 ejit_auto_register (初始化时) |
| BitcodeCache | 无锁 (只读) | 任意 | 仅 ejit_auto_register (初始化时) |
| JITLinkMemoryManager | mutex (per slab) | 代码生成 | 代码生成 (sync + async) |
| Async Request Queue | mutex + cv | - | submitRequest + workerLoop |

### 3.3 LLVM 内部线程安全性

```cpp
// ThreadSafeModule 的正确使用
// 用户线程 (同步模式):
//   getOrCompile → syncCompiler.compile
//     → TSM.withModuleDo([](Module &M) { ... })
//     → LLVMContext 被 withModuleDo 的锁保护

// 后台线程 (异步模式):
//   workerLoop → compileOne
//     → workerEngine→compile  (独立的 LLJIT 实例)
//     → 独立的 LLVMContext, 无竞态
```

### 3.4 异步编译的内存序保证

异步模式下，用户线程和编译线程共享全局变量数据（`ejit_may_const` 字段值）。必须确保以下 happens-before 关系成立：

```
用户线程                          编译线程
───────                          ───────
ejit_activate(name, idx)
  ├─ 写入 RuntimeState               
  │  (设置为 active)               
  ├─ atomic_thread_fence            
  │  (memory_order_release)    ─────────→  compileOne()
  │                                         ├─ isPeriodActive() → active
  │                                         ├─ atomic_thread_fence
  │                                         │  (memory_order_acquire)
  │                                         └─ 读取 may_const 字段值
  │                                            (可见 activate 前的写入)

ejit_deactivate(name, idx)
  ├─ 读取 RuntimeState
  ├─ 设置为 inactive ───────────→  compileOne()
  │  (若 deactivate 先于            ├─ isPeriodActive() → INACTIVE
  │   compileOne 的检查)             └─ 跳过编译, 不读字段值
  └─ ...
```

**实现要点：**

```cpp
// RuntimeState::activate 内部
void EJitRuntimeState::activate(const std::string& periodName, int cellIdx) {
    std::lock_guard<std::mutex> lock(mutex_);
    // ... 更新激活状态 ...
    // Release 屏障：确保所有用户线程对全局变量的写入
    // 在 activate 返回后对编译线程可见
    std::atomic_thread_fence(std::memory_order_release);
}

// RuntimeState::deactivate 内部
void EJitRuntimeState::deactivate(const std::string& periodName, int cellIdx) {
    std::lock_guard<std::mutex> lock(mutex_);
    // ... 更新激活状态 ...
}
```

**关键规则**：
- `ejit_activate` 必须在**写入** may_const 全局变量之后调用（或 activate 本身不代表写入，写入由业务代码在 activate 前完成）。若写入与 activate 并不同步，业务代码应在 activate 前自行加 barrier。
- `compileOne` 在读取 may_const 字段值前执行 `acquire` fence，与 activate 中的 `release` fence 配对，保证字段修改可见。
- `isPeriodActive` 检查本身在 mutex 保护下，确保 deactivate 后提交的编译请求能看到失效状态。
- 若 `isPeriodActive` 返回 false，`compileOne` 直接跳过，不读取任何字段值——因此不存在 "读到半修改值" 的窗口。

---

## 4. 嵌入式优化

### 4.1 内存预算

```cpp
// 嵌入式默认配置
struct EmbeddedDefaults {
    // JIT 代码内存 (JITLink slab)
    static constexpr size_t kCodeSlabSize = 2 * 1024 * 1024;  // 2MB
    static constexpr size_t kDataSlabSize = 128 * 1024;       // 128KB

    // Code Cache
    static constexpr size_t kMaxCacheSize = 32 * 1024 * 1024; // 32MB
    static constexpr size_t kMaxCacheEntries = 4096;
    static constexpr size_t kMaxSingleFuncSize = 512 * 1024;  // 512KB

    // LLVM 内部
    static constexpr size_t kLLVMStackSize = 256 * 1024;  // 256KB (编译线程栈)
};
```

### 4.2 按需加载 Bitcode

```cpp
// BitcodeTracker: 维护 funcName → bitcode 位置映射
// 不预加载所有 bitcode, 而是在首次 JIT 编译时按需解析
class BitcodeTracker {
    struct Entry {
        std::string funcName;
        const uint8_t* bitcodeData;   // 指向 .ejit.bitcode section (进程生命周期有效，见 §2.7.1 BitcodeEntry)
        size_t bitcodeSize;
    };

    std::unordered_map<std::string, Entry> entries_;

    // 按需加载: 首次调用时 parseModule
    // parse 开销 ~50KB 临时内存 (在 LLVMContext 中)
    std::unordered_map<std::string, std::string> parsedModules_;

public:
    void registerBitcode(const std::string& funcName,
                         const uint8_t* data, size_t size);

    Expected<StringRef> getBitcode(const std::string& funcName);
};
```

### 4.3 静态链接优化

```cpp
// 嵌入式场景: libejit.a 静态链接
// 可以移除不需要的功能:
//
//   - 调试符号 (DWARF) → strip
//   - TargetParser (仅 ARM/AArch64) → 编译时选择
//   - Disassembler (MCDisassembler) → 移除
//   - Remarks → 移除
//   - 不需要的 Target (仅 ARM/AArch64) → 链接时 GC

// CMake 配置:
//   -DLLVM_TARGETS_TO_BUILD="AArch64;ARM"
//   -DLLVM_ENABLE_ZSTD=OFF
//   -DLLVM_ENABLE_ZLIB=OFF
//   -DLLVM_ENABLE_TERMINFO=OFF
//   -DLLVM_ENABLE_THREADS=ON          (异步编译需要)
```

---

## 5. 错误处理与 Fallback

### 5.1 错误传播路径

```
JIT 编译失败
    ↓
EJitOrcEngine::compileFunction → return Error
    ↓
EJitCompileDriver::getOrCompile → return nullptr
    ↓
ejit_compile_or_get → return NULL
    ↓
Wrapper → 跳转到 jit_fallback → 执行 AOT 代码
```

### 5.2 错误日志

```cpp
// 结构化错误日志
struct EJitError {
    ErrorCode code;
    std::string message;
    std::string funcName;
    std::string cacheKey;
    uint64_t timestamp;
    size_t attemptedMemUsage;
};

// 日志记录 (环形缓冲区, 避免 malloc 在错误路径失败)
class EJitLogger {
    static constexpr size_t kMaxErrors = 32;
    EJitError errors_[kMaxErrors];
    size_t writeIdx_ = 0;
    std::mutex mutex_;

public:
    void log(ErrorCode code, const std::string& msg,
             const std::string& func, const std::string& key);
    const EJitError* getLastError() const;
};
```

---

## 6. 完整 JIT 编译时序

### 6.1 同步编译时序

```
时间 →
用户调用: ─────────────────────────────────────────────────────────
           adjust_param(idx)
           │
           ├─ wrapper:
           │    ├─构建 dims
           │    ├─ejit_compile_or_get()
           │    │   ├─查 Cache → MISS
           │    │   ├─验证时间窗状态
           │    │   ├─syncCompiler.compile()
           │    │   │   ├─loadBitcodeModule         (~5ms)
           │    │   │   ├─[IRTransformLayer]:
           │    │   │   │   ├─参数预处理              (<1ms)
           │    │   │   │   ├─InstCombine             (~2ms)
           │    │   │   │   ├─Inline                  (~3ms)
           │    │   │   │   ├─EJitStructFieldPass     (~2ms)
           │    │   │   │   └─优化 Pipeline           (~12ms)
           │    │   │   ├─IRCompileLayer:
           │    │   │   │   └─LLVM → Object           (~20ms)
           │    │   │   ├─ObjectLinkingLayer:
           │    │   │   │   └─JITLink 链接+分配        (~3ms)
           │    │   │   └─lookup → 函数指针
           │    │   ├─cache.put(entry)
           │    │   └─return pfn
           │    ├─pfn != NULL
           │    └─调用 pfn(idx) → 特化函数
           │
           └─总延迟: ~48ms → 返回
```

### 6.2 异步编译时序

```
时间 →

调用线程:                后台线程:
───────                ───────────
adjust_param(idx)
│
├─ wrapper:
│    ├─构建 dims
│    ├─ejit_compile_or_get()
│    │   ├─查 Cache → MISS
│    │   ├─提交 CompileRequest
│    │   └─return NULL ─────→    workerLoop():
│    │                              ├─从队列取请求
│    ├─pfn == NULL                  ├─compileOne()
│    └─fallback(AOT)                │   ├─loadBitcode
│                                   │   ├─IRTransform
│    ←返回 (运行 AOT 代码)            │   ├─优化
│                                   │   ├─CodeGen
│                                   │   ├─MemoryAlloc
│                                   │   └─cache.put()
│                                   │
下次调用:                             │
adjust_param(idx)                   │
├─ wrapper:                         │
│    ├─ejit_compile_or_get()       │
│    ├─查 Cache → HIT! ☆          │
│    ├─return pfn ────────────────  │
│    └─调用 pfn → 特化函数
│
└─总延迟: <1ms (Cache 命中)
```

---

## 7. 文件与组件清单

```
llvm/lib/ExecutionEngine/EJIT/
├── EJit.cpp                     # 主类实现 (EJit)
├── EJitRuntime.cpp              # C 运行时接口 (ejit_init/shutdown/activate...)
├── EJitOrcEngine.cpp            # LLJIT 封装 + 引擎创建
├── EJitJITLinkMemoryManager.cpp # 嵌入式 JITLink 内存管理器
├── EJitSyncCompiler.cpp         # 同步编译器
├── EJitAsyncCompiler.cpp        # 异步编译器 (后台线程+队列)
├── EJitCompileDriver.cpp        # 编译调度器 (Sync/Async 统一入口)
├── EJitCache.cpp                # Code Cache (LRU + 大小限制)
├── EJitStructFieldPass.cpp      # 结构体字段特化 Pass (JIT Pipeline)
├── EJitOptimizer.cpp            # 优化 Pipeline (L1/L2/L3)
├── EJitModuleLoader.cpp         # Bitcode 按需加载器
├── EJitLogger.cpp               # 错误日志 (环形缓冲区)
├── EJitRegistration.cpp         # AOT 注册回调实现 (
│                                  ejit_register_period_array,
│                                  ejit_register_bitcode)
└── CMakeLists.txt

llvm/include/llvm/ExecutionEngine/EJIT/
├── EJit.h                       # C++ 主 API
├── EJitRuntime.h                # C 运行时接口
├── EJitOrcEngine.h              # EJitOrcEngine 声明
├── EJitJITLinkMemoryManager.h   # 内存管理器声明
├── EJitCache.h                  # Cache 声明
├── EJitStructFieldPass.h        # StructField Pass 声明
├── EJitOptimizer.h              # 优化 Pipeline 声明
├── EJitOptions.h                # 配置选项
├── EJitError.h                  # 错误类型定义
└── EJitRegistration.h           # 注册 API 声明
```

---

## 8. 测试策略

### 8.1 单元测试

```cpp
// EJitJITLinkMemoryManagerTest.cpp
// TEST(MemoryManager, BasicAllocation)     - 基本分配+释放
// TEST(MemoryManager, OOM_ReturnsError)    - 超出限制
// TEST(MemoryManager, ConcurrentAlloc)     - 并发分配
// TEST(MemoryManager, MixedProtections)    - RX + RW 混合

// EJitCacheTest.cpp
// TEST(Cache, PutAndGet)                   - 基本读写
// TEST(Cache, LRU_Eviction)                - LRU 淘汰
// TEST(Cache, SizeLimit)                   - 大小限制
// TEST(Cache, InvalidateByPeriod)          - 时间窗失效

// EJitSyncCompilerTest.cpp
// TEST(SyncCompiler, CompileBasicFunction) - 基本编译
// TEST(SyncCompiler, StructFieldReplace)   - 字段替换后编译
// TEST(SyncCompiler, MultiPeriod)          - 多时间窗编译

// EJitAsyncCompilerTest.cpp
// TEST(AsyncCompiler, SubmitAndRetrieve)   - 提交+结果
// TEST(AsyncCompiler, ConcurrentSubmit)    - 并发提交
// TEST(AsyncCompiler, StopDuringCompile)   - 编译中停止

// EJitCompileDriverTest.cpp
// TEST(Driver, Sync_CacheHit)              - 同步+Cache 命中
// TEST(Driver, Sync_CacheMiss)             - 同步+Cache 未命中
// TEST(Driver, Async_FirstCallNull)        - 异步首次 NULL
// TEST(Driver, Async_SecondCallHit)        - 异步第二次命中
```

### 8.2 集成测试

```c
// test_ejit_integration.c
// 场景 1: 端到端同步编译
//   定义 struct + ejit_may_const + ejit_period_arr
//   定义 ejit_entry 函数
//   AOT 编译 + bitcode 嵌入
//   运行时: init → activate → 调用 → 验证特化结果与预期一致

// 场景 2: 端到端异步编译
//   同上，配置 EJIT_COMPILE_ASYNC
//   首次调用返回 AOT 结果 → sleep(100ms) → 第二次调用返回 JIT 结果

// 场景 3: 时间窗失效
//   JIT 编译 → activate → deactivate → activate(new_value)
//   验证特化函数使用新值重新编译

// 场景 4: Cache 淘汰
//   填充 Cache 到超过限制
//   验证 LRU 正确淘汰旧条目

// 场景 5: 多时间窗
//   ejit_entry 依赖 cell+trp
//   不同 (cellIdx, trpIdx) 组合生成不同的特化版本
```

---

## 9. 实施注意事项

1. **ExecutionSession 生命周期**: OrcJIT 的 `ExecutionSession` 析构时报告所有未释放的资源（`ResourceTracker`）。确保通过 `clear()` 或 `ResourceTracker::remove()` 正确释放。

2. **LLVMContext 隔离**: 异步编译器的 LLVMContext 必须独立于调用线程的 LLVMContext。`ThreadSafeModule` 的创建使用独立 `LLVMContext` 工厂。

3. **JITLink slab 碎片**: 当前 bump-allocator 设计不支持单独释放。LRU 淘汰后碎片可能无法重用。后续可考虑：
   - 分段 slab（每个函数独立段）
   - 压缩 (移动活跃代码, 修正重定位)
   - 或者接受碎片，限制函数数量

4. **ARM/AArch64 内存保护**: 代码段需要 `PROT_READ | PROT_EXEC`，数据段需要 `PROT_READ | PROT_WRITE`。使用 `mprotect` 或等效系统调用设置。AArch64 同样适用此保护模型。

5. **`CloneToNewContextOnEmit`**: 异步模式下必须调用 `IRLayer::setCloneToNewContextOnEmit(true)` 确保多线程安全。

6. **全局单例**: `ejit_init` 创建的实例是进程级单例。`ejit_shutdown` 释放所有资源。不支持多次 init/shutdown 循环（简化设计）。

7. **Signal-safe 内存操作**: 在某些嵌入式系统上，`mmap`/`munmap` 不可用。提供预分配 slab + 手动内存保护设置的回退。

---

*文档版本: 1.1*
*创建日期: 2026-04-26*
*更新日期: 2026-04-29*

---

## 双路径注册机制 (v1.2)

### 注册路径

| 路径 | 触发方式 | 适用 |
|---|---|---|
| 构造器 | `llvm.global_ctors` → CRT 调用 `ejit_auto_register()` | hosted |
| 静态注册表 | `ejit_init()` 遍历 `__ejit_registry_bitcode[]` + `__ejit_registry_period[]` | 裸核 / `forceStaticRegistry=true` |
| 手动 API | `ejit_register_*()` 在 `ejit_init()` 之后调用 | 运行时动态 |

### 运行时选择

```cpp
StoredData data = EJitRegistrationStore::instance().consume();
if (config_.forceStaticRegistry || data.empty()) {
    walkTable(__ejit_registry_bitcode);   // PASS1
    walkTable(__ejit_registry_period);    // PASS2
} else {
    // 构造器路径(现有)
}
```

### C API dual-path

`ejit_register_bitcode` / `ejit_register_period_array` / `ejit_register_static_var` /
`ejit_register_symbol` 均支持双路径：`gEJIT` 非空 → 直接转发引擎，否则暂存 `EJitRegistrationStore`。

### 裸核运行时修复

- `Builder.setLinkProcessSymbolsByDefault(false)` — 避免 dlopen/dlsym
- `EJIT_DEFAULT_TARGET_TRIPLE` 强制要求 — 代替 detectHost()
- `setNumCompileThreads(0)` — 单线程
- `forceStaticRegistry` 配置项 — 强制走静态注册表路径

### ejit_config_t 新增字段

```c
typedef struct {
    ...
    bool forceStaticRegistry;  // true = 强制静态表路径
} ejit_config_t;
```

### 测试

`ejit_test/ejit_manual_register_test.c` — x86 上 PASS，验证 forceStaticRegistry + JIT 编译执行。

---

*文档版本: 1.2*  
*更新日期: 2026-06-01*
