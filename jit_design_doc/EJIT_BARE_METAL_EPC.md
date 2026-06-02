# EJIT 裸核 ExecutorProcessControl 设计

**版本**: 1.6
**日期**: 2026-06-02
**关联**: PASS7_EJitRuntime_OrcJITLink.md, EJIT_BARE_METAL_STUBS.md, EJIT_LIBRARY_TRIMMING.md

---

## 1. 动机

### 1.1 问题：hosted EPC 在裸核上全部不可用

`LLJIT` 默认使用 `SelfExecutorProcessControl` + `InProcessMemoryManager`。
这些组件在 hosted 环境（Linux/macOS/Windows）正常工作，但在裸核上，
**每一个**依赖的系统调用都不存在：

| hosted 组件 | 依赖的系统调用 | 裸核上 |
|---|---|---|
| `SelfExecutorProcessControl` → `DylibManager` | `dlopen` / `dlsym` | 无动态链接器 |
| `InProcessMemoryManager` | `mmap` / `mprotect` | 无 MMU / 无 OS 内存管理 |
| `InProcessMemoryAccess` | `mprotect` (权限切换) | 无页表保护 |
| `SelfExecutorProcessControl` 构造 | `sysconf(_SC_PAGESIZE)` | 无 sysconf |
| `SelfExecutorProcessControl` → `runAsMain` | `dlopen` / `dlsym` 查找 `main` | 无符号解析 |

**结论**：不是"改一个地方就好"——整个 EPC 层从内存分配到符号查找，
全部建立在 POSIX 系统调用之上。裸核必须从头实现一个新的 EPC。

### 1.2 这不是"符号发现"问题

容易产生的误解：只要用 `ejit_register_symbol` 注册了符号，
JIT 就能在裸核上跑。

实际上符号发现只是 JIT 编译链路中**最后一步的输入**。
在此之前，JIT 编译器自己需要：

1. **内存** — LLVM CodeGen 产出 object file 后，JITLink 需要可执行内存来放代码
2. **写入** — relocation fixup 后，最终机器码要写到目标地址
3. **执行** — ORC runtime 的 init/fini 函数需要通过 wrapper 机制调用

这三件事与"符号从哪来"完全正交。`ejit_register_symbol` 解决的是
"JIT 代码里引用的 `printf` 地址是多少"，而 EPC 解决的是
"JIT 编译器自己需要的内存和函数调用怎么实现"。

---

## 2. 为什么 EPC 无法绕过

### 2.1 JIT 编译链路回顾

一次完整的 JIT 编译经过以下步骤：

```
ejit_get_or_compile()
  │
  ├─ EJitCompileDriver::getOrCompile()
  │   ├─ 查 cache（命中直接返回）
  │   ├─ loader_.getBitcode()           ← 从 registry 取 bitcode
  │   └─ syncEngine_->loadBitcodeModule()
  │
  └─ EJitOrcEngine::loadBitcodeModule()
      │
      ├─ parseBitcodeFile()              ← bitcode → LLVM Module
      ├─ JD.define(absoluteSymbols())    ← 用 ejit_register_symbol 注册的地址
      └─ J->addIRModule()               ← 交给 LLJIT
          │
          └─ LLJIT 内部：
              ├─ IRTransform            ← 参数特化 + 优化 pipeline
              ├─ LLVM CodeGen           ← IR → object file (MemoryBuffer)
              ├─ ObjectLinkingLayer     ← 解析 object file → LinkGraph
              └─ JITLink                ← 核心：链接 + 内存分配 + fixup
                  │
                  ├─ linkPhase1:
                  │   ├─ Ctx->getMemoryManager().allocate()  ★ MemoryManager
                  │   └─ 在 working memory 上做 relocation fixup
                  │       (working memory == target memory，
                  │        进程内模型同一块内存)
                  │
                  ├─ linkPhase2 — InFlightAlloc::finalize():
                  │   ├─ mprotect(PROT_READ|PROT_EXEC)       ★ 权限切换
                  │   │   (裸核: no-op, 裸核默认 RWX)
                  │   ├─ __builtin___clear_cache              ★ I-cache flush
                  │   │   (AArch64 必须; x86 no-op)
                  │   └─ runFinalizeActions(G->allocActions())
                  │       (deallocation cleanup 注册)
                  │
                  └─ callWrapperAsync:                       ★ EPC
                      ORC runtime 通过此接口在"目标进程"执行
                      init/fini 函数（裸核: 同步函数指针调用）
```

**注意**：对于进程内模型（SelfEPC + InProcessMemoryManager，以及裸核的 Slab 分配器），
**working memory 就是 target memory**——JITLink 直接在目标地址上做 fixup，
finalize 阶段**没有** working→target 的 memcpy 步骤。
`MemoryAccess::writeBuffersAsync` 等接口在 finalize 中不被调用
（它们在远程执行器场景中使用，用于跨进程复制代码）。

★ 标记的就是 EPC 提供的核心能力。每一步都必须工作，否则 JIT 编译
在 `addIRModule` → `lookup` 阶段就会失败。

### 2.2 四个必须替换的 EPC 子组件

| 组件 | hosted 实现 | 裸核替代 | 为什么不能跳过 |
|---|---|---|---|
| **MemoryManager** | `InProcessMemoryManager` → `mmap` 申请可执行内存 | `SlabJITLinkMemoryManager` → bump allocator 从预分配 buffer 切 | JITLink `linkPhase1`（`JITLinkGeneric.cpp:56`）调用 `getMemoryManager().allocate()`。不提供 → JIT 编译直接失败。 |
| **MemoryAccess** | `InProcessMemoryAccess` → `memcpy` + `mprotect` 切换权限 | `BareMetalMemoryAccess` → `memcpy`，`writePointersAsync` 按 `IsArch64Bit` 分支 | JITLink 在非进程内场景通过此接口写入目标内存。虽然进程内模型的 finalize 不走 MemoryAccess，但 ORC 框架的其他路径可能调用（如 bootstrap 阶段的写入），必须正确实现 13 个纯虚方法。 |
| **DylibManager** | `SelfEPC` 私有继承 → `dlopen`/`dlsym` 查找进程中的动态库符号 | 私有继承，`loadDylib(nullptr)` 返回合法句柄，`lookupSymbolsAsync` 返回空 | 防御性实现。裸核配置下 `LinkProcessSymbolsByDefault=false` 阻止了 `EPCDynamicLibrarySearchGenerator` 的创建（该生成器是调用 `loadDylib` 的唯一代码路径），因此实际上不会被调用。但返回合法句柄可避免未来代码路径变化导致的崩溃。 |
| **callWrapperAsync** | `SelfEPC` → 函数指针强转同步调用，立即回调 `OnComplete` | 同 SelfEPC，内存中直接函数指针调用 | ORC runtime 的 init/fini 函数（如 `__orc_rt_init`）通过 wrapper 机制执行。裸核无远程 IPC，同步直调即可。 |

### 2.3 ejit_register_symbol 和 EPC 的分工

```
                    ejit_register_symbol
                    ├── 提供: 符号名 → 地址映射
                    │         (printf → 0x40001234)
                    │         (g_cells → 0x20000000)
                    └── 不提供: 内存分配、内存写入、函数调用

                    EPC
                    ├── MemoryManager:     "JIT 代码放哪块内存？"
                    ├── MemoryAccess:      "怎么读写 JIT 目标内存？"
                    ├── DylibManager:      "进程中有哪些动态库符号？"
                    ├── callWrapperAsync:  "ORC runtime 初始化函数怎么调？"
                    └── 不提供: 应用层符号（那是 ejit_register_symbol 的事）
```

**结论**：两者正交、互补、不可互相替代。一个有符号没内存，JIT 编译器连 obj 文件都放不下；一个有内存没符号，JIT 代码编译出来全是未解析引用。

---

## 3. 总体方案

分三阶段实现。

### 第一阶段：实现 BareMetalEPC + EJitOrcEngine 集成

包含两个子任务：
- **BareMetalEPC 类**：继承 `ExecutorProcessControl`，私有继承 `DylibManager`
- **EJitOrcEngine 集成**：在 `#ifdef EJIT_BARE_METAL` 块中注入 BareMetalEPC

#### BareMetalMemoryAccess

**源码参考**：`InProcessMemoryAccess`（`InProcessMemoryAccess.cpp`）。
接口是异步的但实现是同步回调。关键细节：`writePointersAsync` 必须按指针宽度分支
——64 位目标写 8 字节，32 位目标写 4 字节，否则 32 位 ARM 上会写坏相邻内存。

```cpp
class BareMetalMemoryAccess : public orc::MemoryAccess {
  bool IsArch64Bit;
public:
  explicit BareMetalMemoryAccess(bool IsArch64Bit) : IsArch64Bit(IsArch64Bit) {}

  void writeUInt8sAsync(ArrayRef<tpctypes::UInt8Write> Ws,
                        WriteResultFn OnComplete) override {
    for (auto &W : Ws)
      *W.Addr.toPtr<uint8_t *>() = W.Value;
    OnComplete(Error::success());
  }

  void writeUInt16sAsync(ArrayRef<tpctypes::UInt16Write> Ws,
                         WriteResultFn OnComplete) override {
    for (auto &W : Ws)
      *W.Addr.toPtr<uint16_t *>() = W.Value;
    OnComplete(Error::success());
  }

  void writeUInt32sAsync(ArrayRef<tpctypes::UInt32Write> Ws,
                         WriteResultFn OnComplete) override {
    for (auto &W : Ws)
      *W.Addr.toPtr<uint32_t *>() = W.Value;
    OnComplete(Error::success());
  }

  void writeUInt64sAsync(ArrayRef<tpctypes::UInt64Write> Ws,
                         WriteResultFn OnComplete) override {
    for (auto &W : Ws)
      *W.Addr.toPtr<uint64_t *>() = W.Value;
    OnComplete(Error::success());
  }

  void writePointersAsync(ArrayRef<tpctypes::PointerWrite> Ws,
                          WriteResultFn OnComplete) override {
    // 必须按指针宽度分支 —— ExecutorAddr 内部总是 uint64_t，
    // 直接写 8 字节会在 32 位目标上损坏相邻内存。
    if (IsArch64Bit) {
      for (auto &W : Ws)
        *W.Addr.toPtr<uint64_t *>() = W.Value.getValue();
    } else {
      for (auto &W : Ws)
        *W.Addr.toPtr<uint32_t *>() =
            static_cast<uint32_t>(W.Value.getValue());
    }
    OnComplete(Error::success());
  }

  void writeBuffersAsync(ArrayRef<tpctypes::BufferWrite> Ws,
                         WriteResultFn OnComplete) override {
    for (auto &W : Ws)
      memcpy(W.Addr.toPtr<char *>(), W.Buffer.data(), W.Buffer.size());
    OnComplete(Error::success());
  }

  void readUInt8sAsync(ArrayRef<ExecutorAddr> Rs,
                       OnReadUIntsCompleteFn<uint8_t> OnComplete) override {
    ReadUIntsResult<uint8_t> Result;
    Result.reserve(Rs.size());
    for (auto &R : Rs)
      Result.push_back(*R.toPtr<uint8_t *>());
    OnComplete(std::move(Result));
  }

  void readUInt16sAsync(ArrayRef<ExecutorAddr> Rs,
                        OnReadUIntsCompleteFn<uint16_t> OnComplete) override {
    ReadUIntsResult<uint16_t> Result;
    Result.reserve(Rs.size());
    for (auto &R : Rs)
      Result.push_back(*R.toPtr<uint16_t *>());
    OnComplete(std::move(Result));
  }

  void readUInt32sAsync(ArrayRef<ExecutorAddr> Rs,
                        OnReadUIntsCompleteFn<uint32_t> OnComplete) override {
    ReadUIntsResult<uint32_t> Result;
    Result.reserve(Rs.size());
    for (auto &R : Rs)
      Result.push_back(*R.toPtr<uint32_t *>());
    OnComplete(std::move(Result));
  }

  void readUInt64sAsync(ArrayRef<ExecutorAddr> Rs,
                        OnReadUIntsCompleteFn<uint64_t> OnComplete) override {
    ReadUIntsResult<uint64_t> Result;
    Result.reserve(Rs.size());
    for (auto &R : Rs)
      Result.push_back(*R.toPtr<uint64_t *>());
    OnComplete(std::move(Result));
  }

  void readPointersAsync(ArrayRef<ExecutorAddr> Rs,
                         OnReadPointersCompleteFn OnComplete) override {
    ReadPointersResult Result;
    Result.reserve(Rs.size());
    for (auto &R : Rs)
      Result.push_back(ExecutorAddr::fromPtr(*R.toPtr<void **>()));
    OnComplete(std::move(Result));
  }

  void readBuffersAsync(ArrayRef<ExecutorAddrRange> Rs,
                        OnReadBuffersCompleteFn OnComplete) override {
    ReadBuffersResult Result;
    Result.reserve(Rs.size());
    for (auto &R : Rs) {
      Result.push_back({});
      Result.back().resize(R.size());
      memcpy(Result.back().data(), R.Start.toPtr<char *>(), R.size());
    }
    OnComplete(std::move(Result));
  }

  void readStringsAsync(ArrayRef<ExecutorAddr> Rs,
                        OnReadStringsCompleteFn OnComplete) override {
    ReadStringsResult Result;
    Result.reserve(Rs.size());
    for (auto &R : Rs)
      Result.push_back(std::string(R.toPtr<const char *>()));
    OnComplete(std::move(Result));
  }
};
```

要点：
- 所有方法**同步回调** OnComplete（和 `InProcessMemoryAccess` 一样，无 IPC）
- `writePointersAsync` **必须**根据 `IsArch64Bit` 分支（`ExecutorAddr` 内部总是 `uint64_t`）
- 裸核默认 RWX，不做 mprotect 权限切换

#### BareMetalEPC 完整声明

```cpp
class BareMetalEPC : public orc::ExecutorProcessControl,
                     private orc::DylibManager {
public:
  /// @param SSP         符号字符串池
  /// @param TargetTriple 目标三元组（用于判断指针宽度）
  /// @param PageSize    页大小（裸核硬编码 4096）
  /// @param MemMgr      JIT 内存管理器。
  ///   - nullptr: 自动创建 InProcessMemoryManager（仅限 hosted 测试！）
  ///   - 裸核环境: 必须传入 SlabJITLinkMemoryManager
  BareMetalEPC(std::shared_ptr<SymbolStringPool> SSP,
               Triple TargetTriple, unsigned PageSize,
               std::unique_ptr<jitlink::JITLinkMemoryManager> MemMgr)
      : ExecutorProcessControl(std::move(SSP),
                               std::make_unique<InPlaceTaskDispatcher>()),
        BMMA(TargetTriple.isArch64Bit()) {

    this->TargetTriple = std::move(TargetTriple);
    this->PageSize = PageSize;
    this->MemAccess = &BMMA;
    this->DylibMgr = this;   // 私有继承 DylibManager，参考 SelfEPC 模式

    // MemMgr: nullptr → hosted 测试用 InProcessMemoryManager
    //         非空   → 裸核用 SlabJITLinkMemoryManager
    if (MemMgr)
      OwnedMemMgr = std::move(MemMgr);
    else
      OwnedMemMgr = std::make_unique<jitlink::InProcessMemoryManager>(PageSize);
    this->MemMgr = OwnedMemMgr.get();

    // JDI 保持零初始化。裸核没有 ORC runtime 调度机制，
    // JITDispatchFunction / JITDispatchContext 不会被使用。
  }

  // ---- ExecutorProcessControl 纯虚接口 ----

  Expected<int32_t> runAsMain(ExecutorAddr MainFnAddr,
                              ArrayRef<std::string> Args) override {
    return make_error<StringError>("bare-metal: runAsMain not supported",
                                   inconvertibleErrorCode());
  }

  Expected<int32_t> runAsVoidFunction(ExecutorAddr VoidFnAddr) override {
    using FnTy = int32_t (*)();
    return FnTy(VoidFnAddr.toPtr<FnTy>())();
  }

  Expected<int32_t> runAsIntFunction(ExecutorAddr IntFnAddr, int Arg) override {
    using FnTy = int32_t (*)(int);
    return FnTy(IntFnAddr.toPtr<FnTy>())(Arg);
  }

  void callWrapperAsync(ExecutorAddr WrapperFnAddr,
                        IncomingWFRHandler OnComplete,
                        ArrayRef<char> ArgBuffer) override {
    using WrapperFnTy =
        shared::CWrapperFunctionResult (*)(const char *, size_t);
    auto *Fn = WrapperFnAddr.toPtr<WrapperFnTy>();
    shared::WrapperFunctionResult R = Fn(ArgBuffer.data(), ArgBuffer.size());
    OnComplete(std::move(R));
  }

  Error disconnect() override { return Error::success(); }

  // ---- DylibManager 纯虚接口 (私有继承) ----

  Expected<tpctypes::DylibHandle> loadDylib(const char *DylibPath) override {
    // nullptr → 进程"全局符号句柄"。
    // 裸核配置下 LoadProcessSymbolsByDefault=false 阻止了
    // EPCDynamicLibrarySearchGenerator 的创建，因此此方法实际上不会被调用。
    // 但返回合法句柄作为防御性编程。
    if (!DylibPath)
      return tpctypes::DylibHandle(ExecutorAddr::fromPtr(nullptr));
    return make_error<StringError>(
        "bare-metal: dynamic libraries not supported",
        inconvertibleErrorCode());
  }

  void lookupSymbolsAsync(ArrayRef<LookupRequest> Request,
                          SymbolLookupCompleteFn F) override {
    // 裸核无动态链接器。所有应用符号通过 ejit_register_symbol 注入 JITDylib。
    // 返回地址为 0 的 ExecutorSymbolDef 作为"未找到"标记——
    // 这是 ORC 中表示弱符号未解析的标准方式（参考 ExecutorSymbolDef 默认构造函数）。
    std::vector<tpctypes::LookupResult> Results;
    for (const auto &R : Request) {
      tpctypes::LookupResult LR;
      LR.resize(R.Symbols.size(), ExecutorSymbolDef());
      Results.push_back(std::move(LR));
    }
    F(std::move(Results));
  }

private:
  BareMetalMemoryAccess BMMA;
  std::unique_ptr<jitlink::JITLinkMemoryManager> OwnedMemMgr;
};
```

要点：
- **构造函数不将 MemMgr 传给基类**——`ExecutorProcessControl` 只接受 `(SSP, TaskDispatcher)`
- `MemMgr = OwnedMemMgr.get()`：参考 `SelfExecutorProcessControl` 模式，成员持有所有权，裸指针给基类
- `nullptr` → 自动创建 `InProcessMemoryManager`，**仅限 hosted 测试**（`InProcessMemoryManager` 内部用 `mmap`）。裸核必须传 `SlabJITLinkMemoryManager`
- `TaskDispatcher` 用 `InPlaceTaskDispatcher`（单线程，task 在当前线程立即执行）
- `DylibMgr = this`：参考 `SelfExecutorProcessControl` 的私有继承模式
- `JDI`（JITDispatchInfo）保持零初始化，裸核无 ORC runtime 调度

#### EJitOrcEngine.cpp 集成

```cpp
#ifdef EJIT_BARE_METAL
  // 1. 创建裸核 EPC
  //    第一阶段传 nullptr → 内部创建 InProcessMemoryManager（hosted 测试）
  //    第二阶段替换为 SlabJITLinkMemoryManager
  //    注意: EJIT_DEFAULT_TRIPLE 由 CMake 从 EJIT_DEFAULT_TARGET_TRIPLE 映射:
  //      target_compile_definitions(LLVMEJIT PUBLIC
  //          EJIT_DEFAULT_TRIPLE="${EJIT_DEFAULT_TARGET_TRIPLE}")
  auto BareEPC = std::make_unique<BareMetalEPC>(
      std::make_shared<SymbolStringPool>(),
      Triple(EJIT_DEFAULT_TRIPLE),
      /*PageSize=*/4096,
      /*MemMgr=*/nullptr);
  Builder.setExecutorProcessControl(std::move(BareEPC));

  // 2. 关闭 dlopen 自动符号扫描
  Builder.setLinkProcessSymbolsByDefault(false);

  // 3. ProcessSymbols JITDylib：空 JD 满足 LLJIT 平台初始化要求
  Builder.setProcessSymbolsJITDylibSetup(
      [](LLJIT &J) -> Expected<JITDylibSP> {
        auto &JD = J.getExecutionSession().createBareJITDylib(
            "<Process Symbols>");
        return &JD;
      });

  // 4. 使用 InactivePlatform：跳过 EH 帧注册、atexit 支持、
  //    global ctor/dtor 处理。这些在裸核上不需要，且 EH 帧注册
  //    代码在裸核上可能失败。
  Builder.setPlatformSetUp(setUpInactivePlatform);
#endif
```

关于平台设置的选择：

默认的 `setUpGenericLLVMIRPlatform` 会注册 EH 帧（异常处理）、`atexit` 支持、
`llvm.global_ctors/dtors` 处理。这些通过 `callWrapperAsync` 执行，
BareMetalEPC 的同步实现可以工作，但 EH 帧相关的内存分配和注册在裸核上
没有意义且浪费内存。`setUpInactivePlatform` 跳过所有这些，只保留最简平台支持。

`setNumCompileThreads(0)` 不需要在 `#ifdef` 块内重复——当前代码
`EJitOrcEngine.cpp:71` 已无条件设置。此外，使用自定义 EPC 时 LLJIT 要求
`NumCompileThreads` 必须为 0（`LLJIT.cpp:701` 会返回错误）。

### 第二阶段：实现 SlabJITLinkMemoryManager

#### 设计原则

**参考 `InProcessMemoryManager` + `BasicLayout`**。使用 LLVM 内置的
`BasicLayout::getContiguousPageBasedLayoutSizes()` 计算段布局，
参考 `InProcessMemoryManager::allocate` 的 `BL.segments()` 迭代模式分配地址。
保证与 JITLink 链接算法兼容。

#### Slab 结构

```
Base_ →  ┌─────────────────────────────┐
         │  alloc #1 (StandardSegs)     │  ← Used_ = 0 时分配
         │  alloc #1 (FinalizeSegs)     │
         ├─────────────────────────────┤
         │  alloc #2 (StandardSegs)     │  ← Used_ += alignTo(total, PageSize)
         │  alloc #2 (FinalizeSegs)     │
         ├─────────────────────────────┤
         │  ... (bump allocator)        │
         └─────────────────────────────┘
                                       ← Size_ (heap 末尾)
```

#### 类声明

```cpp
class SlabJITLinkMemoryManager : public jitlink::JITLinkMemoryManager {
public:
  SlabJITLinkMemoryManager(char *Base, size_t Size, unsigned PageSize = 4096)
      : Base_(Base), Size_(Size), PageSize_(PageSize) {}

  void allocate(const JITLinkDylib *JD, LinkGraph &G,
                OnAllocatedFunction OnAllocated) override;

  void deallocate(std::vector<FinalizedAlloc> Allocs,
                  OnDeallocatedFunction OnDeallocated) override;

  /// 重置 bump 指针（ejit_clear_cache 调用）。
  /// 警告：调用前必须确保所有 JIT 编译的函数指针已不再被使用，
  /// 否则悬空指针将导致不可预期的行为。
  void reset() { Used_.store(0, std::memory_order_relaxed); }

  size_t used() const { return Used_.load(std::memory_order_relaxed); }
  size_t capacity() const { return Size_; }

private:
  class SlabInFlightAlloc;

  char *Base_;
  size_t Size_;
  unsigned PageSize_;
  std::atomic<size_t> Used_{0};
};
```

#### allocate 实现

参考 `InProcessMemoryManager::allocate`（`JITLinkMemoryManager.cpp:356-448`）：
通过 `BL.segments()` 迭代，按 `MemLifetime` 分流到标准段和最终化段地址区域。

标准段和最终化段放在同一块连续内存中（与 `getContiguousPageBasedLayoutSizes`
的布局一致：标准段在前，最终化段紧随其后）。

```cpp
void SlabJITLinkMemoryManager::allocate(
    const JITLinkDylib *JD, LinkGraph &G,
    OnAllocatedFunction OnAllocated) {

  BasicLayout BL(G);

  auto SegSizes = BL.getContiguousPageBasedLayoutSizes(PageSize_);
  if (!SegSizes) {
    OnAllocated(SegSizes.takeError());
    return;
  }

  size_t TotalSize = SegSizes->total();

  // Bump 分配：从预分配 buffer 中切
  size_t AllocOffset = Used_.fetch_add(alignTo(TotalSize, PageSize_));
  if (AllocOffset + TotalSize > Size_) {
    OnAllocated(make_error<JITLinkError>("JIT heap exhausted"));
    return;
  }

  char *Base = Base_ + AllocOffset;

  // Zero-fill: BasicLayout 的 ZeroFillSize 对应的 block（BSS/零初始化全局变量）
  // 不会被 JITLink 写入内容——依赖内存管理器预先清零。
  // 参考 InProcessMemoryManager::allocate (JITLinkMemoryManager.cpp:403)
  memset(Base, 0, TotalSize);

  // 标准段在前，最终化段在后（匹配 getContiguousPageBasedLayoutSizes 布局）
  auto NextStandardAddr = orc::ExecutorAddr::fromPtr(Base);
  auto NextFinalizeAddr =
      orc::ExecutorAddr::fromPtr(Base + SegSizes->StandardSegs);

  // 参考 InProcessMemoryManager::allocate 的迭代模式
  for (auto &KV : BL.segments()) {
    auto &AG = KV.first;
    auto &Seg = KV.second;

    auto &SegAddr = (AG.getMemLifetime() == orc::MemLifetime::Standard)
                        ? NextStandardAddr
                        : NextFinalizeAddr;

    Seg.WorkingMem = SegAddr.toPtr<char *>();
    Seg.Addr = SegAddr;

    SegAddr += alignTo(Seg.ContentSize + Seg.ZeroFillSize, PageSize_);
  }

  // apply() 将段地址传播回 LinkGraph 的 blocks
  if (auto Err = BL.apply()) {
    OnAllocated(std::move(Err));
    return;
  }

  OnAllocated(std::make_unique<SlabInFlightAlloc>(
      *this, G, Base, TotalSize, AllocOffset));
}
```

#### SlabInFlightAlloc

```cpp
class SlabJITLinkMemoryManager::SlabInFlightAlloc
    : public JITLinkMemoryManager::InFlightAlloc {
public:
  SlabInFlightAlloc(SlabJITLinkMemoryManager &Parent, LinkGraph &G,
                    char *TargetMem, size_t Size, size_t AllocOffset)
      : Parent(Parent), G(&G), TargetMem(TargetMem), Size(Size),
        AllocOffset(AllocOffset) {}

  void finalize(OnFinalizedFunction OnFinalized) override {
    // 1. 无 mprotect —— 裸核默认 RWX

    // 2. I-cache flush: AArch64 需要显式同步 data/instruction cache
    //    __builtin___clear_cache:
    //      x86: no-op（自动 coherence）
    //      AArch64: DC CVAU + DSB ISH + IC IVAU + DSB ISH + ISB
    __builtin___clear_cache(TargetMem, TargetMem + Size);

    // 3. setUpInactivePlatform 保证无 finalize actions（无 EH 帧注册、
    //    无 atexit、无 global ctor/dtor）。assert 作为防御性检查——
    //    如果未来切换到 setUpGenericLLVMIRPlatform，这里会立即发现。
    assert(G->allocActions().finalizeActions().empty() &&
           "Unexpected finalize actions with InactivePlatform");
#ifndef NDEBUG
    G = nullptr;  // 标记已 finalize（参考 IPInFlightAlloc）
#endif
    OnFinalized(FinalizedAlloc(ExecutorAddr::fromPtr(TargetMem)));
  }

  void abandon(OnAbandonedFunction OnAbandoned) override {
    // 回退 bump 指针。裸核 EJIT 是单线程（setNumCompileThreads(0) +
    // InPlaceTaskDispatcher），直接 store 是安全的。
    // 注意：如果未来引入并发 allocate，此处需要改为标记-延迟回收。
    Parent.Used_.store(AllocOffset, std::memory_order_relaxed);
#ifndef NDEBUG
    G = nullptr;
#endif
    OnAbandoned(Error::success());
  }

private:
  SlabJITLinkMemoryManager &Parent;
  LinkGraph *G;
  char *TargetMem;
  size_t Size;
  size_t AllocOffset;  // bump 偏移量，用于 abandon 回退
};
```

要点：
- **BL.apply() 只在 allocate 中调用一次**——finalize 中不再调用，BasicLayout 是 allocate 的局部变量
- SlabInFlightAlloc **不持有** BasicLayout（finalize/abandon 不需要它）
- finalize actions：`setUpInactivePlatform` 保证为空（无 EH/atexit/ctor/dtor），`assert(empty())` 作为防御性检查。如需切换到 `setUpGenericLLVMIRPlatform`，需通过 `AllocActions::runFinalize(EPC)` 同步执行（`runFinalizeActions` 是 JITLinkMemoryManager.cpp 内部的静态函数，外部无法调用）
- 裸核上 working memory == target memory（同一块 bump allocator 内存），无需 memcpy
- **不能省 I-cache flush**：AArch64 不会自动同步 data cache → instruction cache
- **abandon 回退 bump 指针**：`Used_.store(AllocOffset)` 将指针回退到本次分配前的位置。依赖单线程保证（裸核 EJIT 不使用并发编译）

#### deallocate

bump allocator 不支持单独释放。但 **必须调用 `Alloc.release()`**，
否则 `FinalizedAlloc` 析构时断言 `A.getValue() == InvalidAddr` 触发。

```cpp
void SlabJITLinkMemoryManager::deallocate(
    std::vector<FinalizedAlloc> Allocs,
    OnDeallocatedFunction OnDeallocated) {
  // Bump allocator: 不释放内存，但必须 release 每个 alloc
  // 以防止 FinalizedAlloc 析构函数触发断言。
  for (auto &Alloc : Allocs)
    (void)Alloc.release();
  OnDeallocated(Error::success());
}
```

整体重置通过 `ejit_clear_cache()` → `SlabJITLinkMemoryManager::reset()`
将 `Used_` 归零。

#### Slab 重置的安全约束

`reset()` 将 bump 指针归零后，之前分配的所有 JIT 代码内存变为"空闲"，
后续 JIT 编译会覆盖这些地址。**调用方必须确保**：

1. `cache_->clear()` 已先执行（移除所有缓存条目，后续调用会重新编译）
2. 外部持有的旧函数指针在 reset 后不再使用（这些指针现在指向将被覆盖的内存）
3. 实践中：`ejit_clear_cache()` 应在所有特化结果都可安全丢弃时调用，
   例如在时间窗口切换（cell 重新配置）之后

当前 `EJit::clearCache()`（`EJit.cpp:147`）只调用 `cache_->clear()`，
需要增加 `SlabJITLinkMemoryManager::reset()` 的调用连接。

### 第三阶段：配置统一

#### C API 方式（推荐，更可测）

扩展 `ejit_config_t` / `Config`。当前 `Config` 中的 `maxCodeMemory` / `maxDataMemory`
是缓存淘汰的**容量上限**（用于判定何时 LRU 淘汰），新增的 `codeHeapStart` / `codeHeapSize`
是 slab 分配器的**内存来源**。两者关系：

- `maxCodeMemory`：JIT cache 中所有条目 bitcode 大小的上限（软限制，触发 LRU 淘汰）
- `codeHeapStart` / `codeHeapSize`：Slab bump allocator 的实际内存块（硬限制，满了 JIT 编译失败）
- 建议 `codeHeapSize >= maxCodeMemory * 4`（经验值，bitcode → 机器码通常膨胀 2-5x，加 BasicLayout padding。不是严格保证，实际膨胀取决于优化级别和 inline 行为）

```c
typedef struct {
  ...
  // 现有字段（缓存淘汰限制）
  size_t    maxCodeMemory;
  size_t    maxDataMemory;

  // 新增字段（slab 内存来源，裸核必须）
  void     *codeHeapStart;    // JIT heap 起始地址（NULL = hosted 自动 mmap）
  size_t    codeHeapSize;     // JIT heap 大小
} ejit_config_t;
```

`EJitRuntime.cpp` 中的 `parseConfig()` 需要更新以复制新字段。

数据流：

```
ejit_init(&cfg)
  │
  ├─ parseConfig(): C ejit_config_t → C++ Config
  │
  └─ EJit::EJit(config)
      └─ EJitOrcEngine::Create(config, ...)
          ├─ if (config.codeHeapStart)
          │     MemMgr = SlabJITLinkMemoryManager(
          │         config.codeHeapStart, config.codeHeapSize)
          │  else
          │     MemMgr = nullptr  // BareMetalEPC 内部创建 InProcessMemoryManager
          │                      // （仅限 hosted 测试）
          │
          ├─ BareMetalEPC(SSP, Triple, PageSize, std::move(MemMgr))
          └─ Builder.setExecutorProcessControl(std::move(BareEPC))
```

x86 hosted 测试可用 `malloc` 模拟 slab：

```c
ejit_config_t cfg = {0};
cfg.codeHeapStart = malloc(512 * 1024);
cfg.codeHeapSize  = 512 * 1024;
ejit_init(&cfg);
```

#### 链接脚本方式

裸核上通过链接脚本预定义符号，运行时自动读取：

```ld
PROVIDE(__ejit_code_heap_start = .);
. = . + 512K;
PROVIDE(__ejit_code_heap_end   = .);
```

```c
extern char __ejit_code_heap_start, __ejit_code_heap_end;
cfg.codeHeapStart = &__ejit_code_heap_start;
cfg.codeHeapSize  = &__ejit_code_heap_end - &__ejit_code_heap_start;
```

---

## 4. 实现优先级

| 阶段 | 内容 | 工作量 | 可测性 |
|---|---|---|---|
| 1 | BareMetalEPC + EJitOrcEngine 集成 + x86 测试 | 中 | x86 hosted（传 nullptr MemMgr → 内部 InProcessMemoryManager 占位） |
| 2 | SlabJITLinkMemoryManager + x86 malloc buffer 测试 | 中 | x86 hosted（malloc buffer 替代 mmap） |
| 3 | 配置统一 (C API + 链接脚本) + Slab reset 生命周期 | 小 | x86 + 裸核 |

## 5. 测试策略

1. **x86 hosted — 第一阶段**：注入 `BareMetalEPC`（MemoryAccess、DylibMgr、callWrapperAsync），MemoryManager 暂用 `InProcessMemoryManager`。验证 JIT 编译/cache hit/miss 全部正常。测试重点：`setUpInactivePlatform` 不引入额外依赖、`InPlaceTaskDispatcher` 正确执行 task。

2. **x86 hosted — 第二阶段**：替换为 `SlabJITLinkMemoryManager` + `malloc(512K)` buffer。验证 bump allocator + `BasicLayout` 兼容性、`BL.apply()` 地址传播、deallocate 不崩溃。

3. **aarch64 裸核**：交叉编译 + qemu 运行完整 pipeline。验证 I-cache flush、链接脚本 heap 配置、`writePointersAsync` 32/64 位分支正确性。

---

*文档版本: 1.6*
*创建日期: 2026-06-01*
*更新日期: 2026-06-02*

---

## 附录 A：源码检视记录 (v1.6)

基于对 LLVM 源码的检视，以下 API 细节与初始假设不同：

| 设计文档假设 | 实际 API |
|---|---|
| `MemoryAccess` 是 EPC 嵌套类 | 独立类 `llvm::orc::MemoryAccess`，13 个纯虚方法 |
| `DylibMgr` 类型是 `EPCGenericDylibManager` | `DylibManager *`（抽象基类），`SelfEPC` 用私有继承实现 |
| `callWrapperAsync` 回调是 `SendResultFunction` | `IncomingWFRHandler` |
| `setSetupProcessSymbolsJITDylib` | `setProcessSymbolsJITDylibSetup` |
| `makeExecutable` 在 `MemoryAccess` 上 | 在 `JITLinkMemoryManager::InFlightAlloc::finalize` 中处理（mprotect + I-cache flush） |
| `ExecutorProcessControl` 构造接受 `(SSP, D, MemMgr)` | 只接受 `(SSP, D)`；MemMgr 通过 `MemMgr` 成员设置 |
| `BasicLayout::apply(lambda)` 接受回调 | `BL.apply()` 无参数；地址通过 `BL.segments()` 迭代设置 |
| `writePointersAsync` 写 `ExecutorAddr*` | 必须按 `IsArch64Bit` 分支（32 位目标写 4 字节，64 位写 8 字节） |
| `FinalizedAlloc` 可以随意丢弃 | 析构函数断言 `A == InvalidAddr`，必须调用 `release()` |
| 工作内存 ≠ 目标内存（进程内模型） | 进程内模型两者是**同一块内存**，finalize 中无需 memcpy |

### 参考文件

| 组件 | 文件 |
|---|---|
| `ExecutorProcessControl` | `llvm/include/llvm/ExecutionEngine/Orc/ExecutorProcessControl.h` |
| `MemoryAccess` | `llvm/include/llvm/ExecutionEngine/Orc/MemoryAccess.h` |
| `InProcessMemoryAccess` | `llvm/include/llvm/ExecutionEngine/Orc/InProcessMemoryAccess.h` (line 17-133) |
| `DylibManager` | `llvm/include/llvm/ExecutionEngine/Orc/DylibManager.h` |
| `SelfExecutorProcessControl` | `llvm/include/llvm/ExecutionEngine/Orc/SelfExecutorProcessControl.h` |
| `TaskDispatcher` | `llvm/include/llvm/ExecutionEngine/Orc/TaskDispatch.h` |
| `JITLinkMemoryManager` | `llvm/include/llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h` |
| `InProcessMemoryManager` | `llvm/lib/ExecutionEngine/JITLink/JITLinkMemoryManager.cpp` (line 240-448) |
| `JITLinkGeneric` (linkPhase1) | `llvm/lib/ExecutionEngine/JITLink/JITLinkGeneric.cpp` |
| `LLJIT` (platform setup) | `llvm/lib/ExecutionEngine/Orc/LLJIT.cpp` |
| `LLJITBuilder` | `llvm/include/llvm/ExecutionEngine/Orc/LLJIT.h` |
