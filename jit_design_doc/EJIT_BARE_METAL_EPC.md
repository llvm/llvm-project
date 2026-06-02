# EJIT 裸核 ExecutorProcessControl 设计

**版本**: 1.2
**日期**: 2026-06-02
**关联**: PASS7_EJitRuntime_OrcJITLink.md, EJIT_BARE_METAL_STUBS.md, EJIT_LIBRARY_TRIMMING.md

---

## 1. 动机

`LLJIT` 默认使用 `SelfExecutorProcessControl` + `InProcessMemoryManager`，
依赖 mmap/mprotect/dlopen/sysconf。裸核必须替换。

真正阻塞 JIT 编译的是**默认 SelfExecutorProcessControl** 和**默认
InProcessMemoryManager**。`linkProcessSymbolsByDefault` 只是符号发现
策略的问题，优先级排在 EPC 和 MemoryManager 之后。

## 2. 总体方案

分四阶段实现，每个阶段独立可测。

### 第一阶段：最小侵入

在 `EJitOrcEngine::Create` 的 `#ifdef EJIT_BARE_METAL` 块中集中配置：

```cpp
#ifdef EJIT_BARE_METAL
  // 1. EPC 替换（Phase 2 实现 BareMetalEPC）
  Builder.setExecutorProcessControl(
      std::make_unique<BareMetalEPC>(/*heapCfg*/));

  // 2. 符号查找：关闭 dlopen 自动扫描，改为手动注册
  Builder.setLinkProcessSymbolsByDefault(false);

  // 3. ProcessSymbols JITDylib：空 JD 满足 LLJIT 平台初始化要求
  //    符号由 PASS1 生成的 ejit_register_symbol 调用注入
  Builder.setProcessSymbolsJITDylibSetup([](LLJIT &J) -> Expected<JITDylibSP> {
      auto &JD = J.getExecutionSession().createBareJITDylib("<Process Symbols>");
      return &JD;
  });

  // 4. 单线程
  Builder.setNumCompileThreads(0);
#endif
```

要点：
- `EJIT_DEFAULT_TRIPLE` 必填，当前代码 `EJitOrcEngine.cpp:53` 已有
  `#error EJIT_BARE_METAL requires EJIT_DEFAULT_TRIPLE`
- `setProcessSymbolsJITDylibSetup` 不是裸核必须条件，它只是满足
  LLJIT 平台初始化要求的占位 JD，真正的符号通过 ejit_register_symbol 注入

### 第二阶段：实现 BareMetalEPC

继承 `ExecutorProcessControl`，实现以下接口。

#### 2.1 MemoryAccess

**源码事实**：`MemoryAccess` 是独立类 `llvm::orc::MemoryAccess`（非 EPC 嵌套类），
位于 `llvm/include/llvm/ExecutionEngine/Orc/MemoryAccess.h`。
接口是**异步**的（`writeUInt8sAsync`、`readPointersAsync` 等），
但提供同步便捷包装。参考实现：`InProcessMemoryAccess`。

**不能做纯 no-op**——JITLink finalize 阶段通过 MemoryAccess 写入
relocation fixup 后的最终代码。

```cpp
class BareMetalMemoryAccess : public orc::MemoryAccess {
  // 实现全部 write*Async / read*Async 纯虚方法
  // 参考 InProcessMemoryAccess：直接对 ExecutorAddr 做 memcpy，
  // 然后同步回调 OnComplete
  void writeBuffersAsync(ArrayRef<tpctypes::BufferWrite> Ws,
                         WriteResultFn OnComplete) override {
    for (auto &W : Ws)
      memcpy(W.Addr.template toPtr<char *>(), W.Buffer.data(), W.Buffer.size());
    OnComplete(Error::success());
  }
  // ... writeUInt8sAsync, readPointersAsync 等同理 ...
};

// 裸核默认 RWX，MemoryAccess 不做权限切换。
// mprotect 语义在 JITLinkMemoryManager::InFlightAlloc::finalize 中处理（见 3.3）。

#### 2.2 DylibMgr

**源码事实**：EPC 的 `DylibMgr` 字段类型是 `DylibManager *`（独立抽象类），
位于 `llvm/include/llvm/ExecutionEngine/Orc/DylibManager.h`。
`SelfExecutorProcessControl` 通过**私有继承** `DylibManager` 并设
`DylibMgr = this`。`EPCGenericDylibManager` 是远程 EPC 的辅助类，
**不是** `DylibManager` 的子类。

裸核参考 `SelfExecutorProcessControl` 模式，私有继承 `DylibManager`
并返回错误：

```cpp
class BareMetalEPC : public ExecutorProcessControl,
                     private DylibManager {  // 参考 SelfEPC 模式
  // DylibManager 实现：裸核无动态库
  Expected<tpctypes::DylibHandle> loadDylib(const char *DylibPath) override {
    return make_error<StringError>("bare-metal: no dynamic libraries",
                                   inconvertibleErrorCode());
  }
  void lookupSymbolsAsync(ArrayRef<LookupRequest> Request,
                          SymbolLookupCompleteFn F) override {
    // 返回空结果
    ...
  }
};
```

#### 2.3 callWrapperAsync

**源码事实**：`SelfExecutorProcessControl::callWrapperAsync` 实际把
`ExecutorAddr` 强转为函数指针，**同步调用后立即回调** `OnComplete`——
没有 IPC，wrapper 函数就是 in-process 直调。

```cpp
// 实际接口签名（来自 ExecutorProcessControl.h:224）：
// virtual void callWrapperAsync(ExecutorAddr WrapperFnAddr,
//                               IncomingWFRHandler OnComplete,
//                               ArrayRef<char> ArgBuffer) = 0;

void BareMetalEPC::callWrapperAsync(ExecutorAddr WrapperFnAddr,
                                    IncomingWFRHandler OnComplete,
                                    ArrayRef<char> ArgBuffer) override {
  using WrapperFnTy = shared::CWrapperFunctionResult (*)(const char *, size_t);
  auto *WrapperFn = WrapperFnAddr.toPtr<WrapperFnTy>();
  shared::WrapperFunctionResult R = WrapperFn(ArgBuffer.data(), ArgBuffer.size());
  OnComplete(std::move(R));
}
```

裸核和 SelfEPC 一样走同步直调——没有远程执行器需要 IPC。

#### 2.4 runAsMain / runAsVoidFunction / runAsIntFunction

裸核直接调用函数指针即可。LLJIT 平台代码（COFFPlatform/MachOPlatform）
已随格式裁剪排除，不会被调用。

#### 2.5 getPageSize

硬编码 4096。

### 第三阶段：实现 SlabJITLinkMemoryManager

#### 3.1 设计原则

**参考 `InProcessMemoryManager` + `BasicLayout`**，而不是手写简化 allocate。
使用 `BasicLayout::getContiguousPageBasedLayoutSizes(PageSize)` 计算段布局，
保证与 JITLink 的链接算法兼容。

#### 3.2 allocate

```cpp
void allocate(const JITLinkDylib *JD, LinkGraph &G,
              OnAllocatedFunction OnAllocated) override {
  BasicLayout BL(G);
  auto PageSize = 4096;
  auto SegSizes = BL.getContiguousPageBasedLayoutSizes(PageSize);
  size_t TotalSize = SegSizes.total();

  // bump allocator：从预分配 buffer 中切
  size_t AllocOffset = Used_.fetch_add(alignTo(TotalSize, PageSize));
  if (AllocOffset + TotalSize > Size_)
    return OnAllocated(make_error<JITLinkError>("JIT heap exhausted"));

  char *WorkingMem = Base_ + AllocOffset;
  char *TargetMem = Base_ + AllocOffset;

  // 应用 BasicLayout
  BL.apply([&](auto &Seg) {
    Seg.WorkingMem = WorkingMem + Seg.Offset;
    Seg.Addr = orc::ExecutorAddr::fromPtr(TargetMem + Seg.Offset);
  });

  auto FA = std::make_unique<SlabFinalizedAlloc>(std::move(BL), *this, AllocOffset);
  OnAllocated(std::move(FA));
}

#### 3.3 finalize + instruction cache flush

`mprotect` 替换为 no-op（裸核 RWX），但 **AArch64 必须做 instruction cache flush**：

```cpp
Error finalize(std::unique_ptr<FinalizedAlloc> FA) override {
  auto *SFA = static_cast<SlabFinalizedAlloc *>(FA.get());
  // AArch64: CPU 不会自动感知 data cache 写入 → instruction cache
  // 必须显式 flush 再执行 JIT 代码
  __builtin___clear_cache(SFA->targetAddr(), SFA->targetAddr() + SFA->size());
  return Error::success();
}
```

`__builtin___clear_cache` 是 GCC/Clang 内置，AArch64 上映射到正确的
cache maintenance 指令序列。x86 上为 no-op（x86 有 coherency）。

#### 3.4 deallocate

bump allocator 不支持单独释放。整体重置通过 `clearCache()` 调用时
重置 `Used_` 计数器。文档中说明 slab 回收与 LRU 的关系：
- JIT cache 满时按 LRU 淘汰条目
- 淘汰只清除 cache entry，不释放 slab 内存
- `ejit_clear_cache()` 清空 cache + 重置 slab 指针

### 第四阶段：配置统一

#### 4.1 C API 方式（推荐，更可测）

扩展 `ejit_config_t` / `Config`：

```c
typedef struct {
  ...
  void     *codeHeapStart;    // JIT 代码堆起始地址
  size_t    codeHeapSize;     // JIT 代码堆大小
  void     *dataHeapStart;    // JIT 数据堆起始地址（rodata+data）
  size_t    dataHeapSize;     // JIT 数据堆大小
} ejit_config_t;

x86 hosted 测试可用 `malloc` buffer 模拟 slab：

```c
ejit_config_t cfg = {0};
cfg.codeHeapStart = malloc(512 * 1024);
cfg.codeHeapSize  = 512 * 1024;
ejit_init(&cfg);
```

#### 4.2 链接脚本方式

裸核上通过链接脚本预定义符号，运行时自动读取：

```ld
PROVIDE(__ejit_code_heap_start = .);
. = . + 512K;
PROVIDE(__ejit_code_heap_end   = .);

```c
extern char __ejit_code_heap_start, __ejit_code_heap_end;
// 自动计算 heapStart/heapSize
```

## 3. 实现优先级

| 阶段 | 内容 | 工作量 | 可测性 |
|---|---|---|---|
| 1 | EJitOrcEngine 中 EPC 切换 + ProcessSymbols setup | 小 | x86 可测 |
| 2 | BareMetalEPC + MemoryAccess + DylibMgr | 中 | x86 可测 |
| 3 | SlabJITLinkMemoryManager | 中 | x86 可测(malloc buffer) |
| 4 | 配置统一 (C API + 链接脚本) | 小 | x86+裸核 |

## 4. 测试策略

1. **x86 hosted 测试**：将 `BareMetalEPC` 作为 `SelfExecutorProcessControl`
   的替代注入，使用 `malloc` 分配的 buffer 模拟 slab。验证 JIT 编译/执行/cache
   全部正常。
2. **aarch64 裸核测试**：交叉编译 + qemu 运行。

---

*文档版本: 1.2*
*创建日期: 2026-06-01*
*更新日期: 2026-06-02*

---

## 附录 A：源码检视记录 (v1.2)

基于 2026-06-02 对 LLVM 源码的检视，修正了以下 API 假设：

| 设计文档假设 | 实际 API |
|---|---|
| `MemoryAccess` 是 EPC 嵌套类 | 独立类 `llvm::orc::MemoryAccess`，async 接口 (`writeUInt8sAsync` 等) |
| `BareMetalMemoryAccess` 重写 `writeToMemory`/`readFromMemory` | 实际需重写 `writeBuffersAsync`/`readPointersAsync` 等 13 个纯虚方法 |
| `DylibMgr` 类型是 `EPCGenericDylibManager` | 实际是 `DylibManager *`（抽象基类），`SelfEPC` 用私有继承实现 |
| `setSetupProcessSymbolsJITDylib` | 实际是 `setProcessSymbolsJITDylibSetup` |
| `callWrapperAsync` 回调是 `SendResultFunction` | 实际是 `IncomingWFRHandler` |
| `makeExecutable` 在 `MemoryAccess` 上 | 实际在 `JITLinkMemoryManager::InFlightAlloc::finalize` 中处理 |

### 参考文件

| 组件 | 文件 |
|---|---|
| `ExecutorProcessControl` | `llvm/include/llvm/ExecutionEngine/Orc/ExecutorProcessControl.h` |
| `MemoryAccess` | `llvm/include/llvm/ExecutionEngine/Orc/MemoryAccess.h` |
| `InProcessMemoryAccess` | `llvm/include/llvm/ExecutionEngine/Orc/InProcessMemoryAccess.h` |
| `DylibManager` | `llvm/include/llvm/ExecutionEngine/Orc/DylibManager.h` |
| `SelfExecutorProcessControl` | `llvm/include/llvm/ExecutionEngine/Orc/SelfExecutorProcessControl.h` |
| `JITLinkMemoryManager` | `llvm/include/llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h` |
| `InProcessMemoryManager` | `llvm/include/llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h` (line 363) |
| `LLJITBuilder` | `llvm/include/llvm/ExecutionEngine/Orc/LLJIT.h` |
