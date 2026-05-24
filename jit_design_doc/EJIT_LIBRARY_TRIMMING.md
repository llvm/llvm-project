# EmbeddedJIT 运行时库裁剪设计文档

**版本**: 2.0
**日期**: 2026-05-24
**关联**: SPEC4.md, PASS7_EJitRuntime_OrcJITLink.md
**目标**: 提供 X86_64 / AArch64 裸核环境下 EJIT 运行时的最小 .a 集合

---

## 1. 裸核环境约束

| 约束 | 说明 |
|---|---|
| 无 OS | 无 mmap/munmap/mprotect，无 dlopen/dlsym，无 pthread |
| 无文件系统 | 无 fopen/read/write，bitcode 从内存加载 |
| 静态链接 | 所有符号编译期确定，无动态库 |
| 单线程 | 默认同步编译模式，无后台线程 |
| 内存受限 | RAM 100 KB – 2 MB，Flash 对应中等受限 |

---

## 2. 当前现状

### 2.1 链接概览

EJIT 测试二进制当前链接 **53 个** LLVM `.a` 文件，最终体积 **~30 MB**（gc-sections + strip 后）。其中 EJIT 自身仅 **100 KB**，LLVM 基础设施占 99.7%。

各库实际链接大小（降序）：

| 库 | 链接大小 | OS 依赖 | 裸核可用 |
|---|---|---|---|
| libLLVMCodeGen.a | 4,866 KB | 无 | ✓ |
| libLLVMX86CodeGen.a | 4,600 KB | 无 | ✓ |
| libLLVMSelectionDAG.a | 3,066 KB | 无 | ✓ |
| libLLVMAnalysis.a | 2,611 KB | 无 | ✓ |
| libLLVMX86Desc.a | 2,535 KB | 无 | ✓ |
| libLLVMCore.a | 2,454 KB | 无 | ✓ |
| libLLVMTransformUtils.a | 1,094 KB | 无 | ✓ |
| libLLVMInstCombine.a | 1,089 KB | 无 | ✓ |
| libLLVMSupport.a | 690 KB | **mmap/pthread/dlsym/fs** | 需桩函数 |
| libLLVMAsmPrinter.a | 661 KB | 无 | ✓ |
| libLLVMGlobalISel.a | 641 KB | 无 | ✓ |
| libLLVMMC.a | 603 KB | 无 | ✓ |
| libLLVMJITLink.a | 525 KB | **mmap/mprotect** | 需自定义内存管理器 |
| libLLVMScalarOpts.a | 479 KB | 无 | ✓ |
| libLLVMObject.a | 375 KB | 无（文件 I/O 非必需路径） | ✓ |
| libLLVMOrcJIT.a | 346 KB | **mutex/dlsym** | 需补丁 |
| libLLVMBitReader.a | 271 KB | 无 | ✓ |
| libLLVMProfileData.a | 257 KB | 无 | 仅需部分 |
| libLLVMBitWriter.a | 244 KB | 无 | ✗ 不需要（仅 AOT） |
| libLLVMPasses.a | 184 KB | 无 | ✗ 替换为 EJitPassBuilder |
| libLLVMInstrumentation.a | 149 KB | 无 | ✗ 不需要 |
| libLLVMipo.a | 148 KB | 无 | 仅需 AlwaysInliner |
| libLLVMRuntimeDyld.a | 132 KB | 无 | ✗ JITLink 替代 |
| libLLVMDebugInfoCodeView.a | 119 KB | 无 | ✗ 裸核无调试 |
| libLLVMEJIT.a | 100 KB | **mmap（内存管理器桩）** | 需补丁 |
| libLLVMCGData.a | 74 KB | 无 | ✗ 不需要 |
| libLLVMTargetParser.a | 68 KB | 无 | ✓ |
| libLLVMVectorize.a | 48 KB | 无 | ✗ 不需要 |
| libLLVMObjCARCOpts.a | 28 KB | 无 | ✗ 不需要 |
| libLLVMBinaryFormat.a | 28 KB | 无 | ✓ |
| libLLVMDebugInfoDWARF.a | 4 KB | 无 | ✗ 裸核无调试 |
| libLLVMOption.a | 12 KB | 无 | ✗ COFF 指令解析不需要 |
| libLLVMDebugInfoDWARFLowLevel.a | 14 KB | 无 | ✗ 裸核无调试 |
| libLLVMAsmParser.a | 3 KB | 无 | ✗ 不解析汇编 |
| libLLVMFrontendOpenMP.a | 3 KB | 无 | ✗ 不需要 |
| libLLVMCoroutines.a | 0.3 KB | 无 | ✗ 不需要 |
| libLLVMCFGuard.a | 5 KB | 无 | ✗ 不需要 |
| libLLVMMCDisassembler.a | 0.3 KB | 无 | ✗ 不需要 |
| libLLVMObjectYAML.a | 1 KB | 无 | ✗ 不需要 |
| libLLVMLinker.a | 1 KB | 无 | ✗ 不需要 |
| libLLVMIRPrinter.a | 1 KB | 无 | ✗ 不需要 |
| libLLVMAggressiveInstCombine.a | 1 KB | 无 | ✗ 不需要 |
| libLLVMEmbeddedJIT.a | 0.3 KB | 无 | ✗ AOT pass，运行时不需要 |
| libLLVMRemarks.a | 1 KB | 无 | ✗ 不需要 |
| libLLVMDemangle.a | 208 KB | 无 | ✓（OrcJIT 符号解析需要） |
| libLLVMBitstreamReader.a | 16 KB | 无 | ✓（BitReader 依赖） |
| libLLVMCodeGenTypes.a | 6 KB | 无 | ✓（X86Desc 依赖） |
| libLLVMExecutionEngine.a | 5 KB | 无 | ✓（OrcJIT 基类） |
| libLLVMOrcShared.a | 7 KB | 无 | ✓（OrcJIT 依赖） |
| libLLVMOrcTargetProcess.a | 380 KB | **dlopen/dlsym/mmap** | ✗ 裸核不需要 |
| libLLVMTarget.a | 11 KB | 无 | ✓ |
| libLLVMX86Info.a | 6 KB | 无 | ✓ |

### 2.2 依赖链关键放大器

**放大器 #1: Passes**（声明 19 个 LINK_COMPONENTS）

```
Passes → AggressiveInstCombine, CFGuard, CodeGen, Coroutines,
         GlobalISel, Instrumentation, IRPrinter, ObjCARC, Vectorize,
         EmbeddedJIT (→ IRReader → AsmParser), ...
```

EJIT 只用 8 个 pass（InstCombine, SCCP, ADCE, SimplifyCFG, LoopUnroll, LoopSimplify, AlwaysInliner, Promote/Mem2Reg），但 PassBuilder 拉入了 100+ 个 pass 的注册。

**放大器 #2: X86CodeGen**（拉入完整代码生成管线）

```
X86CodeGen → SelectionDAG, AsmPrinter (→ DebugInfoDWARF, CodeView),
             GlobalISel, Instrumentation, ProfileData, ...
```

**放大器 #3: OrcJIT**（拉入 dlsym/mmap 依赖）

```
OrcJIT → OrcTargetProcess (dlopen/dlsym/mmap — 裸核不可用)
       → SelfExecutorProcessControl (dlsym, sysconf)
       → InProcessMemoryMapper (mmap/mprotect)
```

**放大器 #4: IPO + Scalar**（EJitPassBuilder 方案未处理的传递依赖）

```
IPO    → Linker, Instrumentation, Vectorize, BitWriter
Scalar → AggressiveInstCombine
```

移除 Passes 后，EJIT 仍需依赖 IPO（AlwaysInliner）和 Scalar（SCCP/ADCE/SimplifyCFG/LoopUnroll）。但这两个组件在 CMake 中声明了自己的 LINK_COMPONENTS，会通过 `llvm_map_components_to_libnames` 递归解析，把 Vectorize、Instrumentation、Linker、BitWriter、AggressiveInstCombine 重新拉回链接命令。**这意味着仅靠 EJitPassBuilder 替换 PassBuilder 不足以实现最小 .a 集**——IPO 和 Scalar 成为新的传递依赖放大器。

> **实际影响评估**: gc-sections 会移除这些传递依赖的大部分代码（例如 Vectorize.a 的 5,920 KB 中仅 48.4 KB 被保留），所以 **.text 影响较小**。但 .a 文件仍必须出现在链接命令中，影响部署包大小和链接时间。对于"最小 .a 集"目标，需要额外处理（见 §6.5）。

---

## 3. OS 依赖分析

### 3.1 分类

| 分类 | 库 | 关键 OS API | 裸核替代方案 |
|---|---|---|---|
| **干净** | CodeGen, SelectionDAG, GlobalISel, AsmPrinter, MC, MCParser, Core, Analysis, Scalar, InstCombine, IPO, TransformUtils, Object, BitReader, BitWriter, BitstreamReader, BinaryFormat, Target, TargetParser, CodeGenTypes, Demangle, X86CodeGen, X86Desc, X86Info | 无 | 直接可用 |
| **需桩函数** | Support | mmap/munmap/mprotect, pthread_rwlock, dlopen/dlsym, getpid/getenv, sigaction, filesystem | 见 §4.2 |
| **需补丁** | OrcJIT (Core) | std::mutex (159 处), std::condition_variable | 见 §4.3 |
| **需自定义** | JITLink | sys::Memory (mmap/mprotect) | 自定义 JITLinkMemoryManager |
| **需补丁** | EJIT (OrcEngine) | 未禁用 LLJIT 的 dlsym 进程符号查找 | 见 §4.4 |
| **需补丁** | EJIT (MemoryMgr) | mmap/munmap（当前为桩函数，但未接入 LLJIT） | 见 §4.5 |
| **排除** | OrcTargetProcess | dlopen/dlsym, mmap, shm_open, pthread | 裸核不链接 |

### 3.2 Support 库 OS 依赖详情

| 源文件 | OS API | 裸核替代 |
|---|---|---|
| `Unix/Memory.inc` | mmap, munmap, mprotect, sysconf | 预分配静态缓冲区 + no-op mprotect |
| `Unix/RWMutex.inc` | pthread_rwlock_* | `LLVM_ENABLE_THREADS=0` 时已自动 no-op |
| `Unix/Threading.inc` | pthread_create, sysconf | `LLVM_ENABLE_THREADS=0` 时已自动 no-op |
| `Unix/DynamicLibrary.inc` | dlopen, dlclose, dlsym | 返回错误或使用静态符号表 |
| `Unix/Process.inc` | getpid, getenv, sysconf | 返回常量 / nullptr |
| `Signals.cpp` | sigaction, signal | no-op |
| `FileSystem.cpp` | stat, open, read, write, mkdir... | 确保不触达（EJIT 仅用内存缓冲区） |
| `ErrorHandling.cpp` | std::mutex | 单线程 no-op |

**关键发现**: `LLVM_ENABLE_THREADS=0` 已使 RWMutex 和 Threading 的实现自动 no-op，但 OrcJIT Core.cpp 直接使用 `std::mutex` / `std::condition_variable`（不经过 `llvm::sys::MutexImpl`），需要额外处理。

### 3.3 EJIT 现有裸核缺陷

| 缺陷 | 严重度 | 说明 |
|---|---|---|
| LLJIT 未禁用 dlsym 进程符号查找 | **阻塞** | `EJitOrcEngine::Create` 未调用 `setLinkProcessSymbolsByDefault(false)`，LLJIT 默认创建 `EPCDynamicLibrarySearchGenerator` 调用 dlsym，裸核下创建引擎即崩溃 |
| EJitJITLinkMemoryManager 使用 mmap | **阻塞** | 构造函数调用 mmap 分配 slab，裸核不可用 |
| EJitJITLinkMemoryManager::allocate() 是桩函数 | **功能缺失** | 未接入 LLJIT，实际使用默认内存管理器（也用 mmap） |
| std::mutex 在 5+ 个 EJIT 组件中使用 | **需适配** | EJitJITLinkMemoryManager, EJitRuntimeState, EJitRegistrationStore, EJitLogger, EJitCache |
| 无 EJIT_BARE_METAL 宏 | **设计缺失** | 无条件编译路径区分裸核/hosted |

---

## 4. 裸核适配方案

### 4.1 新增 EJIT_BARE_METAL 宏

```cmake
# CMake 选项
option(EJIT_BARE_METAL "Build EJIT for bare-metal (no OS)" OFF)
```

所有裸核适配代码均通过 `#ifdef EJIT_BARE_METAL` / `#if EJIT_BARE_METAL` 守卫。

### 4.2 Support 库裸核桩函数

**策略**: 提供 `llvm/lib/Support/BareMetal/` 目录，包含裸核实现。CMake 在 `EJIT_BARE_METAL=ON` 时替换 Unix 实现。

```
llvm/lib/Support/BareMetal/
├── Memory.inc        # 替代 Unix/Memory.inc
├── DynamicLibrary.inc # 替代 Unix/DynamicLibrary.inc
├── Process.inc       # 替代 Unix/Process.inc
├── Signals.inc       # 替代 Unix/Signals.inc (no-op)
└── Threading.inc     # 替代 Unix/Threading.inc (no-op)
```

**Memory.inc** 核心实现：

```cpp
// 预分配静态缓冲区，替代 mmap
namespace {
  // 外部定义的代码/数据区域，由链接脚本或启动代码提供
  extern "C" {
    extern char __ejit_code_start[];
    extern char __ejit_code_end[];
    extern char __ejit_data_start[];
    extern char __ejit_data_end[];
  }
}

namespace llvm::sys {

MemoryBlock Memory::allocateMappedMemory(size_t NumBytes,
                                          const MemoryBlock *NearBlock,
                                          unsigned Flags,
                                          std::error_code &EC) {
  EC = std::error_code();
  // 裸核: 从预分配区域分配（bump allocator）
  static char *codeCursor = __ejit_code_start;
  static char *dataCursor = __ejit_data_start;

  char *&cursor = (Flags & MF_EXEC) ? codeCursor : dataCursor;
  char *end = (Flags & MF_EXEC) ? __ejit_code_end : __ejit_data_end;

  // 对齐到 4096
  uintptr_t align = (uintptr_t)cursor & 4095;
  if (align) cursor += (4096 - align);

  if (cursor + NumBytes > end) {
    EC = std::make_error_code(std::errc::not_enough_memory);
    return MemoryBlock();
  }

  void *base = cursor;
  cursor += NumBytes;
  return MemoryBlock(base, NumBytes);
}

std::error_code Memory::protectMappedMemory(const MemoryBlock &Block,
                                             unsigned Flags) {
  // 裸核: 无 mprotect，区域在链接脚本中已设置为 RWX
  // AArch64 需要维护页表权限时在此扩展
  return std::error_code();
}

std::error_code Memory::releaseMappedMemory(const MemoryBlock &M) {
  // 裸核: 不释放（bump allocator，仅整体重置）
  return std::error_code();
}

} // namespace llvm::sys
```

**链接脚本片段**（用户提供）：

```ld
  __ejit_code_start = .;
  .ejit_code (NOLOAD) : { . = . + 384K; }  /* 代码区 384KB */
  __ejit_code_end = .;
  __ejit_data_start = .;
  .ejit_data (NOLOAD) : { . = . + 128K; }  /* 数据区 128KB */
  __ejit_data_end = .;
```

**其他桩函数**：

```cpp
// DynamicLibrary.inc
DynamicLibrary DynamicLibrary::getPermanentLibrary(const char *, std::string *) {
  return DynamicLibrary();  // 裸核无动态库
}
void *DynamicLibrary::getAddressOfSymbol(const char *) {
  return nullptr;  // 通过 ejit_register_symbol 替代
}

// Process.inc
unsigned Process::getPageSizeEstimate() { return 4096; }
char *Process::getenv(const char *) { return nullptr; }
unsigned Process::getProcessId() { return 1; }
```

### 4.3 OrcJIT mutex 适配

**问题**: OrcJIT Core.cpp 等文件直接使用 `std::mutex`（~159 处），不经过 `llvm::sys::MutexImpl`，`LLVM_ENABLE_THREADS=0` 不影响它们。

**方案 A: 提供 freestanding mutex（推荐）**

在 EJIT 目录下提供裸核 C++ 运行时支持头文件：

```cpp
// llvm/lib/ExecutionEngine/EJIT/BareMetal/mutex.h
// 通过 -isystem 注入到编译路径，覆盖标准库 <mutex>

#pragma once
#include <system_error>

namespace std {

class mutex {
public:
  void lock() {}      // 裸核单线程，no-op
  void unlock() {}    // no-op
  bool try_lock() { return true; }
};

class recursive_mutex {
public:
  void lock() {}
  void unlock() {}
  bool try_lock() { return true; }
};

class condition_variable {
public:
  void notify_one() {}
  void notify_all() {}
  // wait 在同步模式下不会实际等待（InPlaceTaskDispatcher）
  template<typename Lock>
  void wait(Lock &) {}
};

template<typename Lock>
class unique_lock {
public:
  explicit unique_lock(Lock &m) : m_(m) { m_.lock(); }
  ~unique_lock() { m_.unlock(); }
private:
  Lock &m_;
};

class shared_mutex {
public:
  void lock() {}
  void unlock() {}
  void lock_shared() {}
  void unlock_shared() {}
  bool try_lock() { return true; }
  bool try_lock_shared() { return true; }
};

} // namespace std
```

通过 CMake 注入：

```cmake
if(EJIT_BARE_METAL)
  # 编译 OrcJIT/JITLink 时注入裸核 mutex 头文件
  target_include_directories(LLVMOrcJIT SYSTEM BEFORE PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/EJIT/BareMetal)
  target_include_directories(LLVMJITLink SYSTEM BEFORE PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/EJIT/BareMetal)
  target_include_directories(LLVMSupport SYSTEM BEFORE PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/EJIT/BareMetal)
endif()
```

**方案 B: 补丁 OrcJIT 源码**

将 `std::mutex` 替换为 `llvm::sys::MutexImpl`（已受 `LLVM_ENABLE_THREADS` 控制）。工作量大（~159 处修改），侵入性强，不推荐。

**方案 C: 编译器级别**

使用 `-DLLVM_ENABLE_THREADS=OFF` + 提供空 `libpthread.a` + 在 OrcJIT 编译时定义 `_GLIBCXX_HAS_GTHREADS=0`。可行性取决于工具链，风险高。

### 4.3.1 裸核工具链现实：hosted vs freestanding

LLVM **不能在真正的 freestanding C++ 模式下编译**——它大量使用 STL（`std::vector`、`std::map`、`std::string`、`std::iostream` 等），这些在 freestanding 中不可用。因此裸核方案的实际形态是：

**使用 hosted 模式工具链 + OS 层桩函数**，而非真正的 freestanding。

| 方案 | 工具链 | C++ 标准库 | OS 接口 |
|---|---|---|---|
| **推荐** | `x86_64-elf-g++` hosted 模式 | 完整 libstdc++（含 `<mutex>`） | 桩函数库提供 mmap/pthread 等（§4.2） |
| 备选 | `aarch64-none-elf-g++` + newlib | newlib + libstdc++（部分 STL） | 需额外移植 `<mutex>` 到 newlib |
| 不推荐 | 真正 freestanding | 仅 `<cstddef>`, `<limits>` 等 | LLVM 无法编译 |

**推荐方案的逻辑**：
- 使用标准 hosted 交叉编译器（如 `x86_64-elf-g++`），它提供完整的 libstdc++ 包括 `<mutex>`
- `std::mutex` 的底层实现依赖 `libpthread`——提供空 `libpthread.a` 桩（所有函数返回成功）
- mmap/mprotect/dlopen 等 OS 调用由 §4.2 的 Support/BareMetal 桩函数处理
- 这样既不需要覆盖 `<mutex>` 头文件（方案 A），也不需要 patch OrcJIT 源码（方案 B）

```bash
# 裸核工具链配置示例
CC="x86_64-elf-g++"
CXX="x86_64-elf-g++"

# hosted 模式交叉编译器 + 自建 libpthread 桩
${CXX} -nostdlib \
  libLLVMEJIT.a ... \
  libpthread_stub.a \   # 空实现 pthread_*
  /path/to/libstdc++.a \
  /path/to/libc.a \
  -T bare_metal.ld
```

**libpthread 桩函数**（`libpthread_stub.a`）：

```cpp
// 单线程裸核: 所有 pthread 函数返回成功或 0
extern "C" {
int pthread_mutex_lock(void *m)   { return 0; }
int pthread_mutex_unlock(void *m) { return 0; }
int pthread_mutex_init(void *m, const void *) { return 0; }
int pthread_mutex_destroy(void *m) { return 0; }
int pthread_cond_wait(void *c, void *m) { return 0; }
int pthread_cond_signal(void *c)  { return 0; }
int pthread_create(...)           { return 0; }
int pthread_join(...)             { return 0; }
int pthread_once(int *o, void (*f)(void)) { if (!*o) { *o = 1; f(); } return 0; }
int pthread_rwlock_rdlock(void *r) { return 0; }
int pthread_rwlock_wrlock(void *r) { return 0; }
int pthread_rwlock_unlock(void *r) { return 0; }
// ... 其余 pthread 函数
}
```

> **§4.3 方案 A（注入 mutex.h）降级为备选**：仅在工具链不提供 `<mutex>` 头文件时使用（如某些 bare-metal 工具链的 libstdc++ 配置了 `--disable-threads`）。此时 pthread 桩函数不够（因为 `<mutex>` 本身不存在），需要注入头文件。

### 4.4 EJitOrcEngine 裸核补丁

```cpp
// EJitOrcEngine::Create 修改:
Expected<std::unique_ptr<EJitOrcEngine>>
EJitOrcEngine::Create(const Config &config, ...) {
  // ...
  orc::LLJITBuilder Builder;
  Builder.setJITTargetMachineBuilder(*JTMB);
  Builder.setNumCompileThreads(0);

#ifdef EJIT_BARE_METAL
  // 裸核: 禁用 dlsym 进程符号查找
  Builder.setLinkProcessSymbolsByDefault(false);

  // 裸核: 注入自定义内存管理器（替代默认的 mmap 实现）
  auto *memMgr = getOrCreateBareMetalMemoryManager(config);
  Builder.setJITLinkMemoryManager(
      std::unique_ptr<jitlink::JITLinkMemoryManager>(memMgr));
#endif

  auto J = Builder.create();
  // ...
}
```

### 4.5 EJitJITLinkMemoryManager 裸核接入

当前状态: `allocate()` 是桩函数，`getMemoryManager()` 返回 `nullptr`。需要：

1. 实现 `allocate()`，从预分配 slab 分配
2. 将其实例注入 LLJIT（通过 `Builder.setJITLinkMemoryManager`）
3. 移除 mmap/munmap 调用

```cpp
// EJitJITLinkMemoryManager.cpp — 裸核改造
#ifdef EJIT_BARE_METAL

// 裸核: 使用链接脚本提供的静态区域
extern "C" {
  extern char __ejit_code_start[];
  extern char __ejit_code_end[];
  extern char __ejit_data_start[];
  extern char __ejit_data_end[];
}

EJitJITLinkMemoryManager::EJitJITLinkMemoryManager(size_t, size_t, uint64_t) {
  codeSlab_.baseAddr = __ejit_code_start;
  codeSlab_.totalSize = __ejit_code_end - __ejit_code_start;
  dataSlab_.baseAddr = __ejit_data_start;
  dataSlab_.totalSize = __ejit_data_end - __ejit_data_start;
}

EJitJITLinkMemoryManager::~EJitJITLinkMemoryManager() {
  // 裸核: 不释放
}

#else

// 原有 mmap 实现
static void *allocateSlab(size_t size) { return mmap(...); }
// ...

#endif
```

### 4.6 EJIT 内部 mutex 处理

EJIT 自身使用 `std::mutex` 的 5 个组件：

| 组件 | 位置 | 用途 | 裸核处理 |
|---|---|---|---|
| EJitJITLinkMemoryManager | SlabRegion::mutex | slab 分配锁 | no-op（单线程） |
| EJitRuntimeState | mutex_ | activate/deactivate 状态锁 | no-op |
| EJitRegistrationStore | mutex_ | 注册数据锁 | no-op（constructor 阶段单线程） |
| EJitLogger | mutex_ | 日志写入锁 | no-op |
| EJitCache | mutex_ / shared_mutex | 缓存读写锁 | no-op |

**方案**: 与 §4.3 相同，通过注入 `BareMetal/mutex.h` 统一处理。

### 4.7 JITLink 裸核适配注意事项

JITLink 的设计目标是跨平台 JIT 链接，默认面向 hosted 环境。裸核下需关注以下额外适配点：

**ELF 重定位处理**: JITLink 的 ELF 后端（`ELF_x86_64` / `ELF_aarch64`）在解析重定位时假设目标平台的 ELF ABI 规则。裸核环境的差异（如 TLS 模型、GOT/PLT 布局等）可能导致重定位失败。验证方法：使用 EJIT 在裸核环境编译一个包含重定位的小函数（如调用外部符号），确认 JITLink 能正确处理。

**内存权限**: JITLink 的 `InProcessMemoryMapper`（默认）通过 mmap/mprotect 设置代码段 RWX。裸核替换为 §4.5 的自定义内存管理器后，JITLink 对 section 权限的假设需要与裸核静态区域对齐。X86_64 裸核通常使用 RWX 区域（简单），AArch64 可能需要分离 X 和 RW 区域（取决于 MMU 是否启用）。

**Section 布局**: JITLink 在分配内存时按 section 类型（code/data/readonly）划分区域。自定义 `JITLinkMemoryManager::allocate()` 需要正确映射这些 section（见 §4.5），否则 JITLink 的 `link_StandardSymbols` 等步骤会失败。

> 总体而言 JITLink 的裸核适配工作量不大——主要是内存管理器替换。重定位和 section 布局问题在简单函数上通常不会触发，可在集成测试阶段按需修复。

---

## 5. 最小 .a 集合

### 5.1 裸核最小集（X86_64）

| # | 库 | 链接大小 | 用途 | 是否有条件 |
|---|---|---|---|---|
| 1 | libLLVMCore.a | 2,454 KB | IR 核心 | 必需 |
| 2 | libLLVMSupport.a | 690 KB | 基础支持（需桩函数） | 必需 |
| 3 | libLLVMDemangle.a | 208 KB | OrcJIT 符号解析 | 必需 |
| 4 | libLLVMBinaryFormat.a | 28 KB | ELF 格式 | 必需 |
| 5 | libLLVMBitReader.a | 271 KB | 加载 bitcode | 必需 |
| 6 | libLLVMBitstreamReader.a | 16 KB | BitReader 依赖 | 必需 |
| 7 | libLLVMAnalysis.a | 2,611 KB | 分析 pass | 必需 |
| 8 | libLLVMScalarOpts.a | 479 KB | SCCP/ADCE/SimplifyCFG/LoopUnroll | 必需 |
| 9 | libLLVMInstCombine.a | 1,089 KB | 常量折叠 | 必需 |
| 10 | libLLVMipo.a | 148 KB | AlwaysInliner（仅需此 1 个 pass） | 必需 |
| 11 | libLLVMTransformUtils.a | 1,094 KB | Mem2Reg/LoopSimplify | 必需 |
| 12 | libLLVMCodeGen.a | 4,866 KB | 目标无关后端 | 必需 |
| 13 | libLLVMCodeGenTypes.a | 6 KB | CodeGen 依赖 | 必需 |
| 14 | libLLVMTarget.a | 11 KB | Target 基类 | 必需 |
| 15 | libLLVMTargetParser.a | 68 KB | Triple 解析 | 必需 |
| 16 | libLLVMSelectionDAG.a | 3,066 KB | DAG 指令选择 | 必需（LLJIT 默认路径，与 GlobalISel 二选一） |
| 17 | libLLVMAsmPrinter.a | 661 KB | 代码发射 | 必需 |
| 18 | libLLVMMC.a | 603 KB | MC 层 | 必需 |
| 19 | libLLVMMCParser.a | 186 KB | MC 解析 | **待验证**（JIT 走 MCCodeEmitter 路径，可能不需要 AsmParser） |
| 20 | libLLVMObject.a | 375 KB | 目标文件处理 | 必需 |
| 21 | libLLVMProfileData.a | 257 KB | ItaniumManglingCanonicalizer（OrcJIT 需要） | 需裁剪 |
| 22 | libLLVMExecutionEngine.a | 5 KB | ExecutionEngine 基类 | 必需 |
| 23 | libLLVMOrcJIT.a | 346 KB | JIT 核心（需补丁） | 必需 |
| 24 | libLLVMOrcShared.a | 7 KB | OrcJIT 依赖 | 必需 |
| 25 | libLLVMJITLink.a | 525 KB | JIT 链接（需自定义内存管理器） | 必需 |
| 26 | libLLVMX86CodeGen.a | 4,600 KB | X86 代码生成 | **X86_64 必需** |
| 27 | libLLVMX86Desc.a | 2,535 KB | X86 MC 描述 | **X86_64 必需** |
| 28 | libLLVMX86Info.a | 6 KB | X86 Target 信息 | **X86_64 必需** |
| 29 | libLLVMEJIT.a | 100 KB | EJIT 运行时 | 必需 |

**X86_64 最小集: 29 个 .a，合计 ~22,500 KB 链接大小**（不含 GlobalISel）

> **GlobalISel 说明**: LLJIT 默认使用 SelectionDAG 进行指令选择，GlobalISel 需要显式启用（`setEnableGlobalISel(true)`）。EJIT 不使用 GlobalISel，保留 SelectionDAG 即可。如果后续切换为 GlobalISel-only 模式，可替换但非同时保留。

### 5.2 裸核最小集（AArch64）

将 X86 三项替换为 AArch64 对应库：

| # | 库 | 说明 |
|---|---|---|
| 26 | libLLVMAArch64CodeGen.a | AArch64 代码生成 |
| 27 | libLLVMAArch64Desc.a | AArch64 MC 描述 |
| 28 | libLLVMAArch64Info.a | AArch64 Target 信息 |

额外需添加：

| # | 库 | 说明 |
|---|---|---|
| 30 | libLLVMAArch64Utils.a | AArch64 工具库 |

**AArch64 最小集: 30 个 .a**

### 5.3 可排除的库（23 个）

| 库 | 排除原因 |
|---|---|
| libLLVMPasses.a | 替换为 EJitPassBuilder |
| libLLVMCoroutines.a | 不使用协程 |
| libLLVMObjCARCOpts.a | 不使用 ObjC ARC |
| libLLVMCFGuard.a | 不使用 Windows CFG |
| libLLVMVectorize.a | 不使用向量化 |
| libLLVMAggressiveInstCombine.a | 不使用激进 InstCombine |
| libLLVMInstrumentation.a | 不使用插桩 |
| libLLVMIRPrinter.a | 不需要 IR 打印 |
| libLLVMFrontendOpenMP.a | 不使用 OpenMP |
| libLLVMEmbeddedJIT.a | AOT pass，运行时不需要 |
| libLLVMBitWriter.a | 仅 AOT 编译期需要 |
| libLLVMAsmParser.a | 从 bitcode 加载，不解析汇编 |
| libLLVMIRReader.a | 不需要 IR 文本读取 |
| libLLVMLinker.a | 不做模块链接 |
| libLLVMMCDisassembler.a | 不需要反汇编 |
| libLLVMObjectYAML.a | 不需要 YAML |
| libLLVMOption.a | COFF 指令解析不需要 |
| libLLVMCGData.a | 不需要调用图数据 |
| libLLVMDebugInfoCodeView.a | 裸核无调试 |
| libLLVMDebugInfoDWARF.a | 裸核无调试 |
| libLLVMDebugInfoDWARFLowLevel.a | 裸核无调试 |
| libLLVMRuntimeDyld.a | JITLink 完全替代 |
| libLLVMOrcTargetProcess.a | 使用 dlsym/mmap，裸核不可用 |
| libLLVMGlobalISel.a | **可选排除**（LLJIT 默认用 SelectionDAG；若切换 GlobalISel-only 则保留此、排除 SelectionDAG） |

### 5.4 链接大小对比

| 配置 | .a 数量 | 链接大小 | 与当前对比 |
|---|---|---|---|
| 当前（hosted, 全量） | 53 | ~29 MB | 基准 |
| 裸核最小集（X86_64） | 29 | ~22.5 MB | -22% |
| 裸核最小集（AArch64） | 30 | ~18 MB | -38% |
| + EJitPassBuilder | 29 | ~22 MB | -24% |
| + 裁剪 DebugInfo 发射 | 29 | ~21 MB | -28% |
| + Thin LTO | 29 | ~18 MB | -38% |

> **注意**: 上表 .a 数量不含 C/C++ 运行时（libstdc++/libc/libpthread_stub/libgcc，见 §8.4）及 IPO/Scalar 的传递依赖（见 §6.5）。传递依赖（Vectorize, Instrumentation, Linker, BitWriter, AggressiveInstCombine）的 .text 贡献经 gc-sections 后 <50 KB。

---

## 6. EJitPassBuilder（替代 PassBuilder）

### 6.1 设计

创建 `EJitPassBuilder`，仅注册 EJIT 使用的 ~12 个 analysis + 8 个 pass，替代 `PassBuilder` 的 40+ analysis / 100+ pass 注册。

```cpp
// llvm/include/llvm/ExecutionEngine/EJIT/EJitPassBuilder.h

namespace llvm::ejit {

class EJitPassBuilder {
public:
  EJitPassBuilder();

  void runInstCombine(Module &M);
  void runOptimizationPipeline(Module &M, OptimizationLevel level);

  FunctionAnalysisManager &getFAM() { return FAM_; }
  ModuleAnalysisManager &getMAM() { return MAM_; }

private:
  LoopAnalysisManager LAM_;
  FunctionAnalysisManager FAM_;
  CGSCCAnalysisManager CGAM_;
  ModuleAnalysisManager MAM_;
};

} // namespace llvm::ejit
```

### 6.2 Analysis 注册

推荐策略：先从 `PassBuilder.cpp` 复制完整 analysis 列表，再逐个删除不需要的，用测试验证。不要从零开始补漏——每个 analysis 可能有隐式依赖链。

经过依赖分析的最小集合：

```cpp
void EJitPassBuilder::registerAnalyses() {
  // --- Function Analysis ---
  FAM_.registerPass([&] { return DominatorTreeAnalysis(); });
  FAM_.registerPass([&] { return AssumptionAnalysis(); });
  FAM_.registerPass([&] { return TargetIRAnalysis(); });
  FAM_.registerPass([&] { return TargetLibraryAnalysis(); });
  FAM_.registerPass([&] { return AAManager(); });
  FAM_.registerPass([&] { return BasicAA(); });
  FAM_.registerPass([&] { return ScalarEvolutionAnalysis(); });
  FAM_.registerPass([&] { return LoopAnalysis(); });
  FAM_.registerPass([&] { return MemorySSAAnalysis(); });
  FAM_.registerPass([&] { return PhiValuesAnalysis(); });
  FAM_.registerPass([&] { return MemoryDependenceAnalysis(); });
  FAM_.registerPass([&] { return OptimizationRemarkEmitterAnalysis(); });
  FAM_.registerPass([&] { return TargetTransformInfoAnalysis(); }); // LoopUnroll 需要

  // --- Loop Analysis ---
  LAM_.registerPass([&] { return DominatorTreeAnalysis(); });
  LAM_.registerPass([&] { return LoopAnalysis(); });
  LAM_.registerPass([&] { return ScalarEvolutionAnalysis(); });
  LAM_.registerPass([&] { return TargetTransformInfoAnalysis(); });
  LAM_.registerPass([&] { return SimplifyQueryAnalysis(); });

  // --- Module Analysis ---
  MAM_.registerPass([&] { return PassInstrumentationAnalysis(); });
  MAM_.registerPass([&] { return InlineAdvisorAnalysis(); }); // AlwaysInliner 需要

  // 交叉注册
  FAM_.registerPass([&] { return LoopAnalysisManagerFunctionProxy(LAM_); });
  MAM_.registerPass([&] { return FunctionAnalysisManagerModuleProxy(FAM_); });
  FAM_.registerPass([&] { return ModuleAnalysisManagerFunctionProxy(MAM_); });
  CGAM_.registerPass([&] { return FunctionAnalysisManagerCGSCCProxy(FAM_); });
  CGAM_.registerPass([&] { return ModuleAnalysisManagerCGSCCProxy(MAM_); });
}
```

### 6.3 CMake 修改

```cmake
# llvm/lib/ExecutionEngine/EJIT/CMakeLists.txt
# 移除 Passes，改为直接依赖 EJIT 使用的组件
LINK_COMPONENTS
  Core
  Support
  BitReader
  OrcJIT
  JITLink
  ExecutionEngine
  InstCombine
  TransformUtils
  Analysis
  CodeGen
  ${LLVM_TARGETS_TO_BUILD}
  # 注意: 不在此处声明 IPO 和 Scalar——它们的 LINK_COMPONENTS 会
  # 递归拉入 Vectorize, Instrumentation, Linker, BitWriter, AggressiveInstCombine。
  # 改为在链接命令中直接引用 libLLVMipo.a 和 libLLVMScalarOpts.a（见 §6.5）。
```

### 6.4 ProfileData 裁剪

OrcJIT 需要 `ItaniumManglingCanonicalizer`（来自 ProfileData），但 ProfileData 还包含大量 PGO/覆盖 率代码（257 KB 链接）。方案：

1. **短期**: 直接链接 ProfileData，gc-sections 会移除大部分不需要的代码
2. **长期**: 将 `ItaniumManglingCanonicalizer` 拆分到独立库 `LLVMItaniumMangling`（~50 KB）

### 6.5 IPO / Scalar 传递依赖处理

**问题**: §6.3 中 EJIT 通过 LINK_COMPONENTS 直接依赖 `IPO` 和 `Scalar`，但这两个组件在 CMake 中声明了自己的 LINK_COMPONENTS——IPO 依赖 Linker、Instrumentation、Vectorize、BitWriter；Scalar 依赖 AggressiveInstCombine。`llvm_map_components_to_libnames` 会递归解析，将已标记为"可排除"的库重新拉回链接命令。

**方案 A: 手动链接（推荐短期）**

不使用 CMake LINK_COMPONENTS 的递归解析，改为在链接脚本中显式指定 .a 文件。EJIT 的 CMakeLists.txt 不声明 `IPO` 和 `Scalar` 为 LINK_COMPONENTS，而是在 `ejit_test/build.sh` 的链接命令中直接列出所需的 .a 文件：

```bash
# 不使用 LINK_COMPONENTS 递归，直接显式指定 .a 文件
${CXX} ... \
  -Wl,--whole-archive libLLVMEJIT.a -Wl,--no-whole-archive \
  libLLVMipo.a libLLVMScalarOpts.a \
  # ↑ gc-sections 会自动移除 IPO/Scalar 中不需要的 .o
  # 它们的 LINK_COMPONENTS (Vectorize, Instrumentation, ...) 不会被递归引入
  ...
```

> **注意**: 这取决于链接器的行为。GNU ld / LLD 的 `--gc-sections` 会在 .o 级别做死代码消除，但 `.a` 文件中的未引用 .o 成员不会被提取。所以只要 EJIT 不引用 Vectorize/Instrumentation 中的符号，它们就不会被链接——即使它们的 .a 文件出现在链接命令行中。

**方案 B: 拆分 IPO 组件（长期）**

将 AlwaysInliner 从 IPO 中拆分到独立库 `libLLVMAlwaysInliner.a`（~30 KB），不声明任何 LINK_COMPONENTS。这样做的好处：
- 不再通过 CMake 递归拉入 Linker、Instrumentation、Vectorize、BitWriter
- 适用于所有需要 AlwaysInliner 但不需要其他 IPO pass 的场景

**实际影响**: gc-sections 已能处理大部分冗余代码，因此方案 A 的链接大小影响很小（<50 KB）。方案 B 的主要价值在于减少链接命令行中的 .a 文件数量，使"最小 .a 集"的定义更精确。

---

## 7. 实施计划

### Phase 1: EJitPassBuilder + CMake 依赖清理（2-3 天）

| 步骤 | 工作量 | 风险 |
|---|---|---|
| 创建 EJitPassBuilder.h/.cpp | 0.5 天 | 低 |
| 修改 EJitOptimizer 使用 EJitPassBuilder | 0.5 天 | 低 |
| 修改 CMakeLists.txt 移除 Passes 依赖 | 0.5 天 | 低 |
| 验证: hosted 环境全部 EJIT 测试通过 | 0.5 天 | 低 |
| 验证: 链接的 .a 数量减少（53 → 29） | 0.5 天 | 低 |

**验收标准**: `bash ejit_test/build.sh --run --analyze-deps` 显示 ≤29 个 LLVM .a，全部测试 PASS

### Phase 2: 裸核桩函数 + 补丁（3-5 天）

| 步骤 | 工作量 | 风险 |
|---|---|---|
| 创建 `BareMetal/mutex.h` (freestanding mutex) | 0.5 天 | 低 |
| 创建 `Support/BareMetal/*.inc` 桩函数 | 1 天 | 中 |
| 补丁 EJitOrcEngine: 禁用 dlsym + 注入自定义内存管理器 | 1 天 | 中 |
| 补丁 EJitJITLinkMemoryManager: 从 mmap 改为静态缓冲区 | 1 天 | 中 |
| 补丁 EJIT 内部: 条件编译 EJIT_BARE_METAL 守卫 | 0.5 天 | 低 |
| CMake: EJIT_BARE_METAL 选项 + 注入 include 路径 | 0.5 天 | 低 |
| 交叉编译验证（AArch64 bare-metal 工具链） | 0.5 天 | 中 |

**验收标准**: 使用 `aarch64-none-elf-g++` 交叉编译 EJIT 运行时，无链接错误

### Phase 3: 体积优化（5-7 天）

| 步骤 | 工作量 | 风险 |
|---|---|---|
| 裁剪 AsmPrinter DWARF/CodeView 发射 | 2 天 | 中（需调用点 guard） |
| 裁剪 CodeGen 非必需子模块 | 2 天 | 中高（需梳理 .o 间引用） |
| ProfileData 拆分 | 2 天 | 中 |
| Thin LTO 验证 | 1 天 | 低 |

**验收标准**: AArch64 裸核二进制 < 18 MB

---

## 8. 裸核构建流程

### 8.1 CMake 配置

```bash
# X86_64 裸核
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DEJIT_BARE_METAL=ON \
  -DEJIT_DEFAULT_TARGET_TRIPLE="x86_64-none-elf" \
  -DLLVM_ENABLE_THREADS=OFF \
  -DLLVM_ENABLE_ZLIB=OFF \
  -DLLVM_ENABLE_ZSTD=OFF \
  -DLLVM_ENABLE_TERMINFO=OFF \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  ..

# AArch64 裸核
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD="AArch64" \
  -DEJIT_BARE_METAL=ON \
  -DEJIT_DEFAULT_TARGET_TRIPLE="aarch64-none-elf" \
  -DLLVM_ENABLE_THREADS=OFF \
  -DLLVM_ENABLE_ZLIB=OFF \
  -DLLVM_ENABLE_ZSTD=OFF \
  -DLLVM_ENABLE_TERMINFO=OFF \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  ..
```

### 8.2 编译

```bash
ninja -C build_release_x86 LLVMEJIT
```

### 8.3 链接到用户固件

```bash
aarch64-none-elf-g++ \
  -fuse-ld=lld \
  -Os -Wl,--gc-sections \
  -Wl,--whole-archive libLLVMEJIT.a -Wl,--no-whole-archive \
  -L build_release_aarch64/lib \
  libLLVMCore.a libLLVMSupport.a libLLVMDemangle.a \
  libLLVMBinaryFormat.a libLLVMBitReader.a libLLVMBitstreamReader.a \
  libLLVMAnalysis.a libLLVMScalarOpts.a libLLVMInstCombine.a \
  libLLVMipo.a libLLVMTransformUtils.a libLLVMCodeGen.a \
  libLLVMCodeGenTypes.a libLLVMTarget.a libLLVMTargetParser.a \
  libLLVMSelectionDAG.a libLLVMAsmPrinter.a \
  libLLVMMC.a libLLVMObject.a libLLVMProfileData.a \
  libLLVMExecutionEngine.a libLLVMOrcJIT.a libLLVMOrcShared.a \
  libLLVMJITLink.a \
  libLLVMAArch64CodeGen.a libLLVMAArch64Desc.a \
  libLLVMAArch64Info.a libLLVMAArch64Utils.a \
  libpthread_stub.a \
  -T bare_metal.ld \
  user_code.o -o firmware.elf
```

> **注意**: `libLLVMMCParser.a` 不在链接命令中（待验证 JIT 路径是否需要 AsmParser）。如果需要，链接时会报未定义符号错误，再补回。

### 8.4 C / C++ 运行时依赖

裸核固件链接时还需以下运行时库：

| 库 | 用途 | 说明 |
|---|---|---|
| `libstdc++.a` | C++ STL | LLVM 大量使用 `std::vector/map/string/iostream` |
| `libc.a`（或 `libnewlib.a`/`libpicolibc.a`） | C 标准库 | `memcpy`/`memset` 等以及 LLVM Support 库调用的标准 C 函数 |
| `libpthread_stub.a` | pthread 桩函数 | 所有函数返回成功（见 §4.3.1），单线程裸核 no-op |
| `libgcc.a`（或 `libclang_rt.builtins.a`） | 编译器内置函数 | `__divdi3`/`__udivmoddi4` 等 64 位算术辅助函数 |
| `crt0.o` / `crtbegin.o` / `crtend.o` | C 运行时初始化 | 调用 `.init_array` 中全局构造函数（LLVM 大量使用全局注册） |

**关键**: `.init_array` 必须在 `main()` 或 EJIT 初始化之前被调用。LLVM 的 pass 注册、Target 注册等都依赖全局构造函数。裸核启动代码必须包含：

```assembly
# 简易 crt0.S — 必须在跳转到 C/C++ 代码前执行
_start:
    # ... 初始化栈指针、BSS 清零等 ...
    # 调用 .init_array 中的全局构造函数
    ldr x0, =__init_array_start
    ldr x1, =__init_array_end
1:  cmp x0, x1
    b.ge 2f
    ldr x2, [x0], #8
    blr x2
    b   1b
2:
    bl  main                # 或 ejit_init + 用户入口
```

对应的链接脚本片段：

```ld
SECTIONS {
  .init_array : {
    PROVIDE_HIDDEN (__init_array_start = .);
    KEEP (*(SORT_BY_INIT_PRIORITY(.init_array.*)))
    KEEP (*(.init_array))
    PROVIDE_HIDDEN (__init_array_end = .);
  }
}
```

---

## 9. 风险与约束

| 风险 | 等级 | 缓解措施 |
|---|---|---|
| 自定义 PassBuilder 缺少 analysis 导致 pass 崩溃 | 中 | 从 PassBuilder 完整列表出发删减而非从零构建；充分单元测试 |
| freestanding mutex 与 OrcJIT 条件编译路径冲突 | 中 | 先在 hosted 环境用 no-op mutex 测试，确认 InPlaceTaskDispatcher 路径无死锁 |
| Support 桩函数遗漏导致链接失败 | 中 | 逐步添加桩函数，每个未定义符号对应一个桩实现 |
| 裸核内存管理器未正确设置内存权限 | 低（X86_64） / 中（AArch64） | X86_64 可用 RWX 区域；AArch64 需在链接脚本中设置正确页表权限 |
| 裁剪 AsmPrinter 时 DwarfDebug 调用点遗漏导致链接失败 | 中 | 梳理 AsmPrinter.cpp 中所有 DWARF/CodeView 引用，逐一 guard |
| 上游 LLVM 合并冲突 | 高 | 裁剪改动集中在 EJIT 目录 + CMake 选项，最小化侵入 |
| OrcTargetProcess 排除后 OrcJIT 编译/链接失败 | 低 | OrcTargetProcess 通过 CMake LINK_COMPONENTS 排除，不影响 OrcJIT Core |
| IPO/Scalar 传递依赖拉回已排除的库 | 低 | gc-sections 可移除 >95% 的传递依赖代码；长期拆分 IPO（§6.5） |
| 裸核工具链不提供 libstdc++ `<mutex>` | 中 | 备选方案 A（注入头文件），或使用 hosted 模式工具链 + libpthread 桩（§4.3.1） |
| JITLink ELF 裸核重定位/权限差异 | 低 | 简单函数通常不触发；集成测试阶段按需修复（§4.7） |
| 裸核 `.init_array` 未调用导致 LLVM 全局注册失败 | 中 | crt0.S 必须包含 `.init_array` 遍历（§8.4） |

---

## 10. 附录: 完整依赖图（裸核最小集）

```
EJIT (libLLVMEJIT.a)
├── Core → Remarks  ← 仅 libLLVMRemarks.a (1 KB, 可排除则排除)
├── Support → Demangle, BinaryFormat
├── BitReader → BitstreamReader
├── OrcJIT → OrcShared, ExecutionEngine
│   ├── JITLink → MC, Object, Support
│   └── Analysis (ItaniumManglingCanonicalizer via ProfileData)
├── Analysis
├── Scalar (SCCP, ADCE, SimplifyCFG, LoopUnroll)
├── InstCombine
├── IPO (AlwaysInliner only)
├── TransformUtils (Mem2Reg, LoopSimplify)
├── CodeGen → CodeGenTypes, Target
│   ├── SelectionDAG（默认 ISel，与 GlobalISel 二选一）
│   ├── AsmPrinter
│   └── TargetParser
├── MC → MCParser
├── Object
├── ProfileData (ItaniumManglingCanonicalizer only)
└── {Target}CodeGen + {Target}Desc + {Target}Info
    ├── X86: X86CodeGen, X86Desc, X86Info
    └── AArch64: AArch64CodeGen, AArch64Desc, AArch64Info, AArch64Utils

排除的传递依赖:
✗ Passes (→ Coroutines, ObjCARC, CFGuard, Vectorize, ...)
✗ OrcTargetProcess (dlsym/mmap)
✗ RuntimeDyld (JITLink 完全替代)
✗ BitWriter (仅 AOT)
✗ AsmParser, IRReader, IRPrinter, Linker
✗ DebugInfoDWARF, DebugInfoCodeView, DebugInfoDWARFLowLevel
✗ MCDisassembler, ObjectYAML, Option, CGData
✗ FrontendOpenMP, EmbeddedJIT, AggressiveInstCombine
✗ GlobalISel（可选，LLJIT 默认用 SelectionDAG）

注意：IPO 和 Scalar 的 LINK_COMPONENTS 可能传递拉入 Vectorize,
  Instrumentation, Linker, BitWriter, AggressiveInstCombine 的 .a 文件，
  但 gc-sections 后 .text 贡献 <50 KB（见 §6.5）。
```

---

*文档版本: 2.0*
*创建日期: 2026-05-24*
*更新日期: 2026-05-24*
