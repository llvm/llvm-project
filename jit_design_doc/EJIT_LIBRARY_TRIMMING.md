# EmbeddedJIT 运行时库裁剪设计文档

**版本**: 3.1
**日期**: 2026-05-28
**关联**: SPEC4.md, PASS7_EJitRuntime_OrcJITLink.md, EJIT_BARE_METAL_STUBS.md
**目标**: 提供 X86_64 / AArch64 裸核环境下 EJIT 运行时的最小单一产物

---

## 1. 裸核环境约束

| 约束 | 说明 |
|---|---|
| 无 OS | 无 mmap/munmap/mprotect，无 dlopen/dlsym，无 pthread |
| 无文件系统 | 无 fopen/read/write，bitcode 从内存加载 |
| 静态链接 | 所有符号编译期确定，无动态库 |
| 单线程 | 默认同步编译模式，无后台线程 |
| 内存受限 | RAM 100 KB – 2 MB，Flash 对应中等受限 |
| `ld -r` 部分链接 | 产物为单个 relocatable `.o`，合并后丢失 COMDAT `.group` 信息，导致 `objcopy`/`strip` 拒绝操作合并段中的 LOCAL 符号（如 ARM `$x`/`$d`）；合并后 `.shstrtab` 可被 merge.ld DISCARD 但符号级裁剪需在 merge 前完成或直接操作 ELF |

---

## 2. 裁剪历程与最终结果

### 2.1 总览

```
起点: 53 个 .a (117 MB)  →  44 个 .a (CMake 瘦身)  →  1 个 ejit.o (36-39 MB)
```

| 阶段 | 产物 (x86_64) | 产物 (aarch64) | 方法 |
|---|---|---|---|
| 原始 | 53 个 `.a`, 117 MB | 53 个 `.a`, ~150 MB | 通配符全量链接 |
| Phase 1 | 44 个 `.a` | 44 个 `.a` | EJitPassBuilder + IPO/Scalar LINK_COMPONENTS 裁剪 |
| Phase 2 | 1 个 `.a`, ~98 MB | 1 个 `.a`, ~108 MB | lipo extract（linker map + nm -u 依赖追踪） |
| Phase 3 | 1 个 `.a`, ~58 MB | 1 个 `.a`, ~59 MB | lipo gc-merge（ld -r --gc-sections） |
| Phase 4 | 1 个 `ejit.o`, **36 MB** | 1 个 `ejit.o`, **39 MB** | ld -r -T merge.ld（段合并 + .group DISCARD） |
| +源码裁剪 | — | 1 个 `ejit.o`, **39 MB** | `EJIT_BARE_METAL` guards 排除 CodeView/GlobalISel/DWARF 等 |
| 最终链接 | 二进制, **21 MB** | 二进制, 预估 **~12 MB** | `-Os` + `--gc-sections` + strip |

### 2.2 编译选项

| 选项 | 值 | 说明 |
|---|---|---|
| `CMAKE_BUILD_TYPE` | Release | |
| `CMAKE_CXX_FLAGS_RELEASE` | `-Os -DNDEBUG -ffunction-sections -fdata-sections` | 体积优先 + 段级裁剪 |
| `LLVM_TARGETS_TO_BUILD` | X86 或 AArch64 | 单架构 |
| `EJIT_BARE_METAL` | ON | 去除 mutex/chrono/logger/async |

---

## 3. EJIT 源文件裁剪

### 3.1 已删除

| 文件 | 原因 |
|---|---|
| `EJitJITLinkMemoryManager.cpp/.h` | 死代码桩，`allocate()` 调用 `report_fatal_error` |

### 3.2 宏隔离（`EJIT_BARE_METAL`）

| 文件 | 效果 |
|---|---|
| `EJitAsyncCompiler.cpp/.h` | 裸核排除了 std::thread/mutex/condition_variable |
| `EJitSyncCompiler.cpp/.h` | 只被 Async 调用，一并排除 |
| `EJitLogger.cpp` | no-op 桩 |
| `EJitCache.cpp` | std::shared_mutex → BareMetalMutex |
| `EJitRegistrationStore.cpp` | std::mutex → BareMetalMutex |
| `EJitRuntimeState.cpp` | std::mutex → BareMetalMutex |
| `EJitCompileDriver.cpp` | chrono 计时代码排除 |

### 3.3 LLVM 上游源文件裁剪（`EJIT_BARE_METAL` guards）

`EJIT_BARE_METAL` 宏被提升到顶层 `llvm/CMakeLists.txt` 作为全局 `add_definitions`，
允许在 LLVM 上游源文件中使用 `#ifndef EJIT_BARE_METAL` 排除不需要的功能。
具体 guards 分布在：

| 文件 | 排除内容 | 节省 (aarch64) |
|---|---|---|
| `AsmPrinter/AsmPrinter.cpp` | CodeView/DWARF debug 创建、Win/Wasm/AIX/ARM EH 创建 | ~3 MB |
| `AsmPrinter/CMakeLists.txt` | `CodeViewDebug.cpp`, `DwarfDebug.cpp`, `DwarfCFIException.cpp`, `WinException.cpp` 等 16 个源文件 | — |
| `CodeGen/CodeGen.cpp` | `initializeObjCARC*`, `initializeCFGuard*` pass 注册 | <1 MB |
| `CodeGen/TargetPassConfig.cpp` | `ObjCARCContract` pass | — |
| `AArch64/AArch64TargetMachine.cpp` | GlobalISel pipeline (IRTranslator, Legalizer, RegBankSelect, InstructionSelect), combiner passes, `EnableGlobalISelAtO` cl::opt | ~3 MB |
| `AArch64/AArch64Subtarget.cpp` | `CallLowering`, `LegalizerInfo`, `RegisterBankInfo` 构造和 getter | — |
| `AArch64/CMakeLists.txt` | `GISel/` 下 10 个源文件, `AArch64Arm64ECCallLowering.cpp`, `GlobalISel` LINK_COMPONENT | — |
| `X86/X86TargetMachine.cpp` | GlobalISel pipeline, AMX/EVEX/SLH/LVI/CFGuard/KCFI 等 pass | (x86 only) |

### 3.4 EJitPassBuilder

创建 `EJitPassBuilder` 替代 `PassBuilder`，只注册 EJIT 需要的 ~20 个 analysis（原 ~40 个），去除 Passes 组件依赖（19 个 LINK_COMPONENTS）。

---

## 4. Lipo 裁剪管线

### 4.1 三步流程

```
extract                 gc-merge                  merge
  │                        │                        │
  │ linker map +           │ ld -r --gc-sections    │ ld -r -T merge.ld
  │ nm -u 依赖追踪          │ 死代码段消除             │ 段合并 + .group DISCARD
  │                        │                        │
  ▼                        ▼                        ▼
libejit_lipo_{arch}.a   libejit_lipo_{arch}_gc.a  ejit.o (or ejit_aarch64.o)
```

|x86_64| aarch64 |
|---|---|
| extract: 98 MB (1062 .o) | extract: 108 MB (978 .o) |
| gc-merge: 58 MB | gc-merge: 59 MB |
| ejit.o: **36 MB** | ejit_aarch64.o: **39 MB** |

### 4.2 使用

```bash
# x86_64 (本机构建)
python3 ejit_test/lipo/lipo.py extract  --arch=x86 --build-dir=build_release_x86_os
python3 ejit_test/lipo/lipo.py gc-merge --input=ejit_test/lipo/libejit_lipo_x86.a --build-dir=build_release_x86_os
python3 ejit_test/lipo/lipo.py merge    --input=ejit_test/lipo/libejit_lipo_x86_gc.a --build-dir=build_release_x86_os

# aarch64 (交叉编译，需指定 --cxx 和 --ld)
python3 ejit_test/lipo/lipo.py extract  --arch=aarch64 --build-dir=build_release_aarch64 \
    --cxx=aarch64-linux-gnu-g++ --ld=build_release_x86/bin/ld.lld
python3 ejit_test/lipo/lipo.py gc-merge --input=ejit_test/lipo/libejit_lipo_aarch64.a \
    --build-dir=build_release_aarch64 --ld=build_release_x86/bin/ld.lld
python3 ejit_test/lipo/lipo.py merge    --input=ejit_test/lipo/libejit_lipo_aarch64_gc.a \
    --build-dir=build_release_aarch64 --ld=build_release_x86/bin/ld.lld \
    --output=ejit_test/lipo/ejit_aarch64.o

# 测试
./ejit_test/build.sh --run --lipo
```

### 4.3 extract 原理

1. **linker map 初提取**：用 `--print-map` 生成成功链接的完整 map，匹配 `libLLVM*.a(member.o)` 模式，提取被拉入的 `.o` 文件。使用 `--allow-multiple-definition` 处理 COMDAT 冲突。
2. **nm -u 依赖追踪**：对每个已提取的 `.o`，用 `nm -u` 列出未定义符号，在符号索引（`nm --print-armap` 建立）中查找定义者，迭代提取（通常 4-5 轮收敛）。
3. **同名冲突**：多个 `.a` 中同名 `.o`（如 `Local.cpp.o`），用 `archive__member.o` 唯一命名。
4. **交叉编译支持**：`--cxx` / `--ld` 参数可覆盖编译器/链接器路径。当使用 GCC 交叉编译器时，自动添加 `-B<ld_dir>` 和 `-L/tmp`（用于 dummy libz.a/libdl.a 等桩库）。
5. **输出清理**：构建 archive 前先删除旧文件，避免 `ar crs` 残留过期成员。

### 4.4 gc-merge 原理

`ld -r --gc-sections --entry=ejit_init -u <EJIT_API_symbols>` 对提取的 `.o` 做部分链接，以 `ejit_init` 为根裁剪未引用 function-sections。添加 `--allow-multiple-definition` 处理 COMDAT 重复符号。

### 4.5 merge 原理

`ld -r -T merge.ld --whole-archive` 用链接脚本将所有 per-function 段合并为单一 `.text`/`.rodata`/`.data` 段，DISCARD `.group`。

### 4.6 架构支持

`lipo.py` 通过 `--arch=x86|aarch64` 区分 Target 库：

| 架构 | Target 库 |
|---|---|
| x86 | `X86CodeGen`, `X86Desc`, `X86Info` |
| aarch64 | `AArch64CodeGen`, `AArch64Desc`, `AArch64Info`, `AArch64Utils` |

### 4.7 COMMON_LIBS 精简

以下库已从 `COMMON_LIBS` 移除（通过源码级 `EJIT_BARE_METAL` guards 确保无符号依赖）：

| 被移除的库 | 原因 |
|---|---|
| `libLLVMCFGuard.a` | Windows CFG，嵌入式不需要 |
| `libLLVMInstrumentation.a` | Sanitizer/PGO，嵌入式不需要 |
| `libLLVMDebugInfoCodeView.a` | Windows PDB 调试信息 |
| `libLLVMObjCARCOpts.a` | Objective-C ARC |
| `libLLVMDebugInfoDWARF.a` | DWARF debug info（AsmPrinter 已排除引用） |
| `libLLVMDebugInfoDWARFLowLevel.a` | DWARF low-level 工具 |

新增：

| 新增库 | 原因 |
|---|---|
| `libLLVMGlobalISel.a` | AArch64 后端依赖（代码路径通过 SelectionDAG，但部分符号跨路径引用） |

---

## 4.8 符号膨胀根因分析

ejit.o 最终产物 36-39 MB，实际包含 ~140K (aarch64) / ~78K (x86) 个符号。
为什么一个 JIT 库需要这么多符号？以下是根本原因。

### 4.8.1 C++ 模板实例化爆炸

LLVM 大量使用 C++ 模板类（`DenseMap<K,V>`、`SmallVector<T,N>`、`SmallPtrSet<T>` 等），
每个不同的模板参数组合生成一份独立的 WEAK 符号。例如：

```
DenseMap<Function*, int>       → 模板实例化 #1 (lookupBucketFor, initEmpty, destroyAll...)
DenseMap<BasicBlock*, int>     → 模板实例化 #2
DenseMap<MachineInstr*, MCSymbol*> → 模板实例化 #3
```

仅 `DenseMap` 一族就产生了 ~2,500 个 WEAK 符号（aarch64）。
加上 `SmallVector`、`SmallPtrSet`、`unique_ptr`、`std::function` 等，
WEAK 模板符号占符号表总量约 25%（aarch64 ~26,000 / x86 ~28,000）。

### 4.8.2 ARM 映射符号 `$x` / `$d`

ARM ELF ABI 要求汇编器在每个代码/数据边界插入 mapping symbols：
- `$x` — 标记 ARM 代码区（Thumb 用 `$t`）
- `$d` — 标记数据区（literal pool、跳转表等）

这些是 LOCAL NOTYPE 符号，作用是告诉反汇编器和链接器哪里是代码、哪里是数据。
它们不参与符号解析和重定位，但对 disassembler/linker relaxation 有用。

aarch64 ejit.o 中有 **~52,600 个 `$x`** 和 **~7,000 个 `$d`**，
占全部 LOCAL 符号的 60%（x86 没有这类符号，所以符号总量只有 aarch64 的一半）。
每个 `$x`/`$d` 占用 symtab 24 字节 + strtab ~3 字节 ≈ 27 字节。
约 60K 条 × 27 字节 ≈ **1.6 MB 纯符号表空间**，加上字符串表碎片化后约膨胀 **3-5 MB**。

### 4.8.3 静态库链接的传递膨胀

链接器处理 `.a` 时的基本单位是 `.o` 文件（不是符号）。当 `.o` 中的某个符号被引用，
整个 `.o` 的所有符号都被拉入，即使其中 90% 的代码并不需要。

LLVM 的 `-ffunction-sections -fdata-sections` 将每个函数/数据放入独立段，
使得 `--gc-sections` 可以裁剪未引用的段。但这只解决段级别的膨胀，
**符号级别的膨胀无法通过 linker 解决**——被拉入的 `.text` 段中即使只有一个函数被调用，
该段对应的所有 LOCAL 符号（匿名 namespace 函数、static 变量、字符串常量）都会被保留。

### 4.8.4 LLVM Pass 注册的全局初始化链

LLVM 的 Pass 系统使用全局构造函数注册所有可用 pass：

```cpp
// 在 PassRegistry.def 或各 Target 的初始化函数中
initializeAArch64DAGToDAGISelLegacyPass(Registry);   // → 拉入整个 SelectionDAG
initializeAArch64AsmPrinterPass(Registry);            // → 拉入 AsmPrinter → MC
initializeGlobalISel(Registry);                       // → 拉入整个 GlobalISel 框架
```

每个 `initialize*` 调用都是一个强符号引用，触发该 pass 所在 `.o` 的加载，
进而拉入该 `.o` 的所有依赖（包括该 pass 使用的所有 LLVM IR/CodeGen/MC 基础类）。

这就是为什么 `InitializeNativeTarget()` → `InitializeAllTargets()` 的修复
让 aarch64 ejit.o 从 18 MB 膨胀到 42 MB——它解锁了整套 AArch64 后端的加载。

### 4.8.5 各膨胀来源占比 (aarch64 ejit.o, 39 MB)

| 膨胀来源 | 符号数 | 贡献 | 可裁性 |
|---|---|---|---|
| C++ 模板 (DenseMap/SmallVector/unique_ptr...) | ~26,000 WEAK | ~4 MB | 低 — C++ 模板本质属性 |
| ARM `$x`/`$d` 映射符号 | ~60,000 LOCAL | ~3-5 MB | 中 — 可 strip，但 objcopy 受限 |
| 匿名 namespace 函数/对象 | ~5,000 LOCAL | ~2 MB | 低 — inline/static 函数 |
| GLOBAL LLVM API | ~16,000 | ~10 MB | — 核心功能，不可裁 |
| 字符串常量/重定位/其他 | — | ~16-18 MB | — 核心数据 |

### 4.8.6 与最终二进制的关系

ejit.o 作为中间产物，符号膨胀在最终链接阶段会被大幅削减：

```
ejit.o (39 MB / ~140K symbols)
  → 最终链接 --gc-sections: 裁剪未引用段 (→ ~25 MB)
  → --strip-all: 去除 LOCAL + WEAK 符号 (→ ~12 MB)
```

`--strip-all` 后 LOCAL 和 WEAK 符号全部丢弃（包括 `$x`/`$d` 映射符号），
仅保留 GLOBAL 符号供动态解析使用。因此符号膨胀主要影响**中间产物大小**
和**链接速度**，对最终二进制大小影响有限。

### 4.8.7 InitializeAll* 的依赖链（交叉编译的核心问题）

为什么 `EJit.cpp` 里改 2 行初始化函数会导致 ejit.o 从 18 MB 膨胀到 42 MB？
根本原因是交叉编译：

**修复前（`InitializeNativeTarget` + `InitializeNativeTargetAsmPrinter`）：**
```
build_release_aarch64/include/llvm/Config/llvm-config.h:
  #define LLVM_NATIVE_ARCH X86           ← CMake 检测 host 是 x86
  #undef LLVM_NATIVE_TARGET              ← 但 LLVM_TARGETS_TO_BUILD=AArch64
  #undef LLVM_NATIVE_ASMPRINTER          ← 没有编译 X86 target
```
两个函数展开后都是**空操作**——没有任何符号引用产生。AArch64 后端完全未被拉入。

**修复后（`InitializeAll*`）：**
```
InitializeAllTargetInfos()  → LLVMInitializeAArch64TargetInfo()   → AArch64Info
InitializeAllTargets()      → LLVMInitializeAArch64Target()       → AArch64TargetMachine
                              └→ PassRegistry 注册所有 pass:
                                 ├ initializeAArch64DAGToDAGISel() → SelectionDAG (26 .o)
                                 ├ initializeAArch64AsmPrinter()   → AsmPrinter (24 .o)
                                 ├ initializeGlobalISel()          → GlobalISel (23 .o)
                                 └ AArch64CodeGen 全部 (59 .o)     → CodeGen (221 .o)
InitializeAllTargetMCs()     → LLVMInitializeAArch64TargetMC()    → AArch64Desc (12 .o)
                                                                    → MC (61 .o)
InitializeAllAsmPrinters()   → LLVMInitializeAArch64AsmPrinter()  → AsmPrinter passes
```

真实新增的 `.o` 数量：

| 阶段 | .o 总数 | 新增库 |
|---|---|---|
| 修复前 (Native=noop) | 585 | 无后端代码 |
| 修复后 (InitializeAll) | 1053 | +CodeGen 221, +AArch64CodeGen 59, +MC 61, +SelectionDAG 26, +AsmPrinter 24, +GlobalISel 23, +AArch64Desc 12 = **+468 .o** |

这 468 个 `.o` 是 JIT 编译的**核心功能**——没有它们 TargetMachine 无法创建，JIT 完全不能工作。
修复不是"引入膨胀"，而是"补上了之前缺失的核心功能"。
膨胀主要来自 LLVM 的 monorepo 架构（详见 4.8.1-4.8.5），我们通过 3.3 节的源码裁剪将不需要的
部分（CodeView/GlobalISel（部分）/DWARF debug/ObjCARC/CFGuard）去掉了，最终收敛到 978 .o（39 MB）。

---

## 5. 体积数据

### 5.1 X86_64 (`-Os`, build_release_x86_os)

| 产物 | 大小 |
|---|---|
| extract `.a` | 98 MB (1062 .o) |
| gc-merge `.a` | 58 MB |
| **ejit.o** | **36 MB** |
| `.text` | 14.5 MB |
| `.rodata` | 4.8 MB |
| 最终二进制 | **21 MB** |
| 最终 `.text` | **15.2 MB** |

### 5.2 AArch64 (`-Os`, build_release_aarch64, 交叉编译)

| 产物 | 源码裁剪前 | 源码裁剪后 |
|---|---|---|
| extract `.a` | 115 MB (1053 .o) | **108 MB (978 .o)** |
| gc-merge `.a` | 65 MB | **59 MB** |
| **ejit.o** | **42 MB** | **39 MB** |
| `.text` | 14.4 MB | **13.1 MB** |
| `.rodata` | 4.6 MB | **4.1 MB** |
| `.data` + `.bss` | 779 KB | **694 KB** |
| 实际段总大小 | 19.8 MB | **17.9 MB** |

> **注意**: aarch64 ejit.o 文件比 x86 大 3 MB（39 vs 36 MB），但实际代码段更小（13.1 vs 14.5 MB）。文件膨胀来自 ARM `$x`/`$d` 映射符号(~60K 条)撑大的 `.symtab` 和 `.strtab`，最终 `--strip-all` 后会消除。

### 5.3 源码裁剪明细 (AArch64)

| 裁剪项 | 减少 .o 数 | 节省大小 |
|---|---|---|
| CodeView (DebugInfo library + AsmPrinter handler) | -21 | ~0.3 MB |
| GlobalISel (AArch64 GISel + library) | -28 | ~3 MB |
| ObjCARC + CFGuard + Win/Wasm/AIX EH | -8 | ~0.2 MB |
| DWARF debug info (DwarfDebug + DwarfCompileUnit + DIE 等) | -18 | ~2 MB |
| **合计** | **-75** | **~5.5 MB** |

### 5.4 最终二进制瘦身路径

```
ejit.o (36-39 MB)
  → 最终链接 --gc-sections: 裁剪未引用段
  → --strip-all: 去除符号表
  → 21 MB (x86) / 预估 12 MB (aarch64) 二进制
```

---

## 6. 裁剪效果对比

| 指标 | 原始 | 当前 (x86) | 当前 (aarch64) | 缩减 |
|---|---|---|---|---|
| `.a`/`.o` 文件数 | 53 | **1** | **1** | -98% |
| 部署产物大小 | 117 MB (53 .a) | **36 MB** (1 .o) | **39 MB** (1 .o) | -67~69% |
| `.text` | — | **14.5 MB** | **13.1 MB** | — |
| 最终二进制 | ~30 MB | **21 MB** | 预估 **~12 MB** | -30~60% |
| EJIT 自身大小 | 100 KB | 100 KB | 100 KB | — |

---

## 7. 已解决 / 遗留的裁剪方向

### 7.1 已解决 (v3.1)

| 项目 | 方法 | 节省 (aarch64) |
|---|---|---|
| CodeView (DebugInfo) | `#ifndef EJIT_BARE_METAL` guard AsmPrinter + CMakeLists 排除源文件 | ~0.3 MB |
| GlobalISel | AArch64TargetMachine/AArch64Subtarget/CMakeLists 用 `EJIT_BARE_METAL` 排除 | ~3 MB |
| DWARF debug info | AsmPrinter CMakeLists 排除 DwarfDebug/DwarfCompileUnit/DIE 等 | ~2 MB |
| ObjCARC | CodeGen.cpp + TargetPassConfig.cpp pass 注册 guard | <0.1 MB |
| CFGuard / WinEH / WasmEH / AIXEH | AsmPrinter.cpp 条件排除 | <0.1 MB |
| 交叉编译支持 | EJit.cpp 用 `InitializeAllTarget*()` 代替 `InitializeNative*()` | 使 aarch64 交叉构建可用 |

### 7.2 遗留

| 项目 | 说明 |
|---|---|
| `$x`/`$d` 映射符号 | aarch64 ejit.o 有 ~60K 条 ARM mapping symbols，膨胀 `.symtab`/`.strtab` 约 7 MB。objcopy/strip 无法直接移除（section 合并后无 group info）。`strip_mapping_symbols.py` (WIP) 可直接操作 ELF symtab。 |
| RuntimeDyld | ~0.8 MB，OrcJIT Layer.cpp 核心引用 |
| OrcTargetProcess | ~0.01 MB，LLJIT SelfExecutorProcessControl 依赖 |
| ThinLTO | 预计 -10~30%，未测试 |

### 7.3 编译选项

| 选项 | 预计收益 | 状态 |
|---|---|---|
| `-Os` | -28% 二进制 | ✅ 已落地（x86 + aarch64） |
| ThinLTO | -10~30% | 未测试 |

### 7.4 裸核专用 ExecutorProcessControl

自定义 EPC + JITLinkMemoryManager 可解锁 OrcTargetProcess/RuntimeDyld 裁剪，是裸核部署的必经之路。

---

## 8. 构建命令参考

```bash
# === x86_64 本机构建 ===
cmake -S llvm -B build_release_x86_os -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-ffunction-sections -fdata-sections" \
  -DCMAKE_C_FLAGS="-ffunction-sections -fdata-sections" \
  -DCMAKE_CXX_FLAGS_RELEASE="-Os -DNDEBUG" \
  -DCMAKE_C_FLAGS_RELEASE="-Os -DNDEBUG" \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DBUILD_SHARED_LIBS=OFF -DLLVM_OPTIMIZED_TABLEGEN=ON \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_ENABLE_ZLIB=OFF -DLLVM_ENABLE_ZSTD=OFF \
  -DEJIT_BARE_METAL=ON

ninja -C build_release_x86_os clang LLVMEJIT lld

# === aarch64 交叉编译 ===
cmake -S llvm -B build_release_aarch64 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
  -DCMAKE_C_FLAGS="-ffunction-sections -fdata-sections" \
  -DCMAKE_CXX_FLAGS="-ffunction-sections -fdata-sections" \
  -DCMAKE_C_FLAGS_RELEASE="-Os -DNDEBUG" \
  -DCMAKE_CXX_FLAGS_RELEASE="-Os -DNDEBUG" \
  -DLLVM_TARGETS_TO_BUILD="AArch64" \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_ENABLE_ZLIB=OFF \
  -DEJIT_BARE_METAL=ON

ninja -C build_release_aarch64 LLVMEJIT

# 生成 ejit.o (aarch64 需指定交叉编译器)
python3 ejit_test/lipo/lipo.py extract  --arch=aarch64 --build-dir=build_release_aarch64 \
    --cxx=aarch64-linux-gnu-g++ --ld=build_release_x86/bin/ld.lld
python3 ejit_test/lipo/lipo.py gc-merge --input=ejit_test/lipo/libejit_lipo_aarch64.a \
    --build-dir=build_release_aarch64 --ld=build_release_x86/bin/ld.lld
python3 ejit_test/lipo/lipo.py merge    --input=ejit_test/lipo/libejit_lipo_aarch64_gc.a \
    --build-dir=build_release_aarch64 --ld=build_release_x86/bin/ld.lld \
    --output=ejit_test/lipo/ejit_aarch64.o

# 测试
./ejit_test/build.sh --run --lipo
```

---

*文档版本: 3.1*
*创建日期: 2026-05-24*
*更新日期: 2026-05-28*
