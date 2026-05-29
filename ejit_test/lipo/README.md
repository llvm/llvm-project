# EJIT Lipo 裁剪

将 LLVM 的 44 个 `.a` (130+ MB) 裁剪为单个 `ejit.o` (30 MB)，
用于 bare-metal EJIT 运行时链接。

## 快速开始

```bash
# 1. 构建 aarch64 EJIT（交叉编译）
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

# 2. 生成 ejit_aarch64.o
./ejit_test/lipo/run_aarch64_pipeline.sh

# 输出: ejit_test/lipo/ejit_aarch64.o (~30 MB)
```

## 裁剪原理

### 两层机制

| 层 | 机制 | 说明 |
|---|---|---|
| 编译时 | `EJIT_BARE_METAL` source guard | `#ifndef` 排除 OS/arch 专属代码路径 |
| 链接时 | lipo extract `--exclude` | 排除未使用的 pass/format `.o` 文件 |

### Pipeline 三步

```
44 个 .a (130 MB)
  → extract: linker map + nm -u 依赖追踪 → 1 个 .a (80 MB, ~658 .o)
  → gc-merge: ld -r --gc-sections → 1 个 .a (46 MB)
  → merge: ld -r -T merge.ld → 1 个 ejit.o (30 MB)
```

### 关键修复

`EJit.cpp` 中用 `InitializeAllTarget*()` 替代 `InitializeNative*()`。
交叉编译时 `LLVM_NATIVE_ARCH = X86` 但 target 是 AArch64，
`InitializeNativeTarget()` 展开为空，导致 AArch64 后端完全未加载。

## 体积数据 (aarch64, -Os)

| 阶段 | .o 数 | 大小 |
|---|---|---|
| extract | 658 | 80 MB |
| gc-merge | 1 (merged) | 46 MB |
| **ejit.o** | 1 | **30 MB** |
| `.text` | — | 10.1 MB |
| `.rodata` | — | 4.0 MB |
| 实际段总大小 | — | 14.7 MB |

对比原始：1053 .o / 42 MB / .text 14.4 MB / 实际段 19.8 MB。

## 裁剪明细

### 源码级 (EJIT_BARE_METAL guards)

| 文件 | 排除内容 |
|---|---|
| `AsmPrinter/AsmPrinter.cpp` | CodeView/DWARF debug、Win/Wasm/AIX/ARM EH |
| `AsmPrinter/CMakeLists.txt` | DwarfDebug、DwarfCFIException、WinException 等 16 文件 |
| `CodeGen/CodeGen.cpp` | ObjCARC、CFGuard、MachineOutliner、WasmEH/WinEH pass 注册 |
| `CodeGen/TargetPassConfig.cpp` | ObjCARC、GlobalMergeFunc、MachineOutliner 管线 |
| `AArch64/AArch64TargetMachine.cpp` | GlobalISel pipeline、combiner passes |
| `AArch64/AArch64Subtarget.cpp` | CallLowering、LegalizerInfo、RegisterBankInfo |
| `AArch64/MCTargetDesc/AArch64AsmBackend.cpp` | Darwin/WinCOFF backend |
| `MC/TargetRegistry.cpp` | 非 ELF streamer 工厂分支 |
| `Object/Binary.cpp` | 非 ELF 格式自动检测 |
| `Object/ObjectFile.cpp` | 非 ELF ObjectFile 创建 |
| `IR/Mangler.cpp` | Arm64EC mangling |
| `Demangle/Demangle.cpp` | MS/Rust/DLang demangle |
| `Orc/LLJIT.cpp` | COFF/MachO platform setup |
| `MC/CMakeLists.txt` | AArch64: 排除 MachO/COFF+bloat; x86: 仅排除 bloat |
| `JITLink/CMakeLists.txt` | AArch64: ELF only; x86: ELF+MachO+COFF |

### lipo --exclude 排除的 pass

| 类别 | 数量 | 示例 |
|---|---|---|
| OS 专属 | ~15 | WinEHPrepare、CFGuard、RuntimeDyldCOFF/MachO |
| ScalarOpts 未用 | ~16 | LoopDataPrefetch、EarlyCSE、LICM、LoopStrengthReduce |
| TransformUtils 未用 | ~24 | Debugify、LowerAtomic、SampleProfileInference |
| CodeGen 未用 | ~80 | MachinePipeliner、RegAllocPBQP、ShadowStack、StackMaps |
| MIR/调试 | ~10 | MIRCanonicalizer、LiveDebugValues |
| GC/Instr/San | ~15 | GCMetadata、XRayInstrumentation、SanitizerBinaryMetadata |

详见 `run_aarch64_pipeline.sh` 中 `EXCLUDES` 数组。

## x86 构建注意

x86 不能用全量 `EJIT_BARE_METAL` 编译 clang/lld（lld 需要 Wasm/COFF），
需分两步：

```bash
# 1. 编译 clang/lld（不带 EJIT_BARE_METAL）
cmake ... -DEJIT_BARE_METAL=OFF
ninja -C build_release_x86_os clang lld

# 2. 重新配置并仅编译 LLVMEJIT
cmake ... -DEJIT_BARE_METAL=ON
ninja -C build_release_x86_os LLVMEJIT
```

x86 仍有部分 `CodeGen.cpp` 的 pass 注册需补 guard 才能通过最终测试。
