# EJIT Lipo 裁剪

将 LLVM 的 35-44 个 `.a` (120+ MB) 裁剪为单个 `ejit.o` (30-37 MB)，
通过 `ld -r` 部分链接合并所有依赖，用于测试和 bare-metal 运行时链接。

## 快速开始

### x86（日常测试）

```bash
# 0. 编译 runtime
ninja -C build_release_x86 LLVMEJIT

# 1-3. 三步裁剪
python3 ejit_test/lipo/lipo.py extract \
  --arch=x86 --build-dir=build_release_x86 \
  --output=ejit_test/lipo/tmp/extracted.a

python3 ejit_test/lipo/lipo.py gc-merge \
  --input=ejit_test/lipo/tmp/extracted.a \
  --build-dir=build_release_x86 \
  --output=ejit_test/lipo/tmp/gc_merged.a

python3 ejit_test/lipo/lipo.py merge \
  --input=ejit_test/lipo/tmp/gc_merged.a \
  --build-dir=build_release_x86 \
  --output=ejit_test/lipo/ejit.o

# 4. 验证
bash ejit_test/build.sh --run --lipo
```

产出 `ejit_test/lipo/ejit.o` (~37 MB)。x86 不需要 `--exclude` 参数
（lipo 从 test binary 的 linker map 自动计算依赖闭包）。

### aarch64（裸核交叉编译）

```bash
# 0. 配置并编译
cmake -S llvm -B build_release_aarch64 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
  -DCMAKE_C_FLAGS="-ffunction-sections -fdata-sections" \
  -DCMAKE_CXX_FLAGS="-ffunction-sections -fdata-sections" \
  -DLLVM_TARGETS_TO_BUILD="AArch64" \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_ENABLE_ZLIB=OFF \
  -DEJIT_BARE_METAL=ON

ninja -C build_release_aarch64 LLVMEJIT

# 1-3. 一键脚本（--exclude 排除未用 pass）
./ejit_test/lipo/run_aarch64_pipeline.sh
```

产出 `ejit_test/lipo/ejit_aarch64.o` (~30 MB)。

## Pipeline 三步详解

```
35-44 个 .a (120+ MB)
  │
  ├─ Step 1: extract
  │   编译 reference test binary → linker map → nm -u 依赖追踪
  │   迭代提取引用的 .o → 打包为单个 .a
  │   产出: 1 个 .a (80-100 MB, ~650-1065 .o)
  │
  ├─ Step 2: gc-merge
  │   ld -r --gc-sections: 以 .o 为单位链接后消除未引用的 section
  │   产出: 1 个 .a (46-59 MB)
  │
  └─ Step 3: merge
      ld -r -T merge.ld: 段排序 + 将 .a 合并为单个 .o
      产出: 1 个 ejit.o (30-37 MB)
```

| 架构 | extract 输入 | extract .o 数 | gc-merge 后 | merge 最终 |
|---|---|---|---|---|
| x86 | 35 个 .a | ~1065 | 59 MB | 37 MB |
| aarch64 | 44 个 .a | ~658 | 46 MB | 30 MB |

aarch64 更小因为 `EJIT_BARE_METAL` source guard 在编译时就排除了大量代码。

### 手动 aarch64 命令

`run_aarch64_pipeline.sh` 等价于：

```bash
python3 ejit_test/lipo/lipo.py extract \
  --arch=aarch64 --build-dir=build_release_aarch64 \
  --cxx=aarch64-linux-gnu-g++ --ld=build_release_x86/bin/ld.lld \
  --exclude=InstrProfCorrelator \
  --exclude=WinEHPrepare --exclude=WindowScheduler \
  ... (完整列表见 run_aarch64_pipeline.sh)

python3 ejit_test/lipo/lipo.py gc-merge \
  --input=ejit_test/lipo/libejit_lipo_aarch64.a \
  --build-dir=build_release_aarch64 --ld=build_release_x86/bin/ld.lld

python3 ejit_test/lipo/lipo.py merge \
  --input=ejit_test/lipo/libejit_lipo_aarch64_gc.a \
  --build-dir=build_release_aarch64 --ld=build_release_x86/bin/ld.lld \
  --output=ejit_test/lipo/ejit_aarch64.o
```

### 无 --exclude 对比版本

```bash
# aarch64 不传 --exclude 可得到完整版本（对比裁剪效果）
python3 ejit_test/lipo/lipo.py extract \
  --arch=aarch64 --build-dir=build_release_aarch64 \
  --cxx=aarch64-linux-gnu-g++ --ld=build_release_x86/bin/ld.lld
# → ~42 MB, 1053 .o
```

## 何时重建 ejit.o

**每次修改 EJIT runtime 源码后必须重建。** 否则 `build.sh --lipo`
链接的是旧代码，测试会因 ABI 不一致而失败（典型表现：`entries=0, misses>0`）。

需重建的情况：
- 修改 `llvm/lib/ExecutionEngine/EJIT/` 下任何文件
- 修改 `llvm/include/llvm/ExecutionEngine/EJIT/` 下任何头文件
- 修改 build 配置（cmake flags、`EJIT_BARE_METAL` 开关等）
- 只改设计文档 / 注释 / clang 代码 → 不需要重建

## 裁剪原理

两层机制叠加：

| 层 | 机制 | 说明 |
|---|---|---|
| 编译时 | `EJIT_BARE_METAL` source guard | `#ifndef` 排除 OS/arch 专属代码路径，aarch64 启用，x86 不启用 |
| 链接时 | lipo extract 依赖追踪 | linker map + `nm -u` 计算闭包，只提取引用的 .o |
| 链接时 | `--exclude` | 排除已知不需要的 pass/format .o（aarch64 专用） |
| 链接时 | `ld -r --gc-sections` | 以 section 粒度消除未引用代码 |

### 关键修复

`EJit.cpp` 中用 `InitializeAllTarget*()` 替代 `InitializeNative*()`。
交叉编译时 `LLVM_NATIVE_ARCH = X86` 但 target 是 AArch64，
`InitializeNativeTarget()` 展开为空导致后端未加载。

## 体积数据

### aarch64 (-Os, EJIT_BARE_METAL=ON)

| 阶段 | .o 数 | 大小 |
|---|---|---|
| extract | 658 | 80 MB |
| gc-merge | 1 (merged) | 46 MB |
| **ejit.o** | 1 | **30 MB** |
| `.text` | — | 10.1 MB |
| `.rodata` | — | 4.0 MB |
| 实际段总大小 | — | 14.7 MB |

对比无裁剪：1053 .o / 42 MB / `.text` 14.4 MB / 段 19.8 MB。

### x86 (Release, 无 EJIT_BARE_METAL)

| 阶段 | .o 数 | 大小 |
|---|---|---|
| extract | 1065 | 99 MB |
| gc-merge | 1 (merged) | 59 MB |
| **ejit.o** | 1 | **37 MB** |

## 裁剪明细

### 源码级 (EJIT_BARE_METAL guards, aarch64 only)

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

### lipo --exclude 排除的 pass (aarch64 only)

| 类别 | 数量 | 示例 |
|---|---|---|
| OS 专属 | ~15 | WinEHPrepare、CFGuard、RuntimeDyldCOFF/MachO |
| ScalarOpts 未用 | ~16 | LoopDataPrefetch、EarlyCSE、LICM、LoopStrengthReduce |
| TransformUtils 未用 | ~24 | Debugify、LowerAtomic、SampleProfileInference |
| CodeGen 未用 | ~80 | MachinePipeliner、RegAllocPBQP、ShadowStack、StackMaps |
| MIR/调试 | ~10 | MIRCanonicalizer、LiveDebugValues |
| GC/Instr/San | ~15 | GCMetadata、XRayInstrumentation、SanitizerBinaryMetadata |

详见 `run_aarch64_pipeline.sh` 中 `EXCLUDES` 数组。

## 注意事项

- x86 不能用全量 `EJIT_BARE_METAL` 编译 clang/lld（lld 需要 Wasm/COFF 格式后端），只能对 `LLVMEJIT` target 启用
- x86 `CodeGen.cpp` 仍有部分 pass 注册需补 guard
- `merge.ld` 控制最终 `.o` 的段布局（text/rodata/data 顺序），需要与链接脚本配合
- lipo 输出是 `ld -r` 部分链接的 `.o`（relocatable），不是最终可执行文件或 `.a`
