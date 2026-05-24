# EmbeddedJIT 运行时库裁剪设计文档

**版本**: 1.0
**日期**: 2026-05-24
**关联**: SPEC4.md, PASS7_EJitRuntime_OrcJITLink.md
**目标**: 减少嵌入式场景下 EJIT 运行时库的二进制体积

---

## 1. 现状分析

### 1.1 当前链接概览

EJIT 测试二进制（`ejit_attr_test`）当前链接 **53 个** LLVM `.a` 文件，最终二进制体积 **~30 MB**（gc-sections + strip 后）。

各库对最终 `.text` 段的实际贡献（按大小降序）：

| 库 | .a 大小 | 实际链接 | 说明 |
|---|---|---|---|
| libLLVMCodeGen.a | 15,963 KB | **4,866 KB** | 目标无关后端（寄存器分配、调度等） |
| libLLVMX86CodeGen.a | 10,900 KB | **4,600 KB** | X86 目标代码生成（67/67 .o 全部链接） |
| libLLVMSelectionDAG.a | 7,357 KB | **3,066 KB** | DAG 指令选择 |
| libLLVMAnalysis.a | 10,645 KB | **2,611 KB** | 分析 pass（103/125 .o 链接） |
| libLLVMX86Desc.a | 4,030 KB | **2,535 KB** | X86 MC 描述层 |
| libLLVMCore.a | 8,151 KB | **2,454 KB** | IR 核心 |
| libLLVMTransformUtils.a | 5,387 KB | **1,094 KB** | 变换工具 |
| libLLVMInstCombine.a | 3,489 KB | **1,089 KB** | InstCombine pass |
| libLLVMSupport.a | 4,569 KB | **690 KB** | 基础支持库 |
| libLLVMAsmPrinter.a | 1,983 KB | **661 KB** | 汇编输出（含 DWARF） |
| libLLVMGlobalISel.a | 2,733 KB | **641 KB** | GlobalISel 指令选择 |
| libLLVMMC.a | 2,267 KB | **603 KB** | MC 层 |
| libLLVMJITLink.a | 2,598 KB | **525 KB** | JIT 链接器 |
| libLLVMScalarOpts.a | 6,971 KB | **479 KB** | 标量优化 |
| libLLVMObject.a | 3,180 KB | **375 KB** | 目标文件读写 |
| libLLVMOrcJIT.a | 5,105 KB | **346 KB** | OrcJIT 核心 |
| libLLVMBitReader.a | 899 KB | **271 KB** | Bitcode 读取 |
| libLLVMProfileData.a | 2,677 KB | **257 KB** | Profile 数据 |
| libLLVMBitWriter.a | 562 KB | **244 KB** | Bitcode 写入 |
| libLLVMPasses.a | 6,582 KB | **184 KB** | PassBuilder 注册 |
| libLLVMInstrumentation.a | 3,342 KB | **149 KB** | 插桩 pass |
| libLLVMipo.a | 8,020 KB | **148 KB** | IPO pass |
| libLLVMRuntimeDyld.a | 980 KB | **132 KB** | 运行时动态链接器 |
| libLLVMDebugInfoCodeView.a | 1,644 KB | **119 KB** | CodeView 调试信息 |
| libLLVMEJIT.a | 383 KB | **100 KB** | **EJIT 自身** |
| 其余 28 个库 | — | **< 500 KB** | 贡献较小 |

**关键发现**: EJIT 自身仅占 100 KB，而 LLVM 基础设施占 28,874 KB（99.7%）。

### 1.2 依赖链分析

EJIT 的直接 LLVM 依赖为：

```
EJIT → Core, Support, BitReader, OrcJIT, JITLink, ExecutionEngine,
       Passes, Scalar, InstCombine, IPO, TransformUtils, Analysis,
       {Target}CodeGen/Desc/Info
```

但 **Passes** 是最大的传递依赖放大器——它在 CMake 中声明链接 19 个组件，因为 `PassBuilder.cpp` 必须包含所有变换 pass 的头文件：

```
Passes → AggressiveInstCombine, CFGuard, CodeGen, Coroutines,
         GlobalISel, Instrumentation, IRPrinter, ObjCARC, Vectorize,
         EmbeddedJIT (→ IRReader → AsmParser), ...
```

**X86CodeGen** 是第二大放大器——完整代码生成管线（SelectionDAG + AsmPrinter + GlobalISel + DebugInfo）：

```
X86CodeGen → SelectionDAG, AsmPrinter (→ DebugInfoDWARF, DebugInfoCodeView),
             GlobalISel, Instrumentation, ProfileData, ...
```

### 1.3 哪些库是"不需要"的

以下库在 EJIT 场景中完全不需要，但因传递依赖被拉入：

| 库 | 实际链接大小 | 为什么被拉入 | 为什么不需要 |
|---|---|---|---|
| libLLVMCoroutines.a | 0.3 KB | Passes 直接链接 | EJIT 不使用协程 |
| libLLVMObjCARCOpts.a | 27.8 KB | Passes 直接链接 | EJIT 不使用 ObjC ARC |
| libLLVMCFGuard.a | 5.3 KB | Passes, X86CodeGen | EJIT 不使用 Windows CFG |
| libLLVMFrontendOpenMP.a | 2.8 KB | X86CodeGen | EJIT 不使用 OpenMP |
| libLLVMIRPrinter.a | 0.7 KB | Passes, X86CodeGen | 仅调试用 IR 打印 |
| libLLVMObjectYAML.a | 0.8 KB | 符号级依赖 | YAML 序列化 EJIT 不需要 |
| libLLVMLinker.a | 0.8 KB | IPO 传递 | EJIT JIT 管线不做模块链接 |
| libLLVMAsmParser.a | 3.4 KB | IRReader 传递 | EJIT 从 bitcode 加载，不解析汇编 |
| libLLVMMCDisassembler.a | 0.3 KB | X86Desc 传递 | 嵌入式场景不需要反汇编 |

这些库本身不大，但它们暴露出依赖链中的**架构问题**：`PassBuilder` 和 `X86CodeGen` 过度耦合，拉入了大量不需要的传递依赖。

### 1.4 真正的体积大户

真正占空间的是 **CodeGen 管线**（占最终二进制的 ~55%）：

| 组件 | 链接大小 | 占比 | 必要性 |
|---|---|---|---|
| X86CodeGen + SelectionDAG + GlobalISel | ~8,307 KB | 28.7% | **JIT 必需**（代码生成） |
| CodeGen（目标无关） | ~4,866 KB | 16.8% | **JIT 必需**（后端基础设施） |
| AsmPrinter + DebugInfo | ~785 KB | 2.7% | **JIT 必需**（代码发射），DebugInfo 可裁剪 |
| X86Desc + MC + MCParser | ~3,244 KB | 11.2% | **JIT 必需**（MC 层） |
| Analysis | ~2,611 KB | 9.0% | **优化必需**（分析 pass） |
| Passes + 变换 pass | ~2,088 KB | 7.2% | **部分必需**（只用 7 个 pass） |

---

## 2. 裁剪策略

### 2.1 策略总览

```
┌───────────────────────────────────────────────────────────────┐
│                    裁剪策略层次                                │
│                                                               │
│  Layer 1: 定制 PassBuilder（-3,500 KB）                       │
│  ├─ 自建 EJitPassBuilder 替代 PassBuilder                    │
│  └─ 只注册 EJIT 使用的 7 个 pass + 4 个 analysis             │
│                                                               │
│  Layer 2: 裁剪 CodeGen 依赖（-2,000 KB）                      │
│  ├─ 仅保留 Target 的 JIT 编译路径                             │
│  ├─ 移除 SelectionDAG 中非 JIT 路径                          │
│  └─ 移除 AsmPrinter 的 DWARF 发射                            │
│                                                               │
│  Layer 3: 编译选项优化（-1,500 KB）                            │
│  ├─ -DLLVM_ENABLE_ZLIB=OFF / -DLLVM_ENABLE_ZSTD=OFF         │
│  ├─ -DLLVM_ENABLE_TERMINFO=OFF                               │
│  ├─ 仅构建目标架构 Target                                     │
│  └─ -DLLVM_TARGETS_TO_BUILD="X86" 或 "AArch64;ARM"          │
│                                                               │
│  Layer 4: 深度裁剪（长期，-3,000 KB）                         │
│  ├─ 自建迷你 Analysis 注册                                    │
│  ├─ 裁剪 Support 库                                           │
│  └─ 代码生成替代方案探索                                      │
│                                                               │
│  预期总计: 30 MB → ~20 MB (L1+L2+L3) → ~17 MB (含 L4)       │
└───────────────────────────────────────────────────────────────┘
```

---

## 3. Layer 1: 定制 PassBuilder

### 3.1 问题

`llvm::PassBuilder`（6,582 KB .a）的 `PassBuilder.cpp` 包含了 **所有** LLVM 变换 pass 的头文件（60+ `#include`），其 `LINK_COMPONENTS` 声明了 19 个组件。EJIT 实际只使用 7 个 pass：

| EJIT 使用的 Pass | 头文件 | 所在库 |
|---|---|---|
| `InstCombinePass` | `Transforms/InstCombine/InstCombine.h` | InstCombine |
| `SCCPPass` | `Transforms/Scalar/SCCP.h` | Scalar |
| `ADCEPass` | `Transforms/Scalar/ADCE.h` | Scalar |
| `SimplifyCFGPass` | `Transforms/Scalar/SimplifyCFG.h` | Scalar |
| `LoopFullUnrollPass` | `Transforms/Scalar/LoopUnrollPass.h` | Scalar |
| `AlwaysInlinerPass` | `Transforms/IPO/AlwaysInliner.h` | IPO |
| `PromotePass` (Mem2Reg) | `Transforms/Utils/Mem2Reg.h` | TransformUtils |

加上 `LoopSimplifyPass`，共 **8 个 pass**。但 PassBuilder 注册了 **100+ 个 pass**，拉入了 Coroutines、CFGuard、ObjCARC、Vectorize、Instrumentation、AggressiveInstCombine 等完全不需要的库。

### 3.2 方案: EJitPassBuilder

创建 `EJitPassBuilder`，手动注册 EJIT 所需的 analysis 和 pass，替代 `PassBuilder`：

```cpp
// llvm/include/llvm/ExecutionEngine/EJIT/EJitPassBuilder.h

namespace llvm::ejit {

class EJitPassBuilder {
public:
  EJitPassBuilder();

  // 仅注册 EJIT 需要的 analysis
  void registerAnalyses();

  // 运行单个 pass，不依赖 PassBuilder 的全局注册
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

```cpp
// llvm/lib/ExecutionEngine/EJIT/EJitPassBuilder.cpp

EJitPassBuilder::EJitPassBuilder() {
  registerAnalyses();
}

void EJitPassBuilder::registerAnalyses() {
  // 手动注册 EJIT 需要的 analysis，替代 PassBuilder 的全量注册
  //
  // PassBuilder::registerFunctionAnalyses 注册了 ~40 个 analysis，
  // EJIT 只需要以下核心 analysis:

  // --- Function Analysis ---
  FAM_.registerPass([&] { return DominatorTreeAnalysis(); });
  FAM_.registerPass([&] { return AssumptionAnalysis(); });
  FAM_.registerPass([&] { return TargetIRAnalysis(); });
  FAM_.registerPass([&] { return TargetLibraryAnalysis(); });
  FAM_.registerPass([&] { return AAManager(); });            // AliasAnalysis
  FAM_.registerPass([&] { return BasicAA(); });
  FAM_.registerPass([&] { return ScalarEvolutionAnalysis(); });  // LoopUnroll 需要
  FAM_.registerPass([&] { return LoopAnalysis(); });
  FAM_.registerPass([&] { return MemorySSAAnalysis(); });       // ADCE 需要
  FAM_.registerPass([&] { return PhiValuesAnalysis(); });
  FAM_.registerPass([&] { return OptimizationRemarkEmitterAnalysis(); });

  // --- Loop Analysis ---
  LAM_.registerPass([&] { return DominatorTreeAnalysis(); });   // 依赖
  LAM_.registerPass([&] { return LoopAnalysis(); });
  LAM_.registerPass([&] { return ScalarEvolutionAnalysis(); });
  LAM_.registerPass([&] { return SimplifyQueryAnalysis() });

  // --- Module Analysis ---
  MAM_.registerPass([&] { return PassInstrumentationAnalysis(); });
  MAM_.registerPass([&] { return InlineAdvisorAnalysis() });    // AlwaysInliner 需要

  // 交叉注册
  FAM_.registerPass([&] { return LoopAnalysisManagerFunctionProxy(LAM_); });
  MAM_.registerPass([&] { return FunctionAnalysisManagerModuleProxy(FAM_); });
  FAM_.registerPass([&] { return ModuleAnalysisManagerFunctionProxy(MAM_) });
}

void EJitPassBuilder::runInstCombine(Module &M) {
  FunctionPassManager FPM;
  FPM.addPass(InstCombinePass());
  FPM.addPass(PromotePass());
  FPM.addPass(InstCombinePass());
  for (Function &F : M.functions())
    if (!F.isDeclaration())
      FPM.run(F, FAM_);
}

void EJitPassBuilder::runOptimizationPipeline(Module &M,
                                               OptimizationLevel level) {
  // L1: SCCP + ADCE + SimplifyCFG
  {
    FunctionPassManager FPM;
    FPM.addPass(SCCPPass());
    FPM.addPass(ADCEPass());
    FPM.addPass(SimplifyCFGPass());
    for (Function &F : M.functions())
      if (!F.isDeclaration())
        FPM.run(F, FAM_);
  }

  // L2: AlwaysInliner + SimplifyCFG
  if (static_cast<int>(level) >= 2) {
    ModulePassManager MPM;
    MPM.addPass(AlwaysInlinerPass());
    MPM.run(M, MAM_);
    for (Function &F : M.functions()) {
      if (!F.isDeclaration()) {
        FunctionPassManager FPM2;
        FPM2.addPass(SimplifyCFGPass());
        FPM2.run(F, FAM_);
      }
    }
  }

  // L3: LoopUnroll
  if (static_cast<int>(level) >= 3) {
    FunctionPassManager FPM3;
    FPM3.addPass(LoopSimplifyPass());
    {
      LoopPassManager LPM;
      LPM.addPass(LoopFullUnrollPass());
      FPM3.addPass(createFunctionToLoopPassAdaptor(std::move(LPM)));
    }
    FPM3.addPass(PromotePass());
    FPM3.addPass(SimplifyCFGPass());
    for (Function &F : M.functions())
      if (!F.isDeclaration())
        FPM3.run(F, FAM_);
  }
}
```

### 3.3 修改 EJitOptimizer

```cpp
// 修改前 (EJitOptimizer.cpp):
EJitOptimizer::EJitOptimizer(PeriodArrayRegistry &reg)
    : registry_(reg) {
  PassBuilder PB;                                    // ← 拉入 19 个组件
  PB.registerFunctionAnalyses(FAM_);
  PB.registerLoopAnalyses(LAM_);
  PB.registerCGSCCAnalyses(CGAM_);
  PB.registerModuleAnalyses(MAM_);
  PB.crossRegisterProxies(LAM_, FAM_, CGAM_, MAM_);
}

// 修改后:
EJitOptimizer::EJitOptimizer(PeriodArrayRegistry &reg)
    : registry_(reg), pb_(std::make_unique<EJitPassBuilder>()) {
}
```

### 3.4 EJitOrcEngine 中的 PassBuilder 使用

`EJitOrcEngine.cpp` 不直接使用 PassBuilder——优化管线由 `EJitOptimizer` 驱动。无需修改。

### 3.5 CMake 修改

```cmake
# llvm/lib/ExecutionEngine/EJIT/CMakeLists.txt
# 修改前:
LINK_COMPONENTS
  ...
  Passes          # ← 拉入 19 个传递依赖
  ...

# 修改后: 移除 Passes，改为直接依赖 EJIT 使用的组件
LINK_COMPONENTS
  Core
  Support
  BitReader
  OrcJIT
  JITLink
  ExecutionEngine
  Scalar          # SCCP, ADCE, SimplifyCFG, LoopUnroll
  InstCombine
  IPO             # AlwaysInliner
  TransformUtils  # Mem2Reg, LoopSimplify
  Analysis        # DominatorTree, SCEV, AA 等
  CodeGen         # TargetIRAnalysis (LLJIT 内部需要)
  ${LLVM_TARGETS_TO_BUILD}  # X86/AArch64/ARM
```

### 3.6 Layer 1 预期收益

移除 Passes 依赖链后，以下库可完全消除：

| 可移除的库 | .a 总大小 | 实际链接大小 |
|---|---|---|
| libLLVMCoroutines.a | 585 KB | 0.3 KB |
| libLLVMObjCARCOpts.a | 243 KB | 27.8 KB |
| libLLVMCFGuard.a | 31 KB | 5.3 KB |
| libLLVMAggressiveInstCombine.a | 141 KB | 1.4 KB |
| libLLVMVectorize.a | 5,920 KB | 48.4 KB |
| libLLVMIRPrinter.a | 9.3 KB | 0.7 KB |
| libLLVMInstrumentation.a (大部分) | 3,342 KB | ~149 KB → ~10 KB |
| libLLVMEmbeddedJIT.a | 160 KB | 0.3 KB |
| libLLVMPasses.a (大部分) | 6,582 KB | ~184 KB → ~30 KB |

**预估节省: ~300–500 KB 实际链接大小**（gc-sections 已移除大部分，但 .a 大小减少 ~17 MB 可加速链接）

> **注意**: gc-sections 已经移除了大部分无用 pass 的代码，所以 PassBuilder 的链接大小贡献看似只有 184 KB。但 PassBuilder 的真正代价在于它是**传递依赖放大器**——它使得 CMake 必须链接 19 个组件，增加了编译时间和链接时间。移除 PassBuilder 依赖后，这些组件将完全不出现在链接命令中。

---

## 4. Layer 2: 裁剪 CodeGen 依赖

### 4.1 问题

CodeGen 管线占最终二进制的 ~55%，是最大的体积来源：

```
X86CodeGen (4,600 KB) + SelectionDAG (3,066 KB) + CodeGen (4,866 KB)
+ AsmPrinter (661 KB) + GlobalISel (641 KB) + X86Desc (2,535 KB)
+ MC (603 KB) + MCParser (186 KB)
≈ 17,158 KB (59%)
```

这些库是 JIT 编译**必需**的（LLJIT 需要完整的目标代码生成管线），但其中包含大量 EJIT 不需要的功能：

- **SelectionDAG**: ~3,066 KB，EJIT 的小函数不需要 DAG 合并/合法化的完整能力
- **CodeGen**: 226/233 .o 文件链接，包含 MachinePipeliner (186 KB)、大量调度器、寄存器分配器变体
- **AsmPrinter**: 661 KB，其中 DWARF 发射 (DwarfDebug 165 KB) 在嵌入式场景不需要
- **ProfileData**: 257 KB，JIT 不做 PGO

### 4.2 方案 A: AsmPrinter 精简（短期，低风险）

在 LLVM 源码层面添加 CMake 选项，条件编译排除 DWARF/CodeView 发射：

```cmake
# 顶层 CMakeLists.txt 新增选项:
option(LLVM_EJIT_MINIMAL_ASM_PRINTER
  "Build a minimal AsmPrinter without debug info emission for EJIT" OFF)

# llvm/lib/CodeGen/AsmPrinter/CMakeLists.txt:
if(LLVM_EJIT_MINIMAL_ASM_PRINTER)
  # 排除 DWARF 和 CodeView 相关源文件
  # DwarfDebug.cpp, DwarfCompileUnit.cpp, DwarfUnit.cpp,
  # CodeViewDebug.cpp, WinException.cpp, etc.
else()
  # 正常构建
endif()
```

**预估节省: ~500 KB**（DWARF + CodeView 发射代码）

### 4.3 方案 B: CodeGen 功能裁剪（中期，中等风险）

通过 CMake 选项排除 CodeGen 中 EJIT 不需要的子系统：

```cmake
option(LLVM_EJIT_MINIMAL_CODEGEN
  "Build minimal CodeGen for EJIT: no MachinePipeliner, no alternative schedulers, etc." OFF)
```

可排除的 CodeGen 子模块：

| 子模块 | 大小 | 说明 |
|---|---|---|
| MachinePipeliner.cpp | 186 KB | 软件流水线，EJIT 小函数不需要 |
| AssignmentTrackingAnalysis.cpp | 193 KB | 调试信息追踪 |
| PrologueEpilogueInserter.cpp | ~80 KB | 可简化 |
| 多种 Register Allocator 变体 | ~150 KB | 保留 Greedy，移除 PBQP/Basic |
| 多种调度器变体 | ~100 KB | 保留默认调度器 |

**预估节省: ~700 KB**

### 4.4 方案 C: SelectionDAG 替代（长期，高风险）

EJIT 的 JIT 编译对象是小型函数（通常 <100 条 IR 指令），不需要 SelectionDAG 的全部能力。可以考虑：

1. **FastISel-only 模式**: 完全跳过 SelectionDAG，使用 FastISel。但 X86 的 FastISel 不支持所有指令类型，可能回退到 SelectionDAG
2. **GlobalISel-only 模式**: GlobalISel 更模块化，可裁剪性更好。但成熟度和覆盖率不如 SelectionDAG
3. **自定义轻量 ISel**: 为 EJIT 场景手写简化版指令选择，只覆盖 EJIT 常见的 IR 模式

**此方案复杂度高，建议作为长期探索方向，不作为近期目标。**

### 4.5 方案 D: 移除 ProfileData 依赖（短期，低风险）

分析 `ProfileData` 的依赖链：

```
Instrumentation → ProfileData (PGO, GCOV, MemProf)
X86CodeGen → ProfileData (直接链接)
CodeGen → ProfileData (直接链接)
```

EJIT 不做 PGO/Profile，但 ProfileData 中的 `ItaniumManglingCanonicalizer` (156 KB) 被 OrcJIT 用于符号解析。可以：

1. 将 `ItaniumManglingCanonicalizer` 拆分到独立的小库
2. EJIT 仅链接这个小库

**预估节省: ~100 KB**

---

## 5. Layer 3: 编译选项优化

### 5.1 当前已实施

从 git 历史看，以下选项已实施：

- `LLVM_ENABLE_ZLIB=OFF` (release builds)
- `LLVM_ENABLE_ZSTD=OFF` (release builds)
- `LLVM_ENABLE_TERMINFO=OFF`

### 5.2 待实施

| 选项 | 效果 | 风险 |
|---|---|---|
| `-DLLVM_ENABLE_ASSERTIONS=OFF` (release) | 移除 assert 检查，减小代码 | 低（release 默认 OFF） |
| `-DLLVM_ENABLE_EXPENSIVE_CHECKS=OFF` | 移除昂贵检查 | 低（默认 OFF） |
| `-DLLVM_ENABLE_BACKTRACES=OFF` | 移除回溯支持 | 中（影响错误诊断） |
| `-DLLVM_ENABLE_THREADS=ON` | 异步编译需要 | 已设置 |
| 仅构建单 Target | 移除 X86+AArch64 冗余 | 低（按需选择） |
| `-DLLVM_TARGETS_TO_BUILD="AArch64"` (嵌入式) | 移除 X86 代码 | 低（部署时选择） |
| `-DCMAKE_BUILD_TYPE=MinSizeRel` | `-Os` 优化体积 | 低 |

### 5.3 链接时优化 (LTO)

```bash
# 在 ejit_test/build.sh 中启用 LTO:
"${CXX}" -fuse-ld="${LD_LLD}" -flto=thin \
  -Os -Wl,--gc-sections -Wl,--strip-all \
  ...
```

**预估 LTO 节省: 10-15%**（跨模块内联 + 死代码消除）

---

## 6. Layer 4: 深度裁剪（长期方向）

### 6.1 迷你 Analysis 注册

当前 Analysis 库链接了 103/125 个 .o 文件，但 EJIT 只需要约 15 个 analysis。可以创建 `EJitAnalysisRegistrar` 仅注册必需的 analysis：

```
必需 Analysis:
  DominatorTree, AssumptionCache, TargetIRAnalysis, TargetLibraryInfo,
  AAManager + BasicAA, ScalarEvolution, LoopInfo, MemorySSA,
  OptimizationRemarkEmitter, PhiValues, PassInstrumentation
```

**预估节省: ~500 KB**（移除 ~80 个不使用的 analysis .o）

### 6.2 Support 库裁剪

Support 库链接了 690 KB，包含大量 EJIT 不需要的功能（文件系统、进程管理、命令行解析等）。可以拆分为：

- `LLVMSupportCore`: Error, StringRef, raw_ostream, Debug, MemoryBuffer 等基础功能
- `LLVMSupportFS`: 文件系统相关
- `LLVMSupportProcess`: 进程/信号相关

**预估节省: ~200 KB**

### 6.3 代码生成替代方案

最激进的裁剪方向：不使用 LLVM 的完整 CodeGen 管线，而是：

1. **JIT → C → 编译**: 将优化后的 IR 转换为 C 代码，用外部 C 编译器编译
2. **自定义代码生成器**: 仅支持有限的 IR 子集，直接生成机器码
3. **预编译模板**: 在 AOT 阶段生成代码模板，JIT 阶段仅做常量填入

**这些方案风险高、开发量大，仅在极端体积约束（<5 MB）下考虑。**

---

## 7. 实施路线

### Phase 1: 定制 PassBuilder（1-2 天）

| 步骤 | 工作量 | 风险 |
|---|---|---|
| 创建 `EJitPassBuilder.h/.cpp` | 0.5 天 | 低 |
| 修改 `EJitOptimizer` 使用 `EJitPassBuilder` | 0.5 天 | 低 |
| 修改 CMakeLists.txt 移除 Passes 依赖 | 0.5 天 | 低 |
| 运行全部 EJIT 测试验证 | 0.5 天 | 低 |

### Phase 2: AsmPrinter 精简 + ProfileData 拆分（3-5 天）

| 步骤 | 工作量 | 风险 |
|---|---|---|
| 添加 `LLVM_EJIT_MINIMAL_ASM_PRINTER` CMake 选项 | 1 天 | 低 |
| 排除 DWARF/CodeView 源文件 | 1 天 | 中（需验证 AsmPrinter 不崩溃） |
| ProfileData 拆分（提取 ItaniumManglingCanonicalizer） | 2 天 | 中 |
| 测试验证 | 1 天 | 低 |

### Phase 3: CodeGen 裁剪 + LTO（5-7 天）

| 步骤 | 工作量 | 风险 |
|---|---|---|
| 添加 `LLVM_EJIT_MINIMAL_CODEGEN` CMake 选项 | 2 天 | 中 |
| 排除 MachinePipeliner、多余 RA/Scheduler | 2 天 | 中 |
| 启用 Thin LTO | 1 天 | 低 |
| 全面测试 + 性能回归 | 2 天 | 中 |

### Phase 4: 深度裁剪（2-3 周，按需）

| 步骤 | 工作量 | 风险 |
|---|---|---|
| Analysis 注册精简 | 3 天 | 高 |
| Support 库拆分 | 5 天 | 高 |
| 迭代测试 | 5 天 | 中 |

---

## 8. 裁剪效果预估

| 阶段 | 二进制大小 | 节省 | 累计节省 |
|---|---|---|---|
| 当前 | ~30 MB | — | — |
| Phase 1 (定制 PassBuilder) | ~29.5 MB | ~0.5 MB | 1.7% |
| Phase 2 (AsmPrinter + ProfileData) | ~28.5 MB | ~1.0 MB | 5% |
| Phase 3 (CodeGen 裁剪 + LTO) | ~25 MB | ~3.5 MB | 16.7% |
| Phase 4 (深度裁剪) | ~22 MB | ~3.0 MB | 26.7% |

> **嵌入式目标 (ARM/AArch64)**: 由于 ARM/AArch64 的 CodeGen 库比 X86 小（约 60-70%），Phase 1-3 后 ARM 目标二进制预计 ~18 MB。

---

## 9. 风险与约束

| 风险 | 等级 | 缓解措施 |
|---|---|---|
| 自定义 PassBuilder 缺少 analysis 导致 pass 崩溃 | 中 | 充分的单元测试 + 分析 pass 依赖链 |
| 裁剪 AsmPrinter 后 JIT 代码无法生成调试信息 | 低 | 嵌入式场景不需要 DWARF |
| CodeGen 裁剪导致某些 IR 模式无法编译 | 中 | 保留 fallback 路径，运行时检测 |
| LTO 增加构建时间 | 低 | 仅在 release 构建启用 |
| 上游 LLVM 合并冲突 | 高 | 裁剪改动集中在 EJIT 目录 + CMake 选项，最小化侵入 |

---

## 10. 附录: 完整依赖图

```
EJIT 直接依赖:
├── Core → Remarks
├── Support → Demangle
├── BitReader → BitstreamReader
├── OrcJIT → OrcShared, OrcTargetProcess
│   ├── JITLink → Option
│   ├── Object → IRReader → AsmParser
│   └── RuntimeDyld (private)
├── JITLink → Option, MC, Object, Support
├── ExecutionEngine → RuntimeDyld
├── [Passes] → ← Layer 1 移除此依赖
│   ├── AggressiveInstCombine
│   ├── CFGuard
│   ├── CodeGen → CGData, CodeGenTypes, ObjCARC, ProfileData, BitWriter
│   ├── Coroutines
│   ├── EmbeddedJIT → IRReader → AsmParser, BitWriter
│   ├── GlobalISel
│   ├── IPO → Linker, Instrumentation → ProfileData, Vectorize, BitWriter
│   ├── Instrumentation → ProfileData
│   ├── IRPrinter
│   ├── ObjCARC
│   ├── Scalar → AggressiveInstCombine
│   └── Vectorize
├── Scalar → AggressiveInstCombine (传递)
├── InstCombine
├── IPO → Linker, Instrumentation, Vectorize, BitWriter (传递)
├── TransformUtils
├── Analysis
└── X86CodeGen → AsmPrinter → DebugInfoDWARF, DebugInfoCodeView
    ├── SelectionDAG
    ├── GlobalISel
    ├── X86Desc → CodeGenTypes, MCDisassembler
    ├── MC, MCParser
    └── CFGuard, Instrumentation, ProfileData

Layer 1 移除 Passes 后:
├── 直接: Core, Support, BitReader, OrcJIT, JITLink, ExecutionEngine
├── 直接: Scalar, InstCombine, IPO, TransformUtils, Analysis
├── 直接: X86CodeGen, X86Desc, X86Info
├── 保留传递: CodeGen (通过 X86CodeGen), AsmPrinter, SelectionDAG
├── 保留传递: MC, MCParser, Object, RuntimeDyld
├── 保留传递: ProfileData (通过 CodeGen/IPO), BitWriter (通过 IPO)
└── 消除: Coroutines, ObjCARC, CFGuard, Vectorize, AggressiveInstCombine,
           IRPrinter, EmbeddedJIT, IRReader, AsmParser, Linker (大部分)
```

---

*文档版本: 1.0*
*创建日期: 2026-05-24*
