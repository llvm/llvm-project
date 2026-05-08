# EJitAotModulePass 设计文档

**版本**: 1.1
**日期**: 2026-04-29
**关联**: SPEC4.md, PLAN4.md
**类型**: AOT Module Pass (协调器)
**顺序**: 晚期 AOT Pipeline 入口 — 标准优化之后

---

## 1. 概述

EJitAotModulePass 是 AOT 编译 Pipeline 的协调器，负责按序调度 4 个子 Pass 执行。它不包含独立的变换逻辑，而是确保各子 Pass 以正确的顺序和上下文运行，并管理 PreservedAnalyses 的组合。

### 1.1 核心职责

- 按正确顺序调度 3 个子 Pass（不包含 Bitcode 提取）
- 管理 ModuleAnalysisManager 传递
- 组合各子 Pass 的 PreservedAnalyses
- 注册到 LLVM Pass Pipeline（晚期，标准优化之后）
- 提供统一入口供 Clang 后端调用

> **v1.1 变更**: `EJitRegisterBitcodePass` 已从本协调器移除，作为独立的早期 Pass 在标准优化之前单独注册。详见 §1.3。

### 1.2 子 Pass 执行顺序

| 步骤 | Pass | 职责 |
|------|------|------|
| 1 | EJitRegisterPeriodPass | 时间窗变量注册 |
| 2 | EJitWrapperGenPass | Wrapper 插桩 |
| 3 | EJitPeriodHandlerPass | 生命周期处理 |

### 1.3 Pipeline 位置与设计决策

本 Pass 注册在**标准优化 Pipeline（O2/O3）之后**，原因：

1. **Wrapper 插桩**需要在最终 AOT 函数上操作。优化后的 IR 代码质量更高，生成的 Wrapper 和 fallback 路径执行效率更好
2. **Period 变量注册**依赖函数级 `ejit.metadata`，函数级 metadata 一般不会被优化 Pass 移除
3. **生命周期处理**需要在所有内联/优化完成后插入，避免 deactivate/activate 调用被错误内联

**Bitcode 提取为何不在本协调器内**：

`!ejit.may_const` metadata 标注在 `load` 指令上（instruction-level metadata），标准优化（SROA、InstCombine、GVN）会丢弃它。因此 `EJitRegisterBitcodePass` 必须在优化之前独立运行，单独注册为早期 Pass。

```
Clang CodeGen → [EJitRegisterBitcodePass] → O2/O3 优化 → [本 Pass] → 目标代码生成
                   早期, 提取原始 IR                      晚期, wrapper + period 注册
```

---

## 2. 输入与输出

### 2.1 输入

- 经标准优化（O2/O3）后的 LLVM Module
- 已嵌入 `@__ejit_bitcode` 全局变量（由早期 `EJitRegisterBitcodePass` 生成）
- 包含 `ejit_entry` / `ejit_period_lc` 函数（可能已被内联/优化，但 `ejit.metadata` 保留）
- 包含 `ejit_period` / `ejit_period_arr` 全局变量

### 2.2 输出

- Module 含扩展的 `ejit_auto_register()` 函数（追加变量注册调用）
- ejit_entry 函数被改写为单函数混合 Wrapper
- ejit_period_lc 函数入口/出口添加 activate/deactivate 调用

---

## 3. 核心算法

### 3.1 主流程

```cpp
class EJitAotModulePass : public PassInfoMixin<EJitAotModulePass> {
public:
    PreservedAnalyses run(Module& M, ModuleAnalysisManager& AM) {
        // 检查是否有 EmbeddedJIT 相关 metadata
        // 如果整个 Module 都没有 ejit metadata，提前返回
        if (!hasAnyEjitMetadata(M))
            return PreservedAnalyses::all();

        PreservedAnalyses PA = PreservedAnalyses::none();
        bool anyChange = false;

        // Step 1: 时间窗变量注册
        {
            EJitRegisterPeriodPass P1;
            PreservedAnalyses PA1 = P1.run(M, AM);
            if (!PA1.areAllPreserved()) anyChange = true;
            PA.intersect(PA1);
        }

        // Step 2: Wrapper 插桩
        {
            EJitWrapperGenPass P2;
            PreservedAnalyses PA2 = P2.run(M, AM);
            if (!PA2.areAllPreserved()) anyChange = true;
            PA.intersect(PA2);
        }

        // Step 3: 生命周期处理
        {
            EJitPeriodHandlerPass P3;
            PreservedAnalyses PA3 = P3.run(M, AM);
            if (!PA3.areAllPreserved()) anyChange = true;
            PA.intersect(PA3);
        }

        // 防呆检测: 元数据一致性检测 (SPEC4 §9.4)
        if (anyChange) {
            runDiagnosticCheck(M);
        }

        return PA;
    }

    static StringRef name() { return "ejit-aot-module"; }

    // 注册到 Pass Pipeline (通过 PassBuilder)
    static void registerPasses(PassBuilder& PB) {
        PB.registerPipelineParsingCallback(
            [](StringRef Name, ModulePassManager& MPM,
               ArrayRef<PassBuilder::PipelineElement>) {
                if (Name == "ejit-aot-module") {
                    MPM.addPass(EJitAotModulePass());
                    return true;
                }
                return false;
            });
    }
};
```

### 3.2 快速返回优化

```cpp
bool hasAnyEjitMetadata(Module& M) {
    // 快速扫描: 检查全局变量/函数的 ejit.metadata
    // 若无任何 EmbeddedJIT 标记，跳过整个 AOT Pipeline
    // 注意: ejit.metadata 是 per-function / per-global 的 metadata，
    //   通过 setMetadata() 附加，不是 module-level named metadata

    // 检查全局变量
    for (GlobalVariable& GV : M.globals()) {
        if (GV.hasMetadata("ejit.metadata"))
            return true;
    }

    // 检查函数
    for (Function& F : M.functions()) {
        if (F.hasMetadata("ejit.metadata"))
            return true;
    }

    return false;
}
```

### 3.3 元数据一致性诊断 (SPEC4 §9.4)

```cpp
void runDiagnosticCheck(Module& M) {
    // 检查 ejit_entry 函数实际引用的全局变量
    // 是否与其声明的 ejit_period_arr_ind 一致

    for (Function& F : M.functions()) {
        MDNode* MD = F.getMetadata("ejit.metadata");
        if (!MD) continue;

        // 提取声明的 period 依赖集合
        std::set<std::string> declaredPeriods;
        for (const MDOperand& Op : MD->operands()) {
            MDNode* Entry = cast<MDNode>(Op);
            StringRef tag = cast<MDString>(Entry->getOperand(0))->getString();
            if (tag == "ejit_period_arr_ind") {
                declaredPeriods.insert(
                    cast<MDString>(Entry->getOperand(1))->getString().str());
            }
        }

        // 收集函数中实际引用的 ejit_period_arr 全局变量
        std::set<std::string> actualPeriods;
        for (BasicBlock& BB : F) {
            for (Instruction& I : BB) {
                for (Value* op : I.operands()) {
                    if (GlobalVariable* GV = dyn_cast<GlobalVariable>(op)) {
                        if (MDNode* gvMD = GV->getMetadata("ejit.metadata")) {
                            for (const MDOperand& gvOp : gvMD->operands()) {
                                MDNode* gvEntry = cast<MDNode>(gvOp);
                                StringRef gvTag = cast<MDString>(
                                    gvEntry->getOperand(0))->getString();
                                if (gvTag == "ejit_period_arr") {
                                    actualPeriods.insert(
                                        cast<MDString>(
                                            gvEntry->getOperand(1))->getString().str());
                                }
                            }
                        }
                    }
                }
            }
        }

        // 与声明的依赖比较
        for (auto& p : actualPeriods) {
            if (declaredPeriods.find(p) == declaredPeriods.end()) {
                // 警告: 函数引用了 period_arr 但未在参数中声明依赖
                errs() << "ejit warning: function '" << F.getName()
                       << "' references period_arr '" << p
                       << "' but does not declare a dependency on it\n";
            }
        }
    }
}
```

---

## 4. Pipeline 注册

### 4.1 在 PassBuilder 中注册

```cpp
// 在 llvm/lib/Passes/PassBuilder.cpp 中注册
void PassBuilder::registerPasses() {
    // ...
    EJitAotModulePass::registerPasses(*this);
    // ...
}
```

### 4.2 使用方式

```bash
# 通过 clang 自动调用 (集成在后端 pipeline)
clang -c -O2 -fembedjit input.c -o input.o

# 通过 opt 手动调用 (调试)
opt -passes=ejit-aot-module input.ll -S -o output.ll
```

### 4.3 Clang 后端集成

```cpp
// 在 clang/lib/CodeGen/BackendUtil.cpp 中
// 将 ejit-aot-module 添加到 Clang 的后端 Pass Pipeline
static void addEmbeddedJITPasses(const PassBuilder &PB,
                                  ModulePassManager &MPM) {
    MPM.addPass(EJitAotModulePass());
}
```

---

## 5. PreservedAnalyses 语义

```cpp
// 各子 Pass 的 PreservedAnalyses 语义:
//   - none(): 修改了 Module / Function (常态)
//   - all(): 未做任何修改 (无 ejit metadata 的函数/模块)

// 组合规则:
//   若任一子 Pass 返回 none()，整体返回 none()
//   若全部返回 all()，整体返回 all()
//   使用 intersect() 合并
```

---

## 6. 错误处理

| 错误场景 | 处理策略 |
|---------|---------|
| 子 Pass 抛出异常 | 异常向上传播，Pipeline 终止 |
| 无 ejit metadata | 提前返回 all()，零开销 |
| 子 Pass 内部错误 | 各子 Pass 负责错误处理，report_fatal_error 或 warning |
| MAM 未正确传递 | LLVM Pass 框架保证正确性 |

---

## 7. 测试策略

### 7.1 端到端 Lit 测试

```llvm
; test_aot_pipeline.ll
; RUN: opt -passes=ejit-aot-module -S %s | FileCheck %s

; 验证晚期 AOT Pipeline 输出:
; CHECK: define internal void @ejit_auto_register
; CHECK: call void @ejit_register_period_array
; CHECK: call ptr @ejit_compile_or_get
; CHECK: call void @ejit_deactivate_array
; CHECK: call void @ejit_activate_array
; 注意: @__ejit_bitcode 和 ejit_register_bitcode 调用由早期 EJitRegisterBitcodePass 生成
```

### 7.2 验证点

| 验证项 | 方法 |
|--------|------|
| 执行顺序正确 | 检查输出 IR 中各特性出现的顺序 |
| 无 ejit metadata 无修改 | FileCheck: not check |
| 各子 Pass 独立可测试 | 分别运行单个 Pass 验证 |
| 防呆检测触发 warning | 构造不一致场景，检查 stderr |

---

## 8. 实施注意事项

1. **子 Pass 独立性**: 每个子 Pass 通过独立的 `run()` 方法运行，共用 Module 和 MAM。子 Pass 之间通过 IR 的 metadata / 函数结构调整传递信息。

2. **Pass 注册**: 使用 LLVM 的 `PassBuilder::registerPipelineParsingCallback` 注册 Pass 名和构造函数。确保 Pass 使用 `PassInfoMixin` 继承。

3. **Clang 集成点**: 在 `clang/lib/CodeGen/BackendUtil.cpp` 的 `EmitAssemblyHelper::RunOptimizationPipeline` 或类似位置添加 ejit-aot-module。

4. **LTO 兼容性**: 在 LTO 模式下，Pass 可能在链接时运行。确保能处理多个编译单元合并后的 Module（其中不同单元的 metadata 共存）。

5. **性能考量**: `hasAnyEjitMetadata()` 快速扫描可避免对无 EmbeddedJIT 代码的编译单元产生额外开销。对于大多数普通 C/C++ 代码此检查为常数时间。

6. **调试支持**: 单个 Pass 可用 `-passes=ejit-register-bitcode` 等独立运行，方便调试。协调器 Pass 仅用于集成 Pipeline。

---

*文档版本: 1.0*
*创建日期: 2026-04-26*
