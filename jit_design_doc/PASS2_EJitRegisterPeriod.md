# EJitRegisterPeriodPass 设计文档

**版本**: 2.0
**日期**: 2026-04-27
**关联**: SPEC4.md, PLAN4.md
**类型**: AOT Module Pass
**顺序**: AOT Pipeline 第 2 步

> **v2.0 重大变更**: 从 `EJitRegisterLayoutPass` 重命名为 `EJitRegisterPeriodPass`。移除了结构体布局计算 (`StructLayoutInfo`/`FieldLayout`/`computeStructLayout`)、`!ejit.may_const_offsets` 加载、DW_TAG 交叉校验、`ejit_register_layout` 调用生成。仅保留时间窗变量注册功能。`ejit_may_const` 识别已迁移至 PASS6 的 `load->hasMetadata("ejit.may_const")` 机制（v1.2）。

---

## 1. 概述

EJitRegisterPeriodPass 负责扫描 Module 中所有与 EmbeddedJIT 时间窗关联的全局变量，生成运行时注册代码。这些注册信息在 JIT 编译阶段（PASS6）用于获取全局变量的运行时基地址，从而从进程内存中读取 `ejit_may_const` 字段的运行时值。

> 注意: `ejit_may_const` 字段的**识别**由 PASS6 通过 `load` 指令上的 `!ejit.may_const` metadata 直接完成。本 Pass 不参与 may_const 识别，仅负责为全局变量注册运行时基地址。

### 1.1 核心职责

- 扫描 `!ejit.metadata` 定位所有 `ejit_period` / `ejit_period_arr` 全局变量
- 生成 `ejit_register_period_array()` 调用注册数组时间窗变量
- 生成 `ejit_register_static_var()` 调用注册 static 时间窗变量
- 管理 `ejit_auto_register` 函数（与 PASS1 协调）

### 1.2 设计约束

| 约束项 | 说明 |
|--------|------|
| 类型推导 | 不需要 — 不分析结构体类型 |
| ejit_may_const 识别 | 不在本 Pass 处理 — 由 PASS6 的 `!ejit.may_const` load metadata 完成 |
| 结构体布局 | 不需要 — LayoutRegistry 已移除 |
| 嵌套结构体 | 不需要 |
| 注册时机 | `ejit_auto_register()` 中（constructor 阶段执行） |

---

## 2. 输入 IR 格式

### 2.1 全局变量 Metadata

```llvm
; ejit_period(static) 标量变量
@g_boardCfg = dso_local global %struct.BoardConfig zeroinitializer
; !ejit.metadata = distinct !{!0}
; !0 = !{!"ejit_period", !"static"}

; ejit_period_arr(cell) 数组变量
@g_cellCfg = dso_local global [16 x %struct.CellConfig] zeroinitializer
; !ejit.metadata = distinct !{!1}
; !1 = !{!"ejit_period_arr", !"cell", i32 16}

; ejit_period_arr(trp) 数组变量
@g_trpCfg = dso_local global [8 x %struct.TrpConfig] zeroinitializer
; !ejit.metadata = distinct !{!2}
; !2 = !{!"ejit_period_arr", !"trp", i32 8}
```

---

## 3. 核心算法

### 3.1 主流程

```
输入: Module M
输出: Module M (添加时间窗变量注册调用到 ejit_auto_register)

步骤:
1. CollectPeriodVariables(M) → 收集所有 ejit_period / ejit_period_arr 全局变量
2. GeneratePeriodRegistration(M, periodVars) → 插入 ejit_register_period_array / ejit_register_static_var 调用
```

### 3.2 详细伪代码

```cpp
PreservedAnalyses EJitRegisterPeriodPass::run(Module& M, ModuleAnalysisManager& AM) {
    // Step 1: 收集所有 ejit_period / ejit_period_arr 全局变量
    struct PeriodVarInfo {
        GlobalVariable* GV;
        bool isPeriodArr;           // true: ejit_period_arr, false: ejit_period(static)
        std::string periodName;
        uint64_t arraySize;         // 仅 ejit_period_arr 有效
    };
    std::vector<PeriodVarInfo> periodVars;

    for (GlobalVariable& GV : M.globals()) {
        MDNode* MD = GV.getMetadata("ejit.metadata");
        if (!MD) continue;

        for (const MDOperand& Op : MD->operands()) {
            MDNode* Entry = cast<MDNode>(Op);
            StringRef tag = cast<MDString>(Entry->getOperand(0))->getString();

            if (tag == "ejit_period") {
                periodVars.push_back({
                    &GV, false,
                    cast<MDString>(Entry->getOperand(1))->getString().str(),
                    0
                });
            } else if (tag == "ejit_period_arr") {
                periodVars.push_back({
                    &GV, true,
                    cast<MDString>(Entry->getOperand(1))->getString().str(),
                    cast<ConstantInt>(cast<ConstantAsMetadata>(
                        Entry->getOperand(2))->getValue())->getZExtValue()
                });
            }
        }
    }

    if (periodVars.empty())
        return PreservedAnalyses::all();

    // Step 2: 生成注册代码
    Function* registerFn = getOrCreateAutoRegister(M);
    IRBuilder<> Builder(M.getContext());

    // 插入基础块到函数末尾 (返回指令之前)
    BasicBlock& entryBB = registerFn->getEntryBlock();
    ReturnInst* ret = cast<ReturnInst>(entryBB.getTerminator());
    Builder.SetInsertPoint(ret);

    // 注册数组时间窗变量
    Function* ejitRegPeriodArr = getOrDeclareRegPeriodArray(M);
    Function* ejitRegStaticVar = getOrDeclareRegStaticVar(M);
    for (auto& info : periodVars) {
        if (info.isPeriodArr) {
            emitRegisterPeriodArrayCall(Builder, ejitRegPeriodArr,
                info.periodName, info.GV, info.arraySize);
        } else {
            emitRegisterStaticVarCall(Builder, ejitRegStaticVar,
                info.GV);
        }
    }

    return PreservedAnalyses::none();
}

// 获取或创建 ejit_auto_register 函数
// 若 PASS1 已创建，直接返回；否则创建空函数（含 entry 块和 ret void）
Function* getOrCreateAutoRegister(Module& M) {
    Function* F = M.getFunction("ejit_auto_register");
    if (F) return F;

    // 首次创建: 构造空函数体
    FunctionType* FT = FunctionType::get(
        Type::getVoidTy(M.getContext()), false);
    F = Function::Create(FT, GlobalValue::InternalLinkage,
                         "ejit_auto_register", &M);

    BasicBlock* entry = BasicBlock::Create(M.getContext(), "entry", F);
    IRBuilder<> Builder(M.getContext());
    Builder.SetInsertPoint(entry);
    Builder.CreateRetVoid();

    return F;
}
```

### 3.3 注册调用生成

```cpp
// 生成 ejit_register_period_array 调用
void emitRegisterPeriodArrayCall(IRBuilder<>& B, Function* regFn,
                                  const std::string& periodName,
                                  GlobalVariable* GV, uint64_t arraySize) {
    Value* periodNameStr = B.CreateGlobalString(periodName);
    Value* varNameStr = B.CreateGlobalString(GV->getName());
    Value* baseAddr = B.CreatePointerCast(GV, PointerType::getUnqual(B.getContext()));
    Value* arrSize = ConstantInt::get(Type::getInt64Ty(B.getContext()), arraySize);

    B.CreateCall(regFn, {periodNameStr, varNameStr, baseAddr, arrSize});
}

// 生成 ejit_register_static_var 调用
void emitRegisterStaticVarCall(IRBuilder<>& B, Function* regFn,
                                GlobalVariable* GV) {
    Value* varNameStr = B.CreateGlobalString(GV->getName());
    Value* varAddr = B.CreatePointerCast(GV, PointerType::getUnqual(B.getContext()));

    B.CreateCall(regFn, {varNameStr, varAddr});
}
```

---

## 4. 输出 IR 变化

### 4.1 注册函数扩展

```llvm
define internal void @ejit_auto_register() {
entry:
    ; 来自 EJitRegisterBitcodePass 的注册
    call void @ejit_register_bitcode(i8* ..., i8* ..., i64 ...)

    ; === 本 Pass 新增 ===

    ; 数组时间窗变量注册
    call void @ejit_register_period_array(
        i8* getelementptr ... @".str.cell",
        i8* getelementptr ... @".str.g_cellCfg",
        i8* bitcast ([16 x %CellConfig]* @g_cellCfg to i8*),
        i64 16)

    ; static 时间窗变量注册
    call void @ejit_register_static_var(
        i8* getelementptr ... @".str.g_boardCfg",
        i8* bitcast (%BoardConfig* @g_boardCfg to i8*))

    ; 多个时间窗变量...
    ret void
}
```

---

## 5. 关键数据结构

```cpp
// PeriodVarInfo — 时间窗变量信息（本 Pass 内部使用）
struct PeriodVarInfo {
    GlobalVariable* GV;
    bool isPeriodArr;           // true: ejit_period_arr, false: ejit_period(static)
    std::string periodName;
    uint64_t arraySize;         // 仅 ejit_period_arr 有效
};

// 运行时注册 API
// void ejit_register_period_array(const char* periodName, const char* varName,
//                                  void* baseAddr, uint64_t arraySize);
//
// void ejit_register_static_var(const char* varName, void* varAddr);
```

---

## 6. 错误处理

| 错误场景 | 处理策略 |
|---------|---------|
| 无法解析 metadata | 跳过该变量，记录 warning 日志 |
| 无 period 变量 | 返回 PreservedAnalyses::all()，不做修改 |
| ejit_period 缺少 periodName | 跳过该变量 |

---

## 7. 与其他 Pass 的交互

```
EJitRegisterBitcodePass  →  生成 bitcode 段 + ejit_auto_register 入口
        ↓
EJitRegisterPeriodPass   →  扩展注册函数: 时间窗变量注册 (本 Pass)
        ↓
EJitWrapperGenPass       →  读取 ejit.metadata 生成 Wrapper
        ↓
EJitPeriodHandlerPass    →  读取 ejit.metadata 生成生命周期调用
```

| 依赖项 | 来源 | 说明 |
|--------|------|------|
| ejit_auto_register 函数 | EJitRegisterBitcodePass 创建（或本 Pass 自创建） | 向已有函数追加 BasicBlock |
| ejit.metadata | Clang CodeGen 生成 | 定位 ejit_period / ejit_period_arr 全局变量 |
| 运行时注册函数声明 | libejit 提供符号 | ExternalLinkage 声明 |

---

## 8. 测试策略

### 8.1 Lit 测试

```llvm
; test_register_period.ll
; RUN: opt -passes=ejit-register-period -S %s | FileCheck %s

; 输入: 含 ejit_period_arr + ejit_period 的 IR
; CHECK: call void @ejit_register_period_array
; CHECK: call void @ejit_register_static_var

; 验证:
; 1. 多个时间窗变量均注册
; 2. periodName 正确传递
; 3. 无 period 变量的 Module 无变化
```

### 8.2 单元测试

```cpp
// EJitRegisterPeriodTest.cpp (或 Lit 测试)
// TEST(EJitRegisterPeriod, PeriodArrRegistration)
// TEST(EJitRegisterPeriod, StaticVarRegistration)
// TEST(EJitRegisterPeriod, MultiArraysRegistration)
// TEST(EJitRegisterPeriod, EmptyModule)
```

### 8.3 集成测试

```
1. 编译 C 源文件 (含 struct + ejit_period_arr)
2. 运行 Pass pipeline
3. 检查可执行文件的 llvm.global_ctors 段
4. 运行时验证: ejit_init 后 PeriodArrayRegistry 包含正确条目
```

---

## 9. 实施注意事项

1. **ejit_auto_register 函数生命周期**: 若 EJitRegisterBitcodePass 已创建该函数，本 Pass 只需追加指令；若不存在则自行创建。函数在 `@llvm.global_ctors` 中被引用（不可被 DCE 移除），且内部调用均为 `ExternalLinkage` 运行时函数（不可被内联），因此经过 O2/O3 优化后函数体结构（含 `entry` 块和 `ret void`）保持不变，本 Pass 可安全地在返回前插入新调用。**注意**: 代码中使用 `cast<ReturnInst>(entryBB.getTerminator())` 仅对 entry 块的 terminator 操作，假定函数为单块结构。此假设成立的原因：(1) `InternalLinkage` + `global_ctors` 引用保证不被优化删除；(2) 函数体仅为顺序调用序列 + `ret void`，无控制流分支，优化器不会将单块函数拆分为多块。

2. **无 structName 参数**: v2.0 起 `ejit_register_period_array` 和 `ejit_register_static_var` 不再接收 `structName` 参数。运行时不需要结构体名（LayoutRegistry 已移除）。

3. **无 elementType 分析**: 本 Pass 不分析结构体类型。类型信息由 PASS6 从 `load->getType()` 直接获取。

4. **与 PASS6 的分工**: 本 Pass (AOT) 负责"在哪里"——注册全局变量运行时地址。PASS6 (JIT) 负责"是什么"——通过 `!ejit.may_const` load metadata 识别 may_const 字段并读取值。

5. **Pass 名称**: Pipeline 中通过 `-passes=ejit-register-period` 调用。

---

*文档版本: 2.0*
*创建日期: 2026-04-26*
*最后更新: 2026-04-27*

---

## 静态注册表生成 (v1.7)

裸核环境无构造器，PASS2 同时生成 `__ejit_registry_period[]` 全局常量数组。

- Period 数组条目: `{EJIT_REG_PERIOD_ARRAY(1), periodName, varName, baseAddr, size}`
- Static var 条目: `{EJIT_REG_STATIC_VAR(2), varName, NULL, addr, 0}`
- 末尾 sentinel: `{EJIT_REG_NONE(4), NULL, NULL, NULL, 0}`

`ejit_init()` 中与 `__ejit_registry_bitcode[]` 一起遍历。

---

*文档版本: 1.1*  
*更新日期: 2026-06-01*
