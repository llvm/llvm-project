# EJitPeriodHandlerPass 设计文档

**版本**: 1.0
**日期**: 2026-04-26
**关联**: SPEC4.md, PLAN4.md
**类型**: AOT Module Pass
**顺序**: AOT Pipeline 第 4 步

---

## 1. 概述

EJitPeriodHandlerPass 负责处理 `ejit_period_lc` (lifecycle) 属性标记的函数。在这些函数的入口插入 `ejit_deactivate_array` 调用，在所有出口点插入 `ejit_activate_array` 调用。这确保在修改时间窗数据期间，相关时间窗处于 deactive 状态，防止其他线程读到不一致的数据或触发对旧值的 JIT 编译。

### 1.1 核心职责

- 定位所有带 `!{"ejit_period_lc", !"periodName"}` metadata 的函数
- 识别函数参数中对应 `ejit_period_arr_ind(periodName)` 的参数
- 在函数入口（第一条指令之前）插入 `ejit_deactivate_array` 调用
- 在函数所有出口点（return 指令之前）插入 `ejit_activate_array` 调用
- 支持一个函数标记多个 `ejit_period_lc`（多时间窗管理）
- 处理 early return 场景 — 所有 return 指令前都需插入

### 1.2 设计约束

| 约束项 | 说明 |
|--------|------|
| 多时间窗 | 单函数可标记多个 `ejit_period_lc`，需按序配对激活/去激活 |
| 异常处理 | C 语言无异常，不考虑 exception handling |
| 多出口 | C 函数可有多处 return，均需插入 activate 调用 |
| ejit_period_lc 必须与 ejit_period_arr_ind 配对 | Sema 已检查，Pass 不做验证 |
| static 时间窗 | `ejit_period(static)` 永远激活，不参与 activate/deactivate |

---

## 2. 输入 IR 格式

### 2.1 ejit_period_lc 函数 Metadata

```llvm
; 单时间窗生命周期函数
define void @update_cell(i32 %cellIdx) #0 {
  ; ...
}
; !ejit.metadata = distinct !{!0, !1}
; !0 = !{!"ejit_period_lc", !"cell"}
; !1 = !{!"ejit_period_arr_ind", !"cell", i32 0}    ; 参数 0 对应 cell 的索引
;      ← 此 metadata 由 Clang CodeGen 在函数级别附加

; 多时间窗生命周期函数
define void @update_both(i32 %cellIdx, i32 %trpIdx) #0 {
  ; ...
}
; !ejit.metadata = distinct !{!0, !1, !2, !3}
; !0 = !{!"ejit_period_lc", !"cell"}
; !1 = !{!"ejit_period_lc", !"trp"}
; !2 = !{!"ejit_period_arr_ind", !"cell", i32 0}
; !3 = !{!"ejit_period_arr_ind", !"trp", i32 1}
```

### 2.2 Metadata 格式说明

`ejit_period_arr_ind` metadata 在函数级别编码为：
```llvm
!{!"ejit_period_arr_ind", !"periodName", i32 argIndex}
```

此 metadata 同时服务于 `EJitWrapperGenPass` 和 `EJitPeriodHandlerPass`。

---

## 3. 核心算法

### 3.1 主流程

```
输入: Module M (含 ejit_period_lc 函数)
输出: Module M (函数入口 + 出口插入 activate/deactivate 调用)

步骤:
1. CollectLifecycleFunctions(M) → 收集所有 ejit_period_lc 函数及其 metadata
2. ForEach lifecycleFunc:
     ParseLifecycleInfo(funcMeta) → 解析 periodName → argIdx 映射
3. InsertDeactivateAtEntry(func, lifecycleInfo) → 入口插入 deactivate_array
4. InsertActivateAtExits(func, lifecycleInfo) → 所有 return 前插入 activate_array
```

### 3.2 详细伪代码

```cpp
PreservedAnalyses EJitPeriodHandlerPass::run(Module& M, ModuleAnalysisManager& AM) {
    struct LifecycleInfo {
        std::string periodName;
        unsigned argIdx;         // ejit_period_arr_ind 参数索引
        Value* arrayPtr;         // 可选: 对应全局数组的指针 (arcname 版本)
    };
    std::vector<std::pair<Function*, std::vector<LifecycleInfo>>> lcFuncs;

    for (Function& F : M.functions()) {
        MDNode* MD = F.getMetadata("ejit.metadata");
        if (!MD) continue;

        std::vector<LifecycleInfo> lcInfo;
        bool isLifecycle = false;

        for (const MDOperand& Op : MD->operands()) {
            MDNode* Entry = cast<MDNode>(Op);
            StringRef tag = cast<MDString>(Entry->getOperand(0))->getString();

            if (tag == "ejit_period_lc") {
                isLifecycle = true;
                std::string periodName = cast<MDString>(
                    Entry->getOperand(1))->getString().str();

                // 查找对应的 ejit_period_arr_ind 参数
                int argIdx = findPeriodArrIndArg(MD, periodName);
                if (argIdx >= 0) {
                    lcInfo.push_back({periodName, (unsigned)argIdx, nullptr});
                }
            }
        }

        if (isLifecycle && !lcInfo.empty()) {
            lcFuncs.push_back({&F, lcInfo});
        }
    }

    for (auto& [F, lcInfoList] : lcFuncs) {
        insertDeactivateAtEntry(F, lcInfoList);
        insertActivateAtExits(F, lcInfoList);
    }

    return lcFuncs.empty() ? PreservedAnalyses::all() : PreservedAnalyses::none();
}
```

### 3.3 入口 Deactivate 插入

```cpp
void insertDeactivateAtEntry(Function* F,
                              const std::vector<LifecycleInfo>& lcInfoList) {
    LLVMContext& Ctx = F->getContext();
    BasicBlock& entryBB = F->getEntryBlock();

    // 找到 entry 块的第一条非 alloca 指令
    Instruction* firstNonAlloca = entryBB.getFirstNonPHI();
    while (isa<AllocaInst>(firstNonAlloca) && firstNonAlloca != nullptr) {
        firstNonAlloca = firstNonAlloca->getNextNode();
    }

    IRBuilder<> Builder(Ctx);
    if (firstNonAlloca) {
        Builder.SetInsertPoint(firstNonAlloca);
    } else {
        Builder.SetInsertPoint(&entryBB);
    }

    // 声明运行时函数
    Function* deactivateFn = getOrDeclareDeactivateArray(M);

    // [BiSheng] 为每个 lc 时间窗插入 deactivate_array 调用
    // 调用顺序: metadata 中出现顺序 (与 activate 配对)
    for (auto& lcInfo : lcInfoList) {
        // 参数: periodName 字符串 + arrayPtr + cellIdx
        Value* periodNameStr = Builder.CreateGlobalStringPtr(lcInfo.periodName);
        Value* argVal = F->getArg(lcInfo.argIdx);

        // [BiSheng] 确定对应的全局数组指针
        // 从 metadata 推断: ejit_period_arr_ind 关联的 periodName → 同名 ejit_period_arr 全局变量
        Value* arrayPtr = getArrayPtrForPeriod(M, lcInfo.periodName);
        // 若找不到对应数组，arrayPtr = null (fallback 到 period 级 deactivate)
        if (!arrayPtr) {
            arrayPtr = ConstantPointerNull::get(PointerType::getUnqual(Ctx));
        }

        Value* cellIdx = Builder.CreateZExtOrTrunc(argVal, Type::getInt32Ty(Ctx));

        Builder.CreateCall(deactivateFn, {periodNameStr, arrayPtr, cellIdx});
    }
}
```

### 3.4 出口 Activate 插入

```cpp
void insertActivateAtExits(Function* F,
                            const std::vector<LifecycleInfo>& lcInfoList) {
    LLVMContext& Ctx = F->getContext();
    Function* activateFn = getOrDeclareActivateArray(M);

    // 收集所有 return 指令
    std::vector<ReturnInst*> returnInsts;
    for (BasicBlock& BB : *F) {
        if (ReturnInst* RI = dyn_cast<ReturnInst>(BB.getTerminator())) {
            returnInsts.push_back(RI);
        }
    }

    // 在每个 return 指令之前插入 activate 调用
    // 注意: 插入顺序与 deactivate 相反 (后进先出)
    for (ReturnInst* RI : returnInsts) {
        IRBuilder<> Builder(Ctx);
        Builder.SetInsertPoint(RI); // 插入点: return 之前

        // [BiSheng] 逆序插入 activate (配对: deactivate(A)→deactivate(B)→activate(B)→activate(A))
        for (auto it = lcInfoList.rbegin(); it != lcInfoList.rend(); ++it) {
            Value* periodNameStr = Builder.CreateGlobalStringPtr(it->periodName);
            Value* argVal = F->getArg(it->argIdx);
            Value* arrayPtr = getArrayPtrForPeriod(M, it->periodName);
            if (!arrayPtr) {
                arrayPtr = ConstantPointerNull::get(PointerType::getUnqual(Ctx));
            }
            Value* cellIdx = Builder.CreateZExtOrTrunc(argVal, Type::getInt32Ty(Ctx));

            Builder.CreateCall(activateFn, {periodNameStr, arrayPtr, cellIdx});
        }
    }
}
```

### 3.5 全局数组指针查找

```cpp
Value* getArrayPtrForPeriod(Module& M, const std::string& periodName) {
    // [BiSheng] 查找与 periodName 匹配的 ejit_period_arr 全局变量
    // 遍历所有全局变量的 ejit.metadata:
    //   找 !{"ejit_period_arr", !"<periodName>" ...}

    for (GlobalVariable& GV : M.globals()) {
        MDNode* MD = GV.getMetadata("ejit.metadata");
        if (!MD) continue;

        for (const MDOperand& Op : MD->operands()) {
            MDNode* Entry = cast<MDNode>(Op);
            StringRef tag = cast<MDString>(Entry->getOperand(0))->getString();
            if (tag == "ejit_period_arr") {
                StringRef pn = cast<MDString>(Entry->getOperand(1))->getString();
                if (pn == periodName) {
                    // 返回指向该数组的 i8* 指针
                    return ConstantExpr::getBitCast(&GV,
                        PointerType::getUnqual(M.getContext()));
                }
            }
        }
    }
    return nullptr;
}
```

---

## 4. 输出 IR 变化

### 4.1 单时间窗函数

```llvm
; 输入:
define void @update_cell(i32 %cellIdx) {
entry:
  ; ... 函数体 ...
  ret void
}

; 输出:
define void @update_cell(i32 %cellIdx) {
entry:
  ; === 入口: deactivate ===
  call void @ejit_deactivate_array(ptr @".str.cell",
                                    ptr bitcast ([16 x %CellConfig]* @g_cellCfg to ptr),
                                    i32 %cellIdx)
  ; === 原函数体 ===
  ; ...
  ; === 出口: activate ===
  call void @ejit_activate_array(ptr @".str.cell",
                                  ptr bitcast ([16 x %CellConfig]* @g_cellCfg to ptr),
                                  i32 %cellIdx)
  ret void
}
```

### 4.2 多时间窗函数 (多出口)

```llvm
; 输入:
define void @update_both(i32 %cellIdx, i32 %trpIdx) {
entry:
  %cmp = icmp slt i32 %cellIdx, 0
  br i1 %cmp, label %early_exit, label %body

body:
  ; ... 函数体 ...
  ret void

early_exit:
  ret void
}

; 输出:
define void @update_both(i32 %cellIdx, i32 %trpIdx) {
entry:
  ; [BiSheng] deactivate 按 metadata 出现顺序 (cell 先, trp 后)
  call void @ejit_deactivate_array(ptr @".str.cell", ptr bitcast (%CellConfig* @g_cellCfg to ptr), i32 %cellIdx)
  call void @ejit_deactivate_array(ptr @".str.trp",  ptr bitcast (%TrpConfig* @g_trpCfg to ptr),   i32 %trpIdx)

  %cmp = icmp slt i32 %cellIdx, 0
  br i1 %cmp, label %early_exit, label %body

body:
  ; ... 函数体 ...
  ; [BiSheng] activate 逆序 (trp 先, cell 后)
  call void @ejit_activate_array(ptr @".str.trp",  ptr bitcast (%TrpConfig* @g_trpCfg to ptr),   i32 %trpIdx)
  call void @ejit_activate_array(ptr @".str.cell", ptr bitcast (%CellConfig* @g_cellCfg to ptr), i32 %cellIdx)
  ret void

early_exit:
  ; [BiSheng] 所有 return 前均插入 activate (同样逆序)
  call void @ejit_activate_array(ptr @".str.trp",  ptr bitcast (%TrpConfig* @g_trpCfg to ptr),   i32 %trpIdx)
  call void @ejit_activate_array(ptr @".str.cell", ptr bitcast (%CellConfig* @g_cellCfg to ptr), i32 %cellIdx)
  ret void
}
```

### 4.3 无对应数组的时间窗 (period 级 activate)

```llvm
; 当找不到与 periodName 匹配的 ejit_period_arr 全局变量时
; 使用 null 作为 arrayPtr，运行时做 period 级 activate/deactivate
call void @ejit_deactivate_array(ptr @".str.custom_p", ptr null, i32 %idx)
; ...
call void @ejit_activate_array(ptr @".str.custom_p", ptr null, i32 %idx)
```

---

## 5. 关键数据结构

```cpp
// [BiSheng] 生命周期函数信息
struct LifecycleFuncInfo {
    Function* F;
    std::vector<PeriodAssociation> associations;
};

// [BiSheng] 时间窗 lc → period_arr_ind 关联
struct PeriodAssociation {
    std::string periodName;      // ejit_period_lc 参数
    unsigned paramIdx;           // 对应的 ejit_period_arr_ind 参数索引
    GlobalVariable* periodArr;   // 关联的全局数组 (可为 null)
};

// [BiSheng] 运行时 API 声明
// void ejit_deactivate_array(const char* periodName, void* arrayPtr, int cellIdx);
// void ejit_activate_array(const char* periodName, void* arrayPtr, int cellIdx);
```

---

## 6. 错误处理

| 错误场景 | 处理策略 |
|---------|---------|
| ejit_period_lc 无对应 ejit_period_arr_ind | 跳过（Sema 已处理） |
| 找不到对应 period 全局变量 | 传 null 作为 arrayPtr，运行时做 period 级操作 |
| alloca 干扰入口点定位 | 跳过 alloca 指令，在第一条非 alloca 后插入 |
| 无 return 指令的函数 | 不应发生（well-formed IR 至少有一个 return / unreachable） |
| unreachable 指令 | 不插入（unreachable 不是正常出口） |

---

## 7. 与其他 Pass 的交互

```
EJitRegisterBitcodePass  →  提取 bitcode
EJitRegisterPeriodPass   →  注册时间窗变量
EJitWrapperGenPass       →  生成 Wrapper
        ↓
EJitPeriodHandlerPass    →  处理生命周期函数 (本 Pass - 最后一步)
```

| 依赖项 | 说明 |
|--------|------|
| ejit.metadata | Clang CodeGen 生成的函数 + 全局变量 metadata |
| ejit_deactivate_array / ejit_activate_array | 运行时库提供的外部符号 |
| 全局数组 IR 变量 | 从 Module 中查找 (不需要前序 Pass 的预处理) |

---

## 8. 测试策略

### 8.1 Lit 测试

```llvm
; test_period_handler.ll
; RUN: opt -passes=ejit-period-handler -S %s | FileCheck %s

; 测试场景:
; TEST 1: 单 ejit_period_lc + 单出口
;   CHECK: call void @ejit_deactivate_array
;   CHECK: ret void
;   CHECK: call void @ejit_activate_array
;
; TEST 2: 多 ejit_period_lc (cell + trp) + 多出口
;   CHECK-DAG: call void @ejit_deactivate_array(ptr @".str.cell"
;   CHECK-DAG: call void @ejit_deactivate_array(ptr @".str.trp"
;   CHECK-DAG: call void @ejit_activate_array(ptr @".str.trp"
;   CHECK-DAG: call void @ejit_activate_array(ptr @".str.cell"
;   验证逆序: activate 顺序为 trp 先 cell 后
;
; TEST 3: 无 ejit_period_lc 函数的 Module
;   CHECK-NOT: call void @ejit_deactivate_array
;
; TEST 4: 所有 return 前均有 activate
;   计数 check: activate 出现次数 = return 指令数 × period 数
```

### 8.2 验证点

| 验证项 | 方法 |
|--------|------|
| deactivate/activate 配对 | 计数检查 |
| 逆序配对 | 检查 activate 的插入顺序 |
| 多出口完整性 | 检查所有 return 前均有 activate |
| arrayPtr 正确性 | 检查 bitcast 指向正确的全局变量 |
| 无 lifecycle 函数的 Module | 不做修改 |

---

## 9. 实施注意事项

1. **Deactivate/Activate 顺序**: 对于多时间窗函数，deactivate 按 metadata 出现顺序，activate 逆序。这是经典的 RAII 配对模式。若 cellIdx/trpIdx 的 activate 之间有依赖，可改为显式的顺序配对（同序），但逆序是更安全的默认。

2. **Return 指令定位**: 使用 `BasicBlock::getTerminator()` 获取 return / unreachable 指令，`Builder.SetInsertPoint(RI)` 在 return 之前插入 activate 调用。

3. **空函数**: 仅有 `entry:` 和 `ret void` 的空函数中，deactivate 在 entry 开头插入，activate 在 ret 前插入。

4. **noreturn 函数**: 如果函数有 unreachable terminator（如调用 `abort()` 后），不插入 activate。因为 deactivate 永久生效是没有意义的，应该让 Sema 警告这种情况。

5. **运行时接口选择**: 当前计划使用 `ejit_deactivate_array` / `ejit_activate_array`（数组级）。如果未来需要 period 级接口（`ejit_activate(periodName, cellIdx)`），可通过传 null arrayPtr 切换。

---

*文档版本: 1.0*
*创建日期: 2026-04-26*
