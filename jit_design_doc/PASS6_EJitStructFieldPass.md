# EJitStructFieldPass 设计文档

**版本**: 1.5
**日期**: 2026-04-29
**关联**: SPEC4.md, PLAN4.md
**类型**: JIT Module Pass
**顺序**: JIT Pipeline 第 4 步 (参数预处理 → InstCombine → Inline 之后)

> **v1.5 变更**: 支持指针全局变量。当 GEP 基址是间接 load（load ptr from GV）时，PASS6 多追溯一层 —— 先从 GV 读取指针值作为真正的基地址，再加字段偏移。详见 §2.1 模式 5、§3.2。
>
> **v1.4 变更**: 移除 `getConstantArrayIndex`。`accumulateConstantOffset` 隐含验证所有 GEP 索引为常量。
>
> **v1.3 变更**: JIT Pipeline 重排，Inline 移到本 Pass 之前。新增直接 GlobalVariable load（无 GEP）。
>
> **v1.2 变更**: `ejit_may_const` 识别从 byte offset + LayoutRegistry 匹配改为 `load` 指令上的 `!ejit.may_const` metadata 标注。

---

## 1. 概述

EJitStructFieldPass 是 JIT 编译 Pipeline 中的核心特化 Pass。它在 JIT 编译时运行，负责扫描加载的 Bitcode Module 中所有 Load 指令，识别访问 `ejit_may_const` 字段的模式，从进程内存中读取运行时实际值，并将 Load 指令替换为编译期常量。从而启用后续的常量传播、死代码消除和分支折叠等优化。

### 1.1 核心职责

- 扫描 Module 中所有 `load` 指令，检查 `!ejit.may_const` metadata 标注
- 从标注的 load 指令反向追踪 GEP 链，确定源全局变量和 byte offset（用于运行时地址计算）
- 识别关联的 `ejit_period` / `ejit_period_arr` 时间窗信息
- 从进程内存读取运行时实际值
- 将 Load 指令替换为 LLVM Constant 值
- 删除已替换的死指令

> **v1.2 设计变更**: `ejit_may_const` 识别不再通过 offset + LayoutRegistry 匹配。改为由 Clang CodeGen 在 `load` 指令上直接标注 `!ejit.may_const` metadata（仿造 `volatile` 的传参链，但用 metadata 而非 instruction bit）。LayoutRegistry 已废弃。

### 1.2 前置条件

| 条件 | 说明 |
|------|------|
| ejit_period_arr_ind 参数已替换 | 函数参数已被 JIT 编译器（非 Pass）替换为 ConstantInt |
| InstCombine 已运行 | zext/GEP 常量链已折叠，PHI 节点已清理 |
| Inline 已运行 | 被调用函数已内联，跨函数 GEP 链展开为对全局变量的直接 GEP |
| PeriodArrayRegistry 已初始化 | 由 ejit_init → ejit_auto_register 填入，提供运行时 baseAddr |
| SpecializationContext 可用 | 包含当前的 periodName → cellIdx 映射 |
| 时间窗状态为 Active | 若未激活则不执行特化（调用方提前检查） |

> **v1.3**: 新增 InstCombine + Inline 前置条件。Inline 在 PASS6 之前运行，确保被调用函数内部的 `ejit_may_const` load 已在入口函数体内展开，GEP 链可直接追溯到全局变量。
>
> **v1.2**: `LayoutRegistry` 已移除。may_const 识别通过 `load->hasMetadata("ejit.may_const")` 完成，无需 LayoutRegistry。

---

## 2. 输入 IR 模式识别

### 2.0 `!ejit.may_const` Load Metadata（v1.2 新增）

`ejit_may_const` 标注在 `load` 指令上，而非放在命名 metadata 中通过 offset 间接匹配：

```llvm
; Clang CodeGen 生成: ejit_may_const 字段的 load 携带 metadata
@g_boardCfg = dso_local global %struct.BoardConfig zeroinitializer

define void @jit_entry() {
  %field = getelementptr %struct.BoardConfig, ptr @g_boardCfg, i32 0, i32 0
  ; ejit_may_const 字段 boardType → load 标注 !ejit.may_const
  %val = load i32, ptr %field, !ejit.may_const !{}   ; ← 目标
}
```

**为什么这样设计**（vs `!ejit.may_const_offsets` + offset 匹配）：

| 场景 | offset 匹配（旧） | load metadata（新） |
|------|------------------|---------------------|
| 标准 GEP + load | 需 accumulateConstantOffset + findByOffset | 直接 hasMetadata 判定 |
| SROA 拆分 struct 为标量 | GEP 链消失 → 匹配失败 | metadata 随 load 传播 → 仍可识别 |
| InstCombine GEP 合并 | offset 计算仍可工作 | metadata 不受影响 |
| memcpy 访问 may_const 字段 | 无法识别 | 无法识别（需 scalarized load） |
| 非 may_const 的同 offset 字段 | **误判风险** — 纯 offset 匹配无法区分 | 无风险 — 只有实际标注的 load |
| metadata 被某 pass 丢弃 | — | 不会导致错误，只错过一次 JIT 优化 |

**设计原则**: `!ejit.may_const` 是软标注，不改变任何优化行为。metadata 丢失仅导致错过一次特化机会（JIT fallback 到 AOT 结果），不影响正确性。这区别于 `volatile`（丢失会导致错误行为，因此 `volatile` 必须是 instruction bit）。

### 2.1 基本识别模式

```llvm
; 模式 1: 标量 ejit_period(static) 全局变量字段访问 (通过 GEP)
@g_boardCfg = dso_local global %struct.BoardConfig zeroinitializer

define void @jit_entry() {
  ; GEP 计算字段地址: @g_boardCfg + offsetof(BoardConfig, boardType)
  %field = getelementptr %struct.BoardConfig, ptr @g_boardCfg, i32 0, i32 0
  ; Load 读取字段值
  %val = load i32, ptr %field, !ejit.may_const !{}     ; ← 目标: 替换为运行时常量
}

; 模式 1b: 标量 ejit_period(static) 直接 load (无 GEP, v1.3 新增)
@g_simple_flag = dso_local global i32 0  ; ejit_period(static)
; !ejit.metadata = !{!"ejit_period", !"static"}

define void @jit_entry() {
  %val = load i32, ptr @g_simple_flag, !ejit.may_const !{}  ; ← 目标
}

; 模式 2: ejit_period_arr 数组元素字段访问 (单维度, cellIdx 已特化)
@g_cellCfg = dso_local global [16 x %struct.CellConfig] zeroinitializer

; 参数 %cellIdx 已被替换为常量 3 (由前置参数预处理完成)
; InstCombine 已折叠 zext i8 3 to i64 → i64 3
define void @jit_entry(i32 %cellIdx) {
  ; GEP 计算数组元素地址: @g_cellCfg + 3 * sizeof(CellConfig)
  %elem = getelementptr [16 x %struct.CellConfig], ptr @g_cellCfg, i32 0, i32 3
  ; GEP 计算字段地址
  %field = getelementptr %struct.CellConfig, ptr %elem, i32 0, i32 0
  %val = load i32, ptr %field, !ejit.may_const !{}     ; ← 目标
}

; 模式 3: 多维度 (cellIdx + trpIdx 均已特化)
@g_trpCfg = dso_local global [8 x %struct.TrpConfig] zeroinitializer

define void @jit_entry(i32 %cellIdx, i32 %trpIdx, i32 %iter) {
  ; cellIdx → 常量 1, trpIdx → 常量 5 (已由前置处理替代 + InstCombine 折叠)
  %cell_elem = getelementptr [16 x %struct.CellConfig], ptr @g_cellCfg, i32 0, i32 1
  %cell_field = getelementptr %struct.CellConfig, ptr %cell_elem, i32 0, i32 0
  %cell_val = load i32, ptr %cell_field, !ejit.may_const !{}    ; ← 目标 1

  %trp_elem = getelementptr [8 x %struct.TrpConfig], ptr @g_trpCfg, i32 0, i32 5
  %trp_field = getelementptr %struct.TrpConfig, ptr %trp_elem, i32 0, i32 0
  %trp_val = load i32, ptr %trp_field, !ejit.may_const !{}      ; ← 目标 2
}

; 模式 4: 内联后的跨函数 may_const load (v1.3 新增)
; C 源码: helper(&g_cellCfg[cellIndex]);
; Inline 后:
define void @jit_entry(i32 %cellIdx) {
  ; helper 的 cfg 参数被替换为 @g_cellCfg + cellIndex * sizeof(CellConfig)
  ; cellIdx 已常量化 → 完整 GEP 链可见
  %cfg_field = getelementptr [16 x %struct.CellConfig], ptr @g_cellCfg, i32 0, i32 3, i32 0
  %val = load i32, ptr %cfg_field, !ejit.may_const !{}          ; ← 目标
}

; 模式 5: 指针全局变量间接访问 (v1.5 新增)
; C 源码:
;   ejit_period(static) struct BoardConfig* g_pBoardCfg;
;   ... = g_pBoardCfg->boardType;  // boardType 是 ejit_may_const
@g_pBoardCfg = dso_local global ptr null  ; 存的是指针值

define void @jit_entry() {
  %ptr = load ptr, ptr @g_pBoardCfg            ; 间接 load: 读指针值
  %field = getelementptr %BoardConfig, ptr %ptr, i32 0, i32 0
  %val = load i32, ptr %field, !ejit.may_const !{}  ; ← 目标
}
; 处理: *(void**)&g_pBoardCfg → 得到 &BoardConfig → + offsetof(boardType) → 常量

; 模式 5b: 指针数组 (v1.5 新增)
;   ejit_period_arr(cell) struct CellConfig* g_cellPtrs[N];
;   ... = g_cellPtrs[cellIndex]->cellType;
@g_cellPtrs = dso_local global [16 x ptr] zeroinitializer

define void @jit_entry(i32 %cellIdx) {
  %ptr_slot = getelementptr [16 x ptr], ptr @g_cellPtrs, i32 0, i32 3
  %ptr = load ptr, ptr %ptr_slot                  ; 先读指针值
  %field = getelementptr %CellConfig, ptr %ptr, i32 0, i32 0
  %val = load i32, ptr %field, !ejit.may_const !{}  ; ← 目标
}
; 处理: *(void**)(&g_cellPtrs + 3*sizeof(ptr)) → 得到 &CellConfig → + 0 → 常量
```

### 2.2 GEP 模式分析

GEP 指令的模式提供了完整的信息用于替换：

| GEP 部分 | 提供的信息 | 用途 |
|----------|-----------|------|
| 操作数 operands[0] | 全局变量 `@g_cellCfg` | 确定 periodName (从 ejit.metadata) |
| 操作数 operands[1] | 数组索引 0 (first-class type 索引) | 恒为 0 |
| 操作数 operands[2] | cellIdx 值 (常量) | 确定数组元素索引 |
| 操作数 operands[3] | 结构体字段索引 | 确定字段偏移 |

### 2.3 非模式化 GEP 处理

某些 GEP 链可能因为优化被拆分或合并，需要处理变体：

```llvm
; 变体 1: 多层 GEP 链 (正常模式, 最常见)
%a = getelementptr [N x %S], ptr @g_arr, i32 0, i32 %idx  ; 数组索引
%b = getelementptr %S, ptr %a, i32 0, i32 0                 ; 字段索引
%v = load i32, ptr %b

; 变体 2: 单层 GEP (合并模式, LLVM 优化可能产生)
%field = getelementptr [N x %S], ptr @g_arr, i32 0, i32 %idx, i32 0
%v = load i32, ptr %field
; PASS6 通过 load metadata 识别，GEP 仅用于地址计算

; 变体 3: pb 全局变量的独立 GEP (标量模式)
%field = getelementptr %S, ptr @g_static, i32 0, i32 1
%v = load i32, ptr %field

; 变体 4: 位域操作 (暂不支持)
; 位域通过 { i8*, i32, i32 } 复杂操作访问 → 跳过

; 变体 5: 直接 GlobalVariable load，无 GEP (v1.3 新增)
@g_flag = dso_local global i32 0
%v = load i32, ptr @g_flag, !ejit.may_const !{}
; 指针操作数直接是 GlobalVariable → byteOffset = 0
```

---

## 3. 核心算法

### 3.1 主流程

```
输入: Module M (bitcode), PeriodArrayRegistry, SpecializationContext
输出: Module M (ejit_may_const Load 替换为常量)

步骤:
1. BuildGVPeriodMap(M) → 构建全局变量 → periodName 映射
2. ForEach loadInst in M:
     FilterByMetadata(loadInst) → 检查 hasMetadata("ejit.may_const")，无则跳过  ← v1.2
     ResolveGEPOrGV(loadInst) → 直接 GV → byteOffset=0; GEP → accumulateConstantOffset
                                 (v1.4: accumulateConstantOffset 隐含验证所有索引为常量)
     LookupPeriod(GV, gvPeriodMap) → 查找关联的 period 信息
     ReadRuntimeValue(baseAddr + byteOffset) → 从进程内存读取运行时值
     ReplaceLoadWithConstant(loadInst, runtimeValue) → 替换
     CleanupDeadGEPs → 使用 DCE
```

**v1.2 识别机制变更**:
- 旧: `CheckMayConst(byteOffset, LayoutRegistry)` — 通过 offset 匹配 may_const 字段
- 新: `hasMetadata("ejit.may_const")` — 直接在 load 指令上判断
- GEP 链分析**仍然需要**（计算 byteOffset 用于 `ReadRuntimeValue(baseAddr + byteOffset)`），但不再用于 may_const 查表

**为什么 JIT 端只用 byte offset 做地址计算，不依赖 debug info**:

| 原因 | 说明 |
|------|------|
| bitcode 可能无 debug info | 嵌入式部署通常 strip，DW_TAG 不存在 |
| may_const 正确性由 metadata 保证 | `!ejit.may_const` 直接标注在 load 指令上，无需 offset 匹配 |
| 简单 | `accumulateConstantOffset` + `hasMetadata("ejit.may_const")`，无额外依赖 |
| 性能 | 不解析 debug info tree，JIT 编译路径更快 |

### 3.2 详细伪代码

```cpp
PreservedAnalyses EJitStructFieldPass::run(Module& M, ModuleAnalysisManager& AM) {
    // [BiSheng] context_ 用于日志/断言定位，非地址计算必需
    // v1.4 起地址计算完全由 accumulateConstantOffset 完成，
    // context_ 中的 fnName/periodName/cellIdx 仅用于诊断输出和调试断言
    assert(context_ && "SpecializationContext not set");

    // Step 1: 构建全局变量 → periodName 映射
    GVPeriodMap gvPeriodMap = buildGVPeriodMap(M);

    // Step 2: 收集所有 may_const load 指令的 GEP 链信息
    struct GEPAnalysis {
        GlobalVariable* gv;          // 全局变量
        uint64_t byteOffset;         // 字节偏移，=0 表示标量全局变量直接 load
        LoadInst* load;              // Load 指令 (带 !ejit.may_const metadata)
        int cellIdx = -1;            // 保留字段，不再参与地址计算 (v1.4)
    };

    std::vector<GEPAnalysis> candidates;

    for (Function& F : M.functions()) {
        for (BasicBlock& BB : F) {
            for (Instruction& I : BB) {
                auto* LI = dyn_cast<LoadInst>(&I);
                if (!LI) continue;

                // [BiSheng] v1.2: 先检查 load 上的 !ejit.may_const metadata
                // 这是 may_const 识别的唯一依据，不再依赖 offset 匹配
                if (!LI->hasMetadata("ejit.may_const"))
                    continue;

                Value* PtrOp = LI->getPointerOperand();

                // [BiSheng] v1.3: 处理直接 GlobalVariable load (标量全局变量, 无 GEP)
                if (auto* GV = dyn_cast<GlobalVariable>(PtrOp)) {
                    if (!gvPeriodMap.count(GV)) continue;
                    GEPAnalysis analysis;
                    analysis.gv = GV;
                    analysis.byteOffset = 0;
                    analysis.load = LI;
                    candidates.push_back(analysis);
                    continue;
                }

                auto* GEP = dyn_cast<GEPOperator>(PtrOp);
                if (!GEP)
                    continue;  // PHI/Select → Inline+InstCombine 已消除绝大多数

                // [BiSheng] v1.5: 检查是否通过指针全局变量间接访问
                // GEP 基址可能是一个 LoadInst (加载指针值)
                GlobalVariable* GV = nullptr;
                GEPAnalysis analysis;
                analysis.load = LI;

                if (auto* PtrLoad = dyn_cast<LoadInst>(GEP->getPointerOperand())) {
                    // 间接访问: GEP 的基址来自 load ptr from GV
                    if (resolveIndirectLoad(PtrLoad, GEP, gvPeriodMap, DL, analysis)) {
                        candidates.push_back(analysis);
                        continue;
                    }
                }

                // 直接访问: GEP 操作在全局变量上
                GV = dyn_cast<GlobalVariable>(GEP->getPointerOperand());
                if (!GV && GEP->getNumOperands() > 1)
                    GV = findRootGV(GEP);
                if (!GV) continue;
                if (!gvPeriodMap.count(GV)) continue;

                if (!analyzeGEPChain(GV, GEP, gvPeriodMap, DL, context_, analysis)) {
                    continue;
                }

                candidates.push_back(analysis);
            }
        }
    }

    // Step 3: 批量收集替换信息
    struct LoadReplacement {
        LoadInst* load;
        Constant* constVal;
    };
    std::vector<LoadReplacement> replacements;

    for (auto& info : candidates) {
        // [BiSheng] v1.2: may_const 身份已由 metadata 确认
        // [BiSheng] v1.5: info 包含 indirect/indirectPtrOffset 用于指针间接访问

        RuntimeValue value = readRuntimeValue(info, info.load->getType());
        Constant* constVal = createConstantFromRuntimeValue(info.load->getType(), value);

        replacements.push_back({info.load, constVal});
    }

    // Step 4: 批量替换
    for (auto& repl : replacements) {
        repl.load->replaceAllUsesWith(repl.constVal);
        repl.load->eraseFromParent();
    }

    if (!replacements.empty()) {
        return PreservedAnalyses::none();
    }
    return PreservedAnalyses::all();
}
```

### 3.3 GEP 链分析 (analyzeGEPChain)

> **v1.2**: `structTy` 字段已移除。may_const 身份由 load metadata 确认，无需通过 struct 名查找 LayoutRegistry。GEP 分析仅用于计算运行时地址。

```cpp
bool analyzeGEPChain(GlobalVariable* GV, GEPOperator* GEP,
                     const GVPeriodMap& gvPeriodMap,
                     const DataLayout& DL,
                     SpecializationContext* ctx,
                     GEPAnalysis& result) {
    result.gv = GV;

    // [BiSheng] v1.4: 仅计算 byteOffset 用于运行时地址计算
    // accumulateConstantOffset 隐含验证了所有 GEP 索引为常量
    // （任何非常量索引都会导致它返回 false）
    APInt gepOffset(DL.getIndexSizeInBits(0), 0);
    if (!GEP->accumulateConstantOffset(DL, gepOffset))
        return false;

    result.byteOffset = gepOffset.getZExtValue();
    result.cellIdx = -1;  // 不再从 GEP 提取，需要时从 SpecializationContext 获取
    return true;
}
```

### 3.4 指针间接访问解析 (resolveIndirectLoad) — v1.5 新增

```cpp
// [BiSheng] v1.5: 处理通过指针全局变量间接访问 ejit_may_const 字段
// IR 模式: %ptr = load ptr, ptr @g_pCfg / @g_ptrArr[idx]
//           %field = GEP %S, ptr %ptr, ...
//           %val = load, !ejit.may_const
bool resolveIndirectLoad(LoadInst* PtrLoad, GEPOperator* FieldGEP,
                         const GVPeriodMap& gvPeriodMap,
                         const DataLayout& DL,
                         GEPAnalysis& result) {
    // Step 1: 追溯指针值的来源 GV
    Value* PtrLoadOp = PtrLoad->getPointerOperand();

    GlobalVariable* ptrGV = nullptr;
    uint64_t ptrArrayOffset = 0;  // 指针在 GV 内的偏移 (非数组则为 0)

    if (auto* GV = dyn_cast<GlobalVariable>(PtrLoadOp)) {
        ptrGV = GV;
    } else if (auto* PtrGEP = dyn_cast<GEPOperator>(PtrLoadOp)) {
        ptrGV = dyn_cast<GlobalVariable>(PtrGEP->getPointerOperand());
        if (!ptrGV) return false;
        APInt offset(DL.getIndexSizeInBits(0), 0);
        if (!PtrGEP->accumulateConstantOffset(DL, offset))
            return false;
        ptrArrayOffset = offset.getZExtValue();
    } else {
        return false;
    }

    if (!gvPeriodMap.count(ptrGV)) return false;

    // Step 2: 计算结构体字段偏移
    APInt fieldOffset(DL.getIndexSizeInBits(0), 0);
    if (!FieldGEP->accumulateConstantOffset(DL, fieldOffset))
        return false;

    // Step 3: 填充结果
    result.gv = ptrGV;
    result.byteOffset = fieldOffset.getZExtValue();
    result.indirectPtrOffset = ptrArrayOffset;  // v1.5: 指针值的偏移
    result.indirect = true;
    return true;
}
```

**为什么只有一层间接**: SPEC 约束 `ejit_period` 只标记一级指针。`ejit_may_const` 不支持指针字段。不存在 `g_pCfg->pInner->field` 这样的多层间接。

**GEPAnalysis 扩展** (v1.5):

```cpp
struct GEPAnalysis {
    GlobalVariable* gv;
    uint64_t byteOffset;          // 结构体字段偏移 (accumulateConstantOffset)
    LoadInst* load;
    bool indirect = false;        // v1.5: true = 通过指针 GV 间接访问
    uint64_t indirectPtrOffset;   // v1.5: 指针值在 GV 中的偏移 (非数组 = 0)
};
```

### 3.5 数组索引常量判定（已简化）

> **v1.4**: `getConstantArrayIndex` 已移除。`accumulateConstantOffset` 隐含验证了所有 GEP 索引均为常量（任何非常量索引都会导致它返回 false）。`cellIdx` 不是地址计算所必需的 —— `byteOffset` 已包含数组下标贡献的完整偏移。`cellIdx` 信息对 PASS6 不必要（地址 = baseAddr + byteOffset），仅用于日志/断言，可从 `SpecializationContext` 直接获取。

```
为什么 accumulateConstantOffset 就够了:

  @g_cells[3].inner.data[2]
  → GEP ptr @g_cells, i32 0, i32 3, i32 1, i32 0, i32 2

  accumulateConstantOffset 计算:
    0 * sizeof([16 x CellConfig])  = 0          (first-class type 索引)
  + 3 * sizeof(CellConfig)         = 48         (period 数组下标)
  + offsetof(CellConfig, inner)    = 4
  + 0 * sizeof(Inner)              = 0          (first-class type 索引)
  + offsetof(Inner.data)           = 0
  + 2 * sizeof(int)                = 8
  ─────────────────────────────────────
  byteOffset = 60 → baseAddr + 60 → 正确地址

若任意索引非常量（如循环变量 i）→ accumulateConstantOffset 返回 false → 跳过该 load
```


### 3.6 运行时值读取

> **v1.5**: 新增 `indirect` / `indirectPtrOffset` 参数。当通过指针 GV 间接访问时，先从 GV 读出指针值作为基地址，再加字段偏移。

```cpp
RuntimeValue readRuntimeValue(const GEPAnalysis& info, Type* loadType) {
    void* gvAddr = getGlobalVariableRuntimeAddr(info.gv);
    if (!gvAddr) return RuntimeValue{};

    void* baseAddr;
    if (info.indirect) {
        // v1.5: 间接访问 — 从 GV 读出指针值作为基地址
        // gvAddr + indirectPtrOffset = 指针值在 GV 中的地址
        // *(void**)ptrSlot = 实际的基地址
        uintptr_t ptrSlot = (uintptr_t)gvAddr + info.indirectPtrOffset;
        baseAddr = *(void**)ptrSlot;
        if (!baseAddr) return RuntimeValue{};  // 空指针
    } else {
        baseAddr = gvAddr;
    }

    uintptr_t fieldAddr = (uintptr_t)baseAddr + info.byteOffset;
    return readMemory(fieldAddr, loadType);
}
```

### 3.7 全局变量运行时地址获取

```cpp
void* getGlobalVariableRuntimeAddr(GlobalVariable* GV) {
    // [BiSheng] 全局变量的运行时地址
    // 方法 1: 通过 PeriodArrayRegistry 查询
    //   - PeriodArrayRegistry 在 ejit_auto_register 中填入 baseAddr
    //   - 运行时直接查找
    const PeriodArrayInfo* info = runtimeRegistry_->getArrayInfo(GV->getName());
    if (info) return info->baseAddr;

    // 方法 2: 通过名称从进程符号表查找 (dlsym 等效)
    //   - 全局变量在可执行文件中导出符号
    //   - 适用于 static 时间窗的变量
    void* addr = lookupSymbol(GV->getName());
    if (addr) return addr;

    return nullptr;
}
```

### 3.8 LLVM Constant 创建

> **v1.2**: 移除 `FieldLayout` 参数。类型直接从 `load->getType()` 获取。

```cpp
Constant* createConstantFromRuntimeValue(Type* ty, const RuntimeValue& value) {
    LLVMContext& Ctx = ty->getContext();

    if (ty->isIntegerTy()) {
        unsigned bw = ty->getIntegerBitWidth();
        if (bw <= 32)
            return ConstantInt::get(ty, value.intVal);
        else
            return ConstantInt::get(ty, value.longVal);
    } else if (ty->isFloatTy()) {
        return ConstantFP::get(ty, value.floatVal);
    } else if (ty->isDoubleTy()) {
        return ConstantFP::get(ty, (double)value.floatVal);
    } else if (ty->isPointerTy()) {
        return ConstantExpr::getIntToPtr(
            ConstantInt::get(Type::getInt64Ty(Ctx),
                (uint64_t)(uintptr_t)value.ptrVal),
            ty);
    } else if (ty->isArrayTy()) {
        ArrayType* AT = cast<ArrayType>(ty);
        return createConstantArray(AT, value);
    }

    return nullptr;
}
```

---

## 4. 输出 IR 变化

### 4.1 替换效果

```llvm
; 输入 (JIT 编译时):
@g_cellCfg = dso_local global [16 x %struct.CellConfig] zeroinitializer

define void @process_task(i32 %cellIdx) {
  %elem = getelementptr [16 x %struct.CellConfig], ptr @g_cellCfg, i32 0, i32 3
  %field = getelementptr %struct.CellConfig, ptr %elem, i32 0, i32 0
  %val = load i32, ptr %field                    ; ← load 从进程内存读取

  %cmp = icmp eq i32 %val, 2                     ; ← 使用 g_cellCfg[3].cellType
  br i1 %cmp, label %do_one, label %do_two
}

; 输出 (本 Pass 执行后):
define void @process_task(i32 %cellIdx) {
  ; GEP 链被删除
  ; Load 被替换为运行时常量 (假设 g_cellCfg[3].cellType = 2)
  %cmp = icmp eq i32 2, 2                        ; ← Constant
  br i1 %cmp, label %do_one, label %do_two
}

; 经后续优化 (DCE + CFG 简化) 后:
define void @process_task(i32 %cellIdx) {
  br label %do_one                                ; ← 分支已折叠
}
```

### 4.2 未替换场景 (保持不变)

```llvm
; 场景 1: 非 may_const 字段 → 保留原 load
; CellConfig.xx 不是 ejit_may_const
%field = getelementptr %struct.CellConfig, ptr %elem, i32 0, i32 1
%val = load i32, ptr %field          ; ← 保留不变 (不是 may_const)

; 场景 2: 非 period 全局变量 → 保留原 load
@other_global = dso_local global i32 42
%val = load i32, ptr @other_global   ; ← 保留不变 (不是 period 变量)

; 场景 3: GEP 索引非常量 (v1.4: accumulateConstantOffset 返回 false → 跳过)
; 例如循环变量作为数组下标: g_arr[i].field
; 这不影响正确性，对应循环中的 may_const 访问无法在每个迭代中独立特化
```

---

## 5. 关键数据结构

```cpp
// [BiSheng] JIT 编译时上下文 (由 JIT 编译器传入)
// 注意: 使用 std::string 而非 const char*，防止异步模式下的悬空指针
// (异步编译时，请求可能在线程间传递，原始 C 字符串生命周期不足)
struct SpecializationContext {
    std::string fnName;
    int period_count;                              // 0-4
    struct {
        std::string periodName;                    // "cell", "trp", ...
        int cellIdx;                               // 数组下标值
    } dimensions[4];
    OptimizationLevel optLevel;
};

// [BiSheng] v1.2: StructLayoutInfo 和 LayoutRegistry 已废弃
// may_const 识别由 load->hasMetadata("ejit.may_const") 完成
// 类型信息从 load->getType() 直接获取

// [BiSheng] 全局变量 → period 信息映射 (JIT 时从 metadata 构建)
struct GVPeriodMap {
    struct Info {
        std::string periodName;
        bool isArray;             // true = ejit_period_arr
        bool isStatic;            // true = ejit_period(static)
        StructType* elementType;
        size_t arraySize;
    };
    std::map<GlobalVariable*, Info> mapping;
};

// [BiSheng] 运行时值读取 (联合体)
union RuntimeValue {
    int32_t intVal;
    int64_t longVal;
    float floatVal;
    bool boolVal;
    void* ptrVal;
};
```

---

## 6. 优化等级适配

EJitStructFieldPass 本身不受优化等级影响 — 常量替换是基础操作，在所有等级 (L1/L2/L3) 都执行。优化等级仅控制 PASS6 之后的 Pass 集合。注意 Inline 作为 PASS6 的前置步骤始终执行（非等级控制）。

| 步骤 | L1 | L2 | L3 |
|------|----|----|-----|
| InstCombine | 执行 | 执行 | 执行 |
| Inline | 执行 | 执行 | 执行 |
| **本 Pass** | **执行** | **执行** | **执行** |
| SCCP + DCE | 执行 | 执行 | 执行 |
| 额外 Inline | — | 执行 | 执行 |
| CFGSimplify | — | 执行 | 执行 |
| LoopUnroll | — | — | 执行 |

---

## 7. 错误处理

| 错误场景 | 处理策略 |
|---------|---------|
| load 无 `!ejit.may_const` metadata | 跳过，这是正常路径（非 may_const 字段） | v1.2 |
| 直接 GlobalVariable load | byteOffset = 0，正常处理 | v1.3 |
| 指针 GV 间接访问 | 多追溯一层，从 GV 读指针值后再算地址 | v1.5 |
| GEP 链非标准模式 | 跳过该 load，不做替换 |
| 指针操作数为 PHI/Select | 跳过 — Inline+InstCombine 已消除绝大多数，残余 fallback 到 AOT | v1.3 |
| 间接访问的指针值为 null | 跳过，记录 warning | v1.5 |
| 运行时地址读取失败 | 跳过该 load，记录 warning 日志 |
| cellIdx 越界 | 跳过该 load（运行时检查在 ejit_compile_or_get 之前完成） |
| GEP accumulateConstantOffset 失败 | 跳过该 load |
| 类型不匹配 | 跳过该 load，记录 warning |
| `!ejit.may_const` metadata 被 AOT 优化丢弃 | 不再发生 — bitcode 在标准优化前提取 | v1.3 |

**核心原则**: 替换失败 → 跳过 → 后续优化仍可进行 → 函数可正确 fallback 到 AOT 行为。

---

## 8. 性能考量

### 8.1 批量处理

```
优化: 先收集所有候选 load 指令，再批量替换
原因: 
  - 避免在遍历 instructions 时修改 IR（迭代器失效）
  - 允许先验证全部候选，再统一替换
```

### 8.2 内存读取

```
方式: 直接从进程内存读取 (memcpy / dereference)
耗时: 纳秒级 (L1 cache hit)
频率: 每个 ejit_may_const 字段一次读取
总计: 通常 < 10 个字段 / 函数 → < 100ns
```

### 8.3 GEP 偏移计算（v1.2: 仅用于地址计算）

```
方式: 使用 LLVM DataLayout 的 accumulateConstantOffset
用途: 计算运行时内存地址 (baseAddr + byteOffset)，不再用于 may_const 识别
优点: 处理所有合法的 GEP 链（含多层嵌套），无需手动推导
```

---

## 9. 测试策略

### 9.1 单元测试 (Unit Test)

```cpp
// EJitStructFieldTest.cpp (v1.2: 使用 load metadata 识别)
// TEST(EJitStructField, ReplaceByMetadata)
//   构造 IR: %v = load i32, ptr %field, !ejit.may_const !{}
//   hasMetadata("ejit.may_const") → true
//   设置内存值 → 42
//   验证: load 被替换为 ConstantInt(42)

// TEST(EJitStructField, SkipNoMetadata)
//   构造 IR: %v = load i32, ptr %field  (无 !ejit.may_const metadata)
//   hasMetadata("ejit.may_const") → false
//   验证: load 保留不变

// TEST(EJitStructField, MetadataSurvivesSROA)
//   构造 IR: may_const 字段的 load 带 !ejit.may_const
//   运行 SROA → 验证 split 后的 load 仍带 metadata
//   进而验证: PASS6 仍可替换

// TEST(EJitStructField, SkipNonPeriodGV)
//   构造 IR: @normal_gv → load i32 (即使有 !ejit.may_const metadata)
//   全局变量无 ejit.metadata → 无法确定 period
//   验证: load 保留不变 (正常路径)

// TEST(EJitStructField, SkipNonConstantIndex)
//   构造 IR: @g_arr[%var].field → load i32, !ejit.may_const !{}
//   %var 非常量 → accumulateConstantOffset 返回 false → 跳过
//   验证: load 保留不变

// TEST(EJitStructField, SkipVolatileField)
//   构造 IR: load volatile i32, ptr %field
//   isMayConst=true, isVolatile=true
//   验证: load 保留不变

// TEST(EJitStructField, HandleNestedStruct)
//   构造 IR: @g_var.inner.nestedField → load
//   验证: 嵌套字段正确识别并替换

// TEST(EJitStructField, HandleMultiDim)
//   构造 IR: @g_cell[1].field + @g_trp[5].field
//   两个不同的 period_arr_ind 均已常量化
//   验证: 两个 load 均正确替换

// TEST(EJitStructField, HandleIndirectPointer) — v1.5
//   构造 IR: %ptr = load ptr, ptr @g_pCfg; GEP %S, ptr %ptr, ...
//   设置 &g_pCfg 处内存值为 0x1000，0x1000+offsetof(field) = 42
//   验证: load 被替换为 ConstantInt(42)

// TEST(EJitStructField, HandleIndirectPointerArray) — v1.5
//   构造 IR: %slot = GEP @g_ptrArr, i32 0, i32 3; %ptr = load ptr, ptr %slot; ...
//   设置 g_ptrArr[3] 处指针指向的字段值
//   验证: load 被替换为运行时常量

// TEST(EJitStructField, SkipNullPtr) — v1.5
//   构造 IR: %ptr = load ptr, ptr @g_pCfg (%ptr = null)
//   验证: load 保留不变，记录 warning
```

### 9.2 集成测试

```
1. 编译 C 源文件 (含 ejit_entry + ejit_period_arr + ejit_may_const)
2. AOT 编译 + bitcode 嵌入
3. 运行时: ejit_init → ejit_activate → 调用函数
4. 触发 JIT 编译 (含本 Pass)
5. 验证特化函数执行结果与 AOT 版本一致
6. 变更运行时值 → deactivate → activate → 验证新值生效
```

---

## 10. 实施注意事项

1. **Pipeline 顺序**: 本 Pass 运行顺序为 JIT Pipeline 第 4 步：参数替换 → InstCombine → Inline → **本 Pass**。Inline 先于本 Pass 执行，保证了跨函数的 may_const load 已在入口函数体内展开。

2. **直接 GlobalVariable load (v1.3)**: 标量全局变量（如 `@g_flag`）直接被 load 时没有 GEP。`dyn_cast<GlobalVariable>(LI->getPointerOperand())` 命中时 byteOffset = 0，直接从 GV 运行时地址读取。

3. **PHI 指针跳过 (v1.3)**: 若 `LI->getPointerOperand()` 是 PHI/Select 节点，直接跳过。前置的 InstCombine + GVN 已消除了 incoming values 相同的 PHI。残余的 PHI 场景 fallback 到 AOT，不修改正确性。

4. **GEP 偏移计算**: 优先使用 `GEPOperator::accumulateConstantOffset(DL, offset)` 而非手动解析索引。此 API 处理了所有 GEP 的合法形式（单层、多层、合并等）。依赖 InstCombine 已将常量链折叠。

5. **v1.2 类型映射简化**: may_const 识别不再需要 struct 名 → LayoutRegistry 映射。类型信息直接从 `load->getType()` 获取。

6. **运行时地址获取**: 全局变量的运行时地址可通过:
   - PeriodArrayRegistry（ejit_init 时注册）
   - 进程符号查找（`dlsym` / `GetProcAddress` 等效）
   - 符号优先（所有 ejit_period/ejit_period_arr 全局变量在可执行文件中为导出符号）

7. **内存屏障**: 读取进程内存中的全局变量值时，在 JIT 线程中通常不需要内存屏障（因 activate/deactivate 的同步已由 runtime 保证）。但如果是在异步编译模式下，需确保读取到最新值。

8. **递归结构体**: 若结构体中包含自身类型的指针 (`struct Node { Node* next; }`)，不会标记 ejit_may_const 在指针字段上，Pass 无需处理。

9. **ABI 兼容**: `accumulateConstantOffset` 使用 DataLayout 计算偏移，与 AOT 编译时的布局一致。确保 JIT 编译时使用的 DataLayout 与 AOT 相同。

10. **v1.3 metadata 持久性**: 由于 bitcode 在标准优化前提取（见 PASS1 v1.1），`!ejit.may_const` metadata 在 bitcode Module 中完整保留，不再有被 AOT 优化丢弃的风险。

---

*文档版本: 1.5*
*创建日期: 2026-04-26*
*最后更新: 2026-04-29*
