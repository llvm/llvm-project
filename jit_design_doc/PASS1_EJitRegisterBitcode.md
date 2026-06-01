# EJitRegisterBitcodePass 设计文档

**版本**: 1.2
**日期**: 2026-05-14
**关联**: SPEC4.md, PLAN4.md
**类型**: AOT Module Pass
**顺序**: 早期 AOT Pipeline — 标准优化之前

---

## 1. 概述

EJitRegisterBitcodePass 负责从编译单元中提取所有 `ejit_entry` 标记的函数及其传递依赖（调用的内部函数），序列化为 LLVM Bitcode，并将 Bitcode 数据嵌入到最终可执行文件的独立段中。在序列化前，对提取的 bitcode 运行轻量优化管线（`preOptimizeBitcode`），减少 JIT 编译时的开销和内存压力。运行时 JIT 编译器按需加载这些 Bitcode 进行特化编译。

### 1.1 核心职责

- 识别所有带 `!ejit.metadata` 且包含 `!{"ejit_entry"}` 子节点的函数
- 构建函数依赖图的传递闭包（被这些函数调用的所有内部函数）
- 将选中函数及必要符号提取到独立的 LLVM Module
- **对提取的 bitcode 运行 AOT 预优化**（§1.4）
- **自动扫描外部符号并生成 `ejit_register_symbol` 调用**（§1.3）
- 序列化为 Bitcode 字节数组
- 在原始 Module 中创建全局变量存储 Bitcode 数据
- 生成运行时注册调用，将 Bitcode 指针/大小注册到运行时

### 1.2 设计约束

| 约束项 | 说明 |
|--------|------|
| 外部函数 | 不提取外部函数（external linkage），仅收集声明 |
| 递归 | 已由 Sema 检查排除，Pass 层面不再验证 |
| 函数指针 | 不做函数指针分析，不特化间接调用目标 |
| Bitcode 版本 | 必须与运行时 LLVM 版本匹配 |
| Pipeline 位置 | **必须在标准优化（O2/O3）之前执行** —— 见 §1.3 |

### 1.3 为什么必须在标准优化之前执行

PASS1 必须在 O2/O3 标准优化之前运行，原因：

1. **Wapper 必须在 bitcode 提取前**：PASS3 (WrapperGen) 插入 `ejit_compile_or_get` 调用和 wrapper 块。如果 PASS1 提取含 wrapper 的 bitcode，JIT 会编译 wrapper 自身导致无限递归。
2. **提取闭包**：PASS1 需要在优化前识别完整的函数闭包（内联前），以保证所有被调用的辅助函数都包含在 bitcode 中。

`!ejit.may_const` metadata 的保留通过两个机制保证：
- **LLVM 层面**：`MD_ejit_may_const` 注册为固定 metadata kind，`copyMetadataForLoad()` 在 switch 中显式保留，覆盖 InstCombine、SROA、Mem2Reg 等所有使用该函数的 Pass。
- **GV-level 元数据回退**：Clang CodeGen 在 GV 的 `!ejit.metadata` 中输出 `ejit_may_const_field` 字节偏移。`preOptimizeBitcode` 中 `reAnnotateMayConst` 步骤从 GV metadata 恢复被 Pass 新建 load 丢失的 per-load metadata。

### 1.4 AOT 预优化 (preOptimizeBitcode)

提取 bitcode 后、序列化前，对 bitcode 模块运行轻量优化管线，减少 JIT 编译时的开销和内存压力：

```
AlwaysInline → ModuleInliner(O2) → Mem2Reg → EarlyCSE+InstCombine → SimplifyCFG → reAnnotateMayConst
```

| 步骤 | 说明 |
|------|------|
| AlwaysInline | 展开 `__attribute__((always_inline))` 函数 |
| ModuleInliner(O2) | cost-based 内联小函数（Index 封装函数等） |
| Mem2Reg | alloca → SSA，消除内存访问 |
| EarlyCSE + InstCombine | 公共子表达式消除 + 常量折叠 |
| SimplifyCFG | 扁平化分支、合并基本块 |
| reAnnotateMayConst | 从 GV metadata 恢复可能丢失的 `!ejit.may_const` |

> **NDEBUG 守卫**：debug 构建（共享库）跳过预优化，避免 `LLVMPasses ↔ LLVMEmbeddedJIT` 循环链接依赖。同一套 Pass 在 JIT 管线中运行。

---

## 2. 输入 IR 格式

### 2.1 函数 Metadata

```llvm
; ejit_entry 函数
define void @process_task(i32 %idx) #0 {
  ; ...
}

; 函数关联的 metadata (由 Clang CodeGen 生成)
; !ejit.metadata = distinct !{!0}
; !0 = !{!"ejit_entry"}

; 含 ejit_period_arr_ind 参数的 ejit_entry 函数
define void @process_multi(i32 %cellIdx, i32 %trpIdx) #0 {
  ; ...
}
; metadata 同上 (ejit_entry 仍然是 !{!"ejit_entry"})
```

### 2.2 从 metadata 推断依赖

ejit_entry 函数本身在 IR 中不含 period 依赖的直接声明。依赖通过函数参数上的 `ejit_period_arr_ind` 属性的 debug info / metadata 推导，但 Bitcode 提取不需要此信息。Bitcode 提取阶段仅关注函数调用图。

---

## 3. 核心算法

### 3.1 主流程

```
输入: Module M (AOT 编译后的 IR)
输出: Module M (插入 bitcode 全局变量和注册调用)

步骤:
1. CollectEntryFunctions(M) → 收集所有 ejit_entry 函数
2. ComputeTransitiveClosure(entryFuncs) → 计算依赖函数集
3. ExtractModule(fullFuncSet) → 提取独立的 bitcode Module
4. SerializeToBitcode(extractedModule) → 序列化为字节数组
5. EmbedBitcodeInModule(M, bitcodeBytes) → 在原始 Module 中创建全局变量
6. GenerateRegisterCall(M, globalVar) → 插入注册函数调用
```

### 3.2 详细伪代码

```cpp
PreservedAnalyses EJitRegisterBitcodePass::run(Module& M, ModuleAnalysisManager& AM) {
    // Step 1: 收集所有 ejit_entry 函数
    std::vector<Function*> entryFuncs;
    for (Function& F : M.functions()) {
        if (isEjitEntry(F))
            entryFuncs.push_back(&F);
    }

    if (entryFuncs.empty())
        return PreservedAnalyses::all();

    // Step 2: 计算传递闭包
    std::set<Function*> fullSet;
    for (Function* F : entryFuncs) {
        collectTransitiveCalls(F, fullSet);
    }

    // Step 3: 提取 Module
    // 使用 ValueMapper + 过滤实现: 只保留 fullSet 中的函数和它们引用的全局变量
    // 方法: clone fullSet 中的函数到新 Module, 同时克隆:
    //   - 被引用的全局变量 (ejit_period, ejit_period_arr)
    //   - 被引用的结构体类型声明 (用于 JIT 阶段的类型推导)
    //   - ejit.metadata 命名元数据
    Module* bitcodeM = extractModule(M, fullSet);

    // Step 4: 序列化
    std::string bitcodeStr = serializeToBitcode(*bitcodeM);

    // Step 5: 嵌入
    // 创建常量字节数组
    Constant* bitcodeConst = createByteArrayConstant(M, bitcodeStr);

    // 创建全局变量 (放入独立段 ".ejit.bitcode")
    GlobalVariable* GV = new GlobalVariable(
        M, bitcodeConst->getType(), true,  // isConstant = true
        GlobalValue::InternalLinkage,
        bitcodeConst, "__ejit_bitcode");

    // Step 6: 注册
    // 在 ejit_auto_register 函数中插入调用:
    //   ejit_register_bitcode("func_name", bitcode_ptr, bitcode_size);
    insertBitcodeRegisterCall(M, GV, entryFuncs);

    // 清理: 删除 entryFuncs 的 ejit_entry metadata (防止被后续 Pass 重复处理)
    for (Function* F : entryFuncs) {
        // 保留 metadata 供其他 Pass 使用 (EJitWrapperGenPass 需要)
        // 仅标记已处理
    }

    delete bitcodeM;
    return PreservedAnalyses::none();
}
```

### 3.3 函数依赖图构建

```cpp
void collectTransitiveCalls(Function* F, std::set<Function*>& visited) {
    if (!visited.insert(F).second)
        return;  // 已访问

    for (BasicBlock& BB : *F) {
        for (Instruction& I : BB) {
            if (CallInst* CI = dyn_cast<CallInst>(&I)) {
                Function* callee = CI->getCalledFunction();
                if (callee && !callee->isDeclaration()) {
                    collectTransitiveCalls(callee, visited);
                }
            }
        }
    }
}
```

### 3.4 Module 提取策略

```cpp
Module* extractModule(Module& SrcM, const std::set<Function*>& funcSet) {
    Module* DstM = new Module("ejit_bitcode", SrcM.getContext());
    // 复制 DataLayout 和 TargetTriple
    DstM->setDataLayout(SrcM.getDataLayoutStr());
    DstM->setTargetTriple(SrcM.getTargetTriple());

    // 1. 声明所有被调用的外部函数 (isDeclaration)
    std::set<Function*> externDecls;
    for (Function* F : funcSet) {
        for (BasicBlock& BB : *F) {
            for (Instruction& I : BB) {
                if (CallInst* CI = dyn_cast<CallInst>(&I)) {
                    Function* callee = CI->getCalledFunction();
                    if (callee && callee->isDeclaration() && !callee->isIntrinsic()) {
                        externDecls.insert(callee);
                    }
                }
            }
        }
    }

    // 2. 收集被引用的全局变量 (ejit_period, ejit_period_arr)
    std::set<GlobalVariable*> referencedGVs;
    for (Function* F : funcSet) {
        for (BasicBlock& BB : *F) {
            for (Instruction& I : BB) {
                for (Value* op : I.operands()) {
                    if (GlobalVariable* GV = dyn_cast<GlobalVariable>(op)) {
                        referencedGVs.insert(GV);
                    }
                }
            }
        }
    }

    // 3. 使用 CloneFunctionInto 克隆函数
    ValueToValueMapTy VMap;
    for (Function* F : funcSet) {
        // 先声明外部函数
        for (Function* ext : externDecls) {
            if (!DstM->getFunction(ext->getName())) {
                Function::Create(ext->getFunctionType(),
                                 Function::ExternalLinkage,
                                 ext->getName(), DstM);
            }
        }

        // 克隆全局变量声明
        for (GlobalVariable* GV : referencedGVs) {
            if (!DstM->getNamedGlobal(GV->getName())) {
                new GlobalVariable(*DstM, GV->getValueType(),
                                   GV->isConstant(), GV->getLinkage(),
                                   nullptr, GV->getName());
            }
        }

        // 克隆函数体
        Function* newF = Function::Create(F->getFunctionType(),
                                          Function::InternalLinkage,
                                          F->getName(), DstM);
        // 保留 ejit.metadata
        if (MDNode* MD = F->getMetadata("ejit.metadata")) {
            // 需要映射 metadata (使用 MapMetadata)
        }

        // 映射参数
        for (auto I = F->arg_begin(), J = newF->arg_begin();
             I != F->arg_end(); ++I, ++J) {
            VMap[&*I] = &*J;
            J->setName(I->getName());
        }

        SmallVector<ReturnInst*, 4> Returns;
        CloneFunctionInto(newF, F, VMap,
                          CloneFunctionChangeType::DifferentModule,
                          Returns);
    }

    return DstM;
}
```

### 3.5 Bitcode 嵌入格式

```llvm
; 生成的 LLVM IR
@__ejit_bitcode = internal constant [4096 x i8] c"\xDE\xC0\x17\x0B...", section ".ejit.bitcode", align 1

; 字节数组长度 = bitcode 序列化后的字节数
; section 属性确保链接器将其放入独立段，便于运行时查找
```

### 3.6 注册函数生成

```cpp
void insertBitcodeRegisterCall(Module& M, GlobalVariable* bitcodeGV,
                                const std::vector<Function*>& entryFuncs) {
    // 查找或创建 ejit_auto_register 初始化函数
    Function* registerFn = M.getFunction("ejit_auto_register");
    if (!registerFn) {
        auto* FT = FunctionType::get(Type::getVoidTy(M.getContext()), false);
        registerFn = Function::Create(FT, Function::InternalLinkage,
                                      "ejit_auto_register", &M);
        // 添加到 llvm.global_ctors (优先级 65535, 最低)
        appendToGlobalCtors(M, registerFn, 65535);
    }

    // 获取运行时函数声明
    Function* ejitRegBitcode = M.getFunction("ejit_register_bitcode");
    if (!ejitRegBitcode) {
        auto* FT = FunctionType::get(Type::getVoidTy(M.getContext()),
            { PointerType::getUnqual(M.getContext()),   // func_name: char*
              PointerType::getUnqual(M.getContext()),    // bitcode_ptr: void*
              Type::getInt64Ty(M.getContext()) }, false); // bitcode_size: uint64_t
        ejitRegBitcode = Function::Create(FT, Function::ExternalLinkage,
                                          "ejit_register_bitcode", &M);
    }

    IRBuilder<> Builder(M.getContext());
    BasicBlock* BB = BasicBlock::Create(M.getContext(), "entry", registerFn);
    Builder.SetInsertPoint(BB);

    for (Function* F : entryFuncs) {
        // 每个 ejit_entry 函数注册一次，但指向同一份 bitcode 数据
        // 运行时 BitcodeTracker 维护 funcName → bitcode 映射
        // JIT 编译时按 funcName 从 bitcode Module 中定位目标函数
        Constant* nameStr = Builder.CreateGlobalStringPtr(F->getName());

        // bitcode 指针 (从全局变量首地址)
        Value* bcPtr = Builder.CreatePointerCast(bitcodeGV,
            PointerType::getUnqual(M.getContext()));

        // bitcode 大小
        Value* bcSize = ConstantInt::get(Type::getInt64Ty(M.getContext()),
            bitcodeSize);

        Builder.CreateCall(ejitRegBitcode, {nameStr, bcPtr, bcSize});
    }

    Builder.CreateRetVoid();
}
```

---

## 4. 输出 IR 变化

### 4.1 新增结构

```llvm
; 1. Bitcode 数据段
@__ejit_bitcode = internal constant [N x i8] <serialized bitcode bytes>, section ".ejit.bitcode", align 1

; 2. 运行时注册调用 (在 llvm.global_ctors 中注册)
define internal void @ejit_auto_register() {
entry:
    call void @ejit_register_bitcode(i8* getelementptr(...) @".str.process_task",
                                     i8* bitcast ([N x i8]* @__ejit_bitcode to i8*),
                                     i64 <bitcode_size>)
    ; ... 每个 ejit_entry 函数一条注册调用
    ret void
}

@llvm.global_ctors = appending global [
    ...,
    { i32, void ()*, i8* } { i32 65535, void ()* @ejit_auto_register, i8* null }
]
```

### 4.2 保持不变

- ejit_entry 函数的函数体（由后续 EJitWrapperGenPass 修改）
- ejit.metadata（后续 Pass 仍需读取）
- 其他函数和全局变量

---

## 5. 关键数据结构

```cpp
// Bitcode 提取结果
struct BitcodeExtractResult {
    GlobalVariable* bitcodeVar;         // 嵌入的 bitcode 全局变量
    std::vector<Function*> entryFuncs;  // 所有 ejit_entry 函数列表
    size_t bitcodeSize;                 // bitcode 字节数
};

// 运行时注册 API (在 libejit 中实现)
// void ejit_register_bitcode(const char* funcName, void* bitcodePtr, size_t bitcodeSize);
//   - funcName: ejit_entry 函数名
//   - bitcodePtr: bitcode 数据指针
//   - bitcodeSize: bitcode 数据大小
// 运行时维护 funcName → bitcode 的映射表，JIT 编译时按需解析
```

---

## 6. 错误处理

| 错误场景 | 处理策略 |
|---------|---------|
| 无 ejit_entry 函数 | 直接返回 PreservedAnalyses::all()，不做任何修改 |
| Bitcode 序列化失败 | report_fatal_error — 这是 LLVM 内部错误，不应发生 |
| 空依赖函数集 | 仍然生成 bitcode（仅含入口函数声明） |
| 外部函数调用 | 仅声明不克隆，运行时 JIT 编译时这些符号从进程加载 |

---

## 7. 与其他 Pass 的交互

本 Pass 是独立的早期 Pass，不包含在 `EJitAotModulePass` 内。

```
EJitRegisterBitcodePass  (早期: 标准优化前, 提取原始 bitcode)
        ↓
[标准优化 Pipeline: O2/O3]
        ↓
EJitRegisterPeriodPass   (晚期: 时间窗变量注册)
        ↓
EJitWrapperGenPass       (晚期: Wrapper 生成)
        ↓
EJitPeriodHandlerPass    (晚期: 生命周期处理)
```

| 后续组件 | 依赖本 Pass 提供 | 说明 |
|---------|-----------------|------|
| EJitRegisterPeriodPass | 无直接依赖 | 独立读取 M 中的 metadata（优化后可能部分丢失，但不影响 period 变量识别） |
| EJitWrapperGenPass | ejit.metadata 保留 | 函数级 metadata 通常不会被优化 Pass 丢弃 |
| EJitPeriodHandlerPass | ejit.metadata 保留 | 同上 |
| Runtime EJIT | `@__ejit_bitcode` + `ejit_auto_register` | 运行时加载 metadata 完整的原始 bitcode |

**关键点**：

- `ejit.metadata`（函数/全局变量级别）是 Named Metadata，LLVM 优化 Pass 通常不会移除，因此晚期 Pass 仍可正常读取
- `!ejit.may_const`（load 指令级别）是 instruction metadata，容易被优化 Pass 丢弃。这是本 Pass 必须提前执行的核心原因
- `ejit_period_arr_ind` 参数信息已编码在函数级 `ejit.metadata` 中（通过 `!{!"ejit_period_arr_ind", !"periodName", i32 argIdx}`），不依赖参数上的 debug info，因此可被晚期 Pass 正常读取

---

## 8. 测试策略

### 8.1 Lit 测试 (test/Transforms/EmbeddedJIT/)

```llvm
; test_register_bitcode.ll
; RUN: opt -passes=ejit-register-bitcode -S %s | FileCheck %s

; CHECK: @__ejit_bitcode = internal constant
; CHECK: section ".ejit.bitcode"
; CHECK: define internal void @ejit_auto_register
; CHECK: call void @ejit_register_bitcode

; 验证:
; 1. 无 ejit_entry 函数的 Module 不做修改
; 2. ejit_entry 函数及其调用链被正确提取
; 3. bitcode 全局变量正确创建
; 4. 注册函数正确生成
; 5. 外部函数声明被保留
```

### 8.2 验证点

| 验证项 | 方法 |
|--------|------|
| Bitcode 序列化正确性 | 将嵌入的 bitcode 重新反序列化，验证函数列表一致 |
| 传递闭包完整性 | 检查调用链中所有内部函数都在提取结果中 |
| 注册调用参数正确性 | FileCheck 匹配函数名、指针、大小 |
| 无 ejit_entry 的 Module | FileCheck: not check |

---

## 9. 实施注意事项

1. **CloneFunctionChangeType**: 必须使用 `DifferentModule`（非 `LocalChangesOnly`），因为目标函数克隆到独立的新 Module 中。`LocalChangesOnly` 要求源和目标在同一 Module 内，否则会导致 VMap 中的全局变量引用指向源 Module 的 GV 而非目标 Module 的 GV，引发运行时 crash。

2. **Metadata 映射**: `CloneFunctionInto` 默认不映射命名 metadata。需要手动遍历源 Module 的 metadata 并调用 `MapMetadata` 映射到目标 Module。使用 `DifferentModule` 时，`CloneFunctionInto` 会通过 VMap 自动映射函数级 metadata（如 `!ejit.metadata`），但 Module 级 named metadata 需要手动处理。

3. **结构体类型**: 依赖函数可能引用 `%struct.CellConfig` 等命名结构体类型。提取 Module 时需要保留这些类型的声明。可以从引用全局变量的类型中自动收集。

4. **llvm.used**: 被提取的全局变量可能在 `@llvm.used` 中，提取时需注意。

5. **与 easyJIT 的关系**: easyJIT 的 `RegisterBitcodePass` 可作为参考实现，但 EmbeddedJIT 的提取条件不同（基于 metadata 而非参数属性）。

6. **多 Module 支持**: 若链接时优化 (LTO) 启用，多个编译单元合并为一个 Module，此时所有 ejit_entry 函数在一个 Module 中，一次提取即可。若为普通编译（每个 .c 文件独立 Module），则每个 Module 有独立的 bitcode 段，运行时按需加载。

7. **多 ejit_entry 函数的 Bitcode 合并**: 当前设计将所有 ejit_entry 函数（含传递闭包）放入一个 `@__ejit_bitcode` 全局变量。运行时通过 `ejit_register_bitcode(funcName, ptr, size)` 为每个函数名注册同一份 bitcode。JIT 编译时，从 bitcode 中按函数名定位目标函数，仅编译该函数及其依赖。

8. **自动符号注册 (v1.6)**: `ejit_auto_register` 函数中除了 `ejit_register_bitcode` 调用外，还自动生成 `ejit_register_symbol` 调用。PASS1 扫描闭包中所有外部函数调用（`isDeclaration() && !isIntrinsic()`）和全局变量引用，为每个唯一符号生成注册代码。这使 JIT 在裸核环境无需 dlsym 即可解析外部符号。符号地址暂存于 `EJitRegistrationStore`，在 `ejit_init` 时消费。

9. **常量全局变量保留 (v1.6)**: `collectReferencedGlobals` 现在包含所有被引用的全局变量（含常量）。`extractAndSerialize` 在转换为外部声明时跳过了常量（`GV.isConstant()`），避免版本字符串等编译器生成的常量在 JIT 链接时出现 "Symbols not found" 错误。这些常量作为定义保留在 bitcode 中。

---

*文档版本: 1.0*
*创建日期: 2026-04-26*

10. **静态注册表生成 (v1.7)**: 裸核环境无 `llvm.global_ctors`，PASS1 同时生成全局常量数组 `__ejit_registry_bitcode[]` 作为 fallback 路径：

```
五项结构体 { i32 type, ptr name1, ptr name2, ptr data, i64 size }
```

- Bitcode 条目: `{EJIT_REG_BITCODE(0), funcName, NULL, bitcodePtr, size}`
- 符号条目: `{EJIT_REG_SYMBOL(3), symName, NULL, addr, 0}`
- 末尾 sentinel: `{EJIT_REG_NONE(4), NULL, NULL, NULL, 0}`

`ejit_init()` 中若 `EJitRegistrationStore` 为空（裸核无构造器）或 `forceStaticRegistry=true`，
则遍历此数组完成注册。

---

*文档版本: 1.1*  
*创建日期: 2026-04-26*  
*更新日期: 2026-06-01*
