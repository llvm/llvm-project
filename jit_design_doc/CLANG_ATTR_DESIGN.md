# EmbeddedJIT Clang 前端 Attribute 设计文档

**版本**: 1.0
**日期**: 2026-05-03
**关联**: SPEC4.md §2, PLAN4.md §4.4, PLAN4.md 阶段 2
**类型**: Clang 前端 Attribute 实现方案

---

## 1. 概述

为 EmbeddedJIT 系统在 Clang 前端实现 6 种属性：`ejit_may_const`、`ejit_period`、`ejit_period_arr`、`ejit_period_arr_ind`、`ejit_entry`、`ejit_period_lc`。覆盖属性 TableGen 定义、Sema 语义检查、AST 分析（防呆检测）和 LLVM IR CodeGen 全流程。

### 1.1 参考文档

| 文档 | 内容 |
|------|------|
| SPEC4.md §2 | 用户标注接口需求规格 |
| PLAN4.md §4.4 | Clang 属性接口设计（伪代码） |
| PLAN4.md §4.4.5 | 属性约束检查表 |
| PLAN4.md §4.4.7 | 防呆设计实现方案 |

### 1.2 属性总览

| 属性 | 标记对象 | 参数 | 继承性 | 说明 |
|------|---------|------|--------|------|
| `ejit_may_const` | FieldDecl | 无 | Inheritable | 标记结构体成员在时间窗内可视为常量 |
| `ejit_period(name)` | VarDecl (标量) | 1 个 String | Inheritable | 标记全局变量所属时间窗 |
| `ejit_period_arr(name)` | VarDecl (数组) | 1 个 String | Inheritable | 标记全局数组所属时间窗数组 |
| `ejit_period_arr_ind(name)` | ParmVarDecl | 1 个 String | InheritableParam | 标记函数参数为特化维度 |
| `ejit_entry` | FunctionDecl | 无 | Inheritable | 标记函数为 JIT 优化入口 |
| `ejit_period_lc(name)` | FunctionDecl | 1 个 String | Inheritable | 标记函数为时间窗生命周期管理函数 |

---

## 2. Attribute TableGen 定义 (Attr.td)

### 2.1 现有 Attr.td 语法规则

在 LLVM/Clang 中，属性使用 TableGen 定义在 `clang/include/clang/Basic/Attr.td` 中。关键语法要素：

- **Spellings**: `[Clang<"attr_name">]` 定义 `__attribute__((attr_name))` 拼写
- **Subjects**: `SubjectList<[TargetType]>` 约束属性可标记的 AST 节点
- **Args**: 属性参数（必须与 C++ Attr 类中的 getter 顺序一致）
- **let SemaHandler = 0**: 标记此属性不由 TableGen 自动生成的 Sema 处理（需手写 handler）
- **let SimpleHandler = 1**: 标记使用内置的简单 Sema handler（不做额外语义检查）
- **let HasCustomParsing = 1**: 标记需要自定义参数解析
- **let Documentation**: 指向 AttrDocs.td 中的文档定义

### 2.2 6 种属性定义

以下定义应添加到 `clang/include/clang/Basic/Attr.td` 中，建议放在 EmbeddedJIT 相关注释块内。

#### 2.2.1 ejit_may_const

```tablegen
// EmbeddedJIT: 标记结构体成员在时间窗内可视为常量
def EjitMayConst : InheritableAttr {
  let Spellings = [Clang<"ejit_may_const">];
  let Subjects = SubjectList<[Field]>;
  let Documentation = [EjitMayConstDocs];
  let SimpleHandler = 1;
}
```

**设计要点**：
- `InheritableAttr`：保证属性在 Redeclaration 链上传递
- `Subjects = [Field]`：限制只能标记在结构体/联合体字段上
- `SimpleHandler = 1`：TableGen 自动生成 `handleEjitMayConstAttr`，只做 Subject 匹配检查。额外语义检查（类型、volatile）在 CodeGen 阶段做软降级

#### 2.2.2 ejit_period

```tablegen
// EmbeddedJIT: 标记全局变量所属的时间窗
def EjitPeriod : InheritableAttr {
  let Spellings = [Clang<"ejit_period">];
  let Args = [StringArgument<"PeriodName", 1>];
  let Subjects = SubjectList<[Var]>;
  let Documentation = [EjitPeriodDocs];
  let HasCustomParsing = 1;
}
```

**设计要点**：
- `StringArgument<"PeriodName", 1>`：字符串参数，如 `"static"` `"cell"`
- `HasCustomParsing = 1`：需要自定义 Sema handler（检查是否数组、是否重复标记等）
- `SubjectList<[Var]>`：限制标记在 VarDecl 上

#### 2.2.3 ejit_period_arr

```tablegen
// EmbeddedJIT: 标记全局数组所属的时间窗数组
def EjitPeriodArr : InheritableAttr {
  let Spellings = [Clang<"ejit_period_arr">];
  let Args = [StringArgument<"PeriodName", 1>];
  let Subjects = SubjectList<[Var]>;
  let Documentation = [EjitPeriodArrDocs];
  let HasCustomParsing = 1;
}
```

**设计要点**：
- 与 `ejit_period` 结构一致，但 Sema handler 中检查是否为数组类型
- `HasCustomParsing = 1`：自定义 handler 中检查数组长度 < 100

#### 2.2.4 ejit_period_arr_ind

```tablegen
// EmbeddedJIT: 标记 JIT 特化维度参数，关联对应时间窗数组
def EjitPeriodArrInd : InheritableParamAttr {
  let Spellings = [Clang<"ejit_period_arr_ind">];
  let Args = [StringArgument<"PeriodName", 1>];
  let Subjects = SubjectList<[ParmVar]>;
  let Documentation = [EjitPeriodArrIndDocs];
  let HasCustomParsing = 1;
}
```

**设计要点**：
- `InheritableParamAttr`：参数属性可被子声明继承
- `SubjectList<[ParmVar]>`：限制标记在函数参数上
- `HasCustomParsing = 1`：Sema handler 检查参数为整数类型、最多 4 个等

#### 2.2.5 ejit_entry

```tablegen
// EmbeddedJIT: 标记函数将进行 JIT 优化
def EjitEntry : InheritableAttr {
  let Spellings = [Clang<"ejit_entry">];
  let Subjects = SubjectList<[Function]>;
  let Documentation = [EjitEntryDocs];
  let HasCustomParsing = 1;
}
```

**设计要点**：
- `SubjectList<[Function]>`：限制标记在函数声明上
- `HasCustomParsing = 1`：Sema handler 检查不支持递归

#### 2.2.6 ejit_period_lc

```tablegen
// EmbeddedJIT: 标记时间窗生命周期管理函数
def EjitPeriodLc : InheritableAttr {
  let Spellings = [Clang<"ejit_period_lc">];
  let Args = [StringArgument<"PeriodName", 1>];
  let Subjects = SubjectList<[Function]>;
  let Documentation = [EjitPeriodLcDocs];
  let HasCustomParsing = 1;
}
```

**设计要点**：
- `HasCustomParsing = 1`：Sema handler 检查是否有对应 `ejit_period_arr_ind` 参数
- 支持函数标记多个 `ejit_period_lc`（同名属性可重复出现）

### 2.3 属性拼写生成

根据以上定义，TableGen 自动生成以下 C/C++ 拼写：

```c
// 1. ejit_may_const - 无参属性
__attribute__((ejit_may_const))

// 2. ejit_period(static) - 单字符串参数
__attribute__((ejit_period("static")))

// 3. ejit_period_arr(cell) - 单字符串参数
__attribute__((ejit_period_arr("cell")))

// 4. ejit_period_arr_ind(cell) - 单字符串参数
__attribute__((ejit_period_arr_ind("cell")))

// 5. ejit_entry - 无参属性
__attribute__((ejit_entry))

// 6. ejit_period_lc(cell) - 单字符串参数
__attribute__((ejit_period_lc("cell")))
```

用户常用的宏定义封装（不在编译器实现范围内，由用户或 SDK 头文件提供）：

```c
#define ejit_may_const          __attribute__((ejit_may_const))
#define ejit_period(x)          __attribute__((ejit_period(x)))
#define ejit_period_arr(x)      __attribute__((ejit_period_arr(x)))
#define ejit_period_arr_ind(x)  __attribute__((ejit_period_arr_ind(x)))
#define ejit_entry              __attribute__((ejit_entry))
#define ejit_period_lc(x)       __attribute__((ejit_period_lc(x)))
```

---

## 3. 诊断消息定义 (DiagnosticSemaKinds.td)

在 `clang/include/clang/Basic/DiagnosticSemaKinds.td` 中添加以下诊断消息：

```tablegen
// === EmbeddedJIT 诊断消息 ===

// [错误] 全局变量归属冲突
def err_ejit_period_conflict : Error<
  "variable %0 cannot have multiple ejit_period or ejit_period_arr attributes">;

// [错误] ejit_period 只能用于标量全局变量，不能用于数组
def err_ejit_period_not_array : Error<
  "ejit_period attribute cannot be used on array variable %0; "
  "use ejit_period_arr for arrays">;

// [错误] 数组变量必须使用 ejit_period_arr，不能使用 ejit_period
def err_ejit_period_arr_not_scalar : Error<
  "ejit_period_arr attribute requires an array type; "
  "%0 is not an array">;

// [错误] ejit_period_arr 数组大小超过限制
def err_ejit_period_arr_too_large : Error<
  "ejit_period_arr array %0 has size %1, which exceeds the maximum of 100">;

// [错误] ejit_period_arr_ind 参数类型必须为整数
def err_ejit_period_arr_ind_invalid_type : Error<
  "ejit_period_arr_ind parameter %0 must have integer type">;

// [错误] ejit_period_arr_ind 参数维度过多
def err_ejit_period_arr_ind_too_many : Error<
  "function %0 has %1 ejit_period_arr_ind parameters, "
  "which exceeds the maximum of 4">;

// [错误] ejit_entry 函数不支持递归
def err_ejit_entry_recursive : Error<
  "ejit_entry function %0 cannot be recursive">;

// [错误] ejit_period_lc 缺少对应的 ejit_period_arr_ind 参数
def err_ejit_period_lc_no_index : Error<
  "ejit_period_lc(%0) requires a corresponding ejit_period_arr_ind(%0) parameter">;

// [警告] 修改 ejit_may_const 字段但未标记 ejit_period_lc
def warn_ejit_may_const_modified_without_lc : Warning<
  "modifying ejit_may_const field %0 of %1 without ejit_period_lc attribute">,
  InGroup<EmbeddedJIT>;

// [警告] 函数引用 period_arr 但未通过 ejit_period_arr_ind 声明依赖
def warn_ejit_undeclared_period_dep : Warning<
  "function %0 references ejit_period_arr '%1' but does not declare "
  "a dependency on it via ejit_period_arr_ind">,
  InGroup<EmbeddedJIT>;

// [备注] period_arr 定义位置
def note_ejit_period_arr_defined_here : Note<
  "ejit_period_arr '%0' defined here">;

// [备注] ejit_entry 定义位置
def note_ejit_entry_defined_here : Note<
  "ejit_entry function %0 defined here">;
```

**Warning Group 注册**：在 `clang/include/clang/Basic/DiagnosticGroups.td` 中添加：

```tablegen
def EmbeddedJIT : DiagGroup<"embedded-jit">;
```

---

## 4. Sema 语义处理

### 4.1 实现策略

存在两种方案：

| 方案 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| A | 直接在 `SemaDeclAttr.cpp` 中添加 handler | 无需新建文件 | 增加已 8350 行巨型文件 |
| B | 新建 `SemaEJIT.cpp` | 逻辑隔离，可维护性好 | 需要修改 CMakeLists.txt |

**推荐方案 B**：新建 `clang/lib/Sema/SemaEJIT.cpp`，将 6 种属性的 Sema handler 和防呆检测逻辑集中管理。

### 4.2 文件修改清单

| 文件 | 操作 | 内容 |
|------|------|------|
| `clang/lib/Sema/SemaDeclAttr.cpp` | 修改 | 在 switch-case dispatch 中添加 6 个属性分支，转发到新增 handler |
| `clang/lib/Sema/SemaEJIT.cpp` | **新建** | 6 个 handle 函数 + SemaCheck 函数 |
| `clang/lib/Sema/CMakeLists.txt` | 修改 | 添加 `SemaEJIT.cpp` |

### 4.3 Handlers 详细实现

#### 4.3.1 handleEjitMayConstAttr

```cpp
// SemaEJIT.cpp

/// handleEjitMayConstAttr - 处理 ejit_may_const 属性
/// 语义检查:
///   1. 仅可标记在 FieldDecl 上 (由 TableGen Subjects 保证)
///   2. 检查字段类型: 仅支持整型、布尔型、浮点型、嵌套结构体
///   3. volatile 字段警告 (不阻止编译，JIT 时跳过即可)
static void handleEjitMayConstAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  auto *FD = cast<FieldDecl>(D);
  QualType FT = FD->getType();

  // 检查类型: 整型、布尔型、浮点型、嵌套结构体/结构体数组
  if (!FT->isIntegerType() && !FT->isBooleanType() &&
      !FT->isFloatingType() && !FT->isStructureOrClassType() &&
      !FT->isArrayType()) {
    S.Diag(AL.getLoc(), diag::warn_attribute_wrong_decl_type)
        << AL << ExpectedField;
    return;
  }

  // volatile 字段不视为常量 — 仅在 CodeGen 时做软降级（不加 metadata），
  // 此处不报错也不警告，仅在 Ewarn 级别可选的开发诊断
  if (FT.isVolatileQualified()) {
    // 静默: volatile 字段的 load 不会带 !ejit.may_const metadata，
    // JIT 时自然跳过
  }

  D->addAttr(::new (S.Context) EjitMayConstAttr(S.Context, AL));
}
```

#### 4.3.2 handleEjitPeriodAttr

```cpp
/// handleEjitPeriodAttr - 处理 ejit_period(name) 属性
/// 语义检查:
///   1. 仅可标记在 VarDecl (非数组) 上
///   2. 检查是否有重复的 period/period_arr 标记
///   3. 提取 periodName
static void handleEjitPeriodAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  auto *VD = cast<VarDecl>(D);

  // 检查 1: 必须是全局变量（file scope 或 namespace scope）
  if (!VD->hasGlobalStorage()) {
    S.Diag(AL.getLoc(), diag::err_attribute_wrong_decl_type)
        << AL << ExpectedGlobalVar;
    return;
  }

  // 检查 2: 不能是数组（数组应使用 ejit_period_arr）
  if (VD->getType()->isArrayType()) {
    S.Diag(AL.getLoc(), diag::err_ejit_period_not_array) << VD;
    return;
  }

  // 检查 3: 变量归属冲突 — 不能同时有多个 ejit_period 或 ejit_period_arr
  if (VD->hasAttr<EjitPeriodAttr>() || VD->hasAttr<EjitPeriodArrAttr>()) {
    S.Diag(AL.getLoc(), diag::err_ejit_period_conflict) << VD;
    return;
  }

  // 解析 periodName
  StringRef periodName;
  if (!S.checkStringLiteralArgumentAttr(AL, 0, periodName))
    return;

  // 对于非 static 的自定义标量时间窗，当前版本标记为未来扩展
  // （不阻止编译，但 CodeGen 发出警告或备注）
  if (periodName != "static") {
    // 自定义 ejit_period(name) 为未来扩展，当前仅支持 "static"
    // 此处不报错，仅在后端处理时记录
  }

  VD->addAttr(::new (S.Context) EjitPeriodAttr(S.Context, AL, periodName));
}
```

#### 4.3.3 handleEjitPeriodArrAttr

```cpp
/// handleEjitPeriodArrAttr - 处理 ejit_period_arr(name) 属性
/// 语义检查:
///   1. 仅可标记在数组 VarDecl 上
///   2. 数组长度固定且 < 100
///   3. 检查归属冲突
static void handleEjitPeriodArrAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  auto *VD = cast<VarDecl>(D);

  // 检查 1: 必须是全局变量
  if (!VD->hasGlobalStorage()) {
    S.Diag(AL.getLoc(), diag::err_attribute_wrong_decl_type)
        << AL << ExpectedGlobalVar;
    return;
  }

  // 检查 2: 必须是数组类型
  const ArrayType *AT = S.Context.getAsArrayType(VD->getType());
  if (!AT) {
    S.Diag(AL.getLoc(), diag::err_ejit_period_arr_not_scalar) << VD;
    return;
  }

  // 检查 3: 数组大小 < 100
  if (const ConstantArrayType *CAT = dyn_cast<ConstantArrayType>(AT)) {
    uint64_t size = CAT->getSize().getZExtValue();
    if (size > 100) {
      S.Diag(AL.getLoc(), diag::err_ejit_period_arr_too_large)
          << VD << (unsigned)size;
      return;
    }
  } else {
    // 非常量数组大小（VLA 等）→ 报错
    S.Diag(AL.getLoc(), diag::err_ejit_period_arr_not_scalar) << VD;
    return;
  }

  // 检查 4: 归属冲突
  if (VD->hasAttr<EjitPeriodAttr>() || VD->hasAttr<EjitPeriodArrAttr>()) {
    S.Diag(AL.getLoc(), diag::err_ejit_period_conflict) << VD;
    return;
  }

  // 解析 periodName
  StringRef periodName;
  if (!S.checkStringLiteralArgumentAttr(AL, 0, periodName))
    return;

  VD->addAttr(::new (S.Context) EjitPeriodArrAttr(S.Context, AL, periodName));
}
```

#### 4.3.4 handleEjitPeriodArrIndAttr

```cpp
/// handleEjitPeriodArrIndAttr - 处理 ejit_period_arr_ind(name) 属性
/// 语义检查:
///   1. 仅可标记在 ParmVarDecl 上
///   2. 参数类型必须为整数类型
///   3. 所在函数最多 4 个 ejit_period_arr_ind 参数
static void handleEjitPeriodArrIndAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  auto *PVD = cast<ParmVarDecl>(D);

  // 检查 1: 参数类型必须为整数类型
  QualType PT = PVD->getType();
  if (!PT->isIntegerType()) {
    S.Diag(AL.getLoc(), diag::err_ejit_period_arr_ind_invalid_type) << PVD;
    return;
  }

  // 检查 2: 所在函数最多 4 个 ejit_period_arr_ind 参数
  if (auto *FD = dyn_cast<FunctionDecl>(PVD->getDeclContext())) {
    unsigned count = 0;
    for (auto *P : FD->parameters()) {
      if (P->hasAttr<EjitPeriodArrIndAttr>())
        count++;
    }
    if (count >= 4) {
      S.Diag(AL.getLoc(), diag::err_ejit_period_arr_ind_too_many)
          << FD << (count + 1);
      return;
    }
  }

  StringRef periodName;
  if (!S.checkStringLiteralArgumentAttr(AL, 0, periodName))
    return;

  PVD->addAttr(::new (S.Context)
      EjitPeriodArrIndAttr(S.Context, AL, periodName));
}
```

#### 4.3.5 handleEjitEntryAttr

```cpp
/// handleEjitEntryAttr - 处理 ejit_entry 属性
/// 语义检查:
///   1. 仅可标记在 FunctionDecl 上 (由 Subjects 保证)
///   2. 不支持递归函数
static void handleEjitEntryAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  auto *FD = cast<FunctionDecl>(D);

  // 检查 1: 必须是函数定义（非仅声明）
  if (!FD->isThisDeclarationADefinition()) {
    // 仅声明不报错，CodeGen 时再处理
  }

  // 检查 2: 递归检测
  if (FD->isRecursive()) {
    S.Diag(AL.getLoc(), diag::err_ejit_entry_recursive) << FD;
    return;
  }

  D->addAttr(::new (S.Context) EjitEntryAttr(S.Context, AL));
}
```

#### 4.3.6 handleEjitPeriodLcAttr

```cpp
/// handleEjitPeriodLcAttr - 处理 ejit_period_lc(name) 属性
/// 语义检查:
///   1. 仅可标记在 FunctionDecl 上
///   2. 必须有对应的 ejit_period_arr_ind(name) 参数
static void handleEjitPeriodLcAttr(Sema &S, Decl *D, const ParsedAttr &AL) {
  auto *FD = cast<FunctionDecl>(D);

  StringRef periodName;
  if (!S.checkStringLiteralArgumentAttr(AL, 0, periodName))
    return;

  // 检查是否有对应的 ejit_period_arr_ind(name) 参数
  bool hasMatchingIdx = false;
  for (auto *P : FD->parameters()) {
    if (auto *IdxAttr = P->getAttr<EjitPeriodArrIndAttr>()) {
      if (IdxAttr->getPeriodName() == periodName) {
        hasMatchingIdx = true;
        break;
      }
    }
  }

  if (!hasMatchingIdx) {
    S.Diag(AL.getLoc(), diag::err_ejit_period_lc_no_index) << periodName;
    return;
  }

  D->addAttr(::new (S.Context) EjitPeriodLcAttr(S.Context, AL, periodName));
}
```

### 4.4 在 SemaDeclAttr.cpp 中集成

在 `ProcessDeclAttribute()` 函数的 switch-case dispatch 中添加：

```cpp
// EmbeddedJIT attributes — 转发到 SemaEJIT 中的 handler
case ParsedAttr::AT_EjitMayConst:
  handleEjitMayConstAttr(S, D, AL);
  break;
case ParsedAttr::AT_EjitPeriod:
  handleEjitPeriodAttr(S, D, AL);
  break;
case ParsedAttr::AT_EjitPeriodArr:
  handleEjitPeriodArrAttr(S, D, AL);
  break;
case ParsedAttr::AT_EjitPeriodArrInd:
  handleEjitPeriodArrIndAttr(S, D, AL);
  break;
case ParsedAttr::AT_EjitEntry:
  handleEjitEntryAttr(S, D, AL);
  break;
case ParsedAttr::AT_EjitPeriodLc:
  handleEjitPeriodLcAttr(S, D, AL);
  break;
```

同时在 `SemaDeclAttr.cpp` 头部添加函数声明或在 `Sema.h` 中声明：

```cpp
// clang/lib/Sema/SemaEJIT.cpp 中的函数声明
void handleEjitMayConstAttr(Sema &S, Decl *D, const ParsedAttr &AL);
void handleEjitPeriodAttr(Sema &S, Decl *D, const ParsedAttr &AL);
void handleEjitPeriodArrAttr(Sema &S, Decl *D, const ParsedAttr &AL);
void handleEjitPeriodArrIndAttr(Sema &S, Decl *D, const ParsedAttr &AL);
void handleEjitEntryAttr(Sema &S, Decl *D, const ParsedAttr &AL);
void handleEjitPeriodLcAttr(Sema &S, Decl *D, const ParsedAttr &AL);
```

### 4.5 防呆检测：变量归属冲突

归属冲突检查在 `handleEjitPeriodAttr` 和 `handleEjitPeriodArrAttr` 中内联处理。每个 handler 在添加属性前检查 `hasAttr<EjitPeriodAttr>()` 和 `hasAttr<EjitPeriodArrAttr>()`。

相同变量同时被两个不同 period 标记的错误用例由 Clang 的属性唯一约束自然覆盖——同种属性不能在同一个 Decl 上出现多次（除非声明 `let Inherited = 0` 并手动允许多次数）。对 `ejit_period_lc` 需要支持多次出现（函数可管理多个时间窗）：

```tablegen
// ejit_period_lc 允许多次标记（每标记一个时间窗名称）
// 可通过设置 let Inherited = 0 或在 AdditionalMembers 中特殊处理
```

但默认情况下 `InheritableAttr` 的同种属性在 Clang 中可多次出现（例如 `annotate` 属性）。需要考虑两种模型：

- **Model 1 多 attribute 实例**: 允许同一函数多次标记 `ejit_period_lc(cell)` 和 `ejit_period_lc(trp)`——两个独立的属性实例
- **Model 2 单 attribute 实例**: 仅允许一次标记

对于 `ejit_period_lc`，采用 **Model 1**（允许多实例）。Clang 的 `InheritableAttr` 默认允许同类属性多次出现在不同拼写下，对于相同拼写需要验证。若存在限制，可在 TableGen 中添加：

```tablegen
let Inherited = 0; // 暂不需要，Clang InheritableAttr 默认可多次出现
```

---

## 5. CodeGen IR 生成

### 5.1 实现策略

新建 `clang/lib/CodeGen/CGEJIT.cpp`，包含：

1. `emitEJITMetadata(Module &M)` — 在模块 emit 完成后，遍历所有已收集的 EJIT 元数据，生成 Named Metadata
2. Load 指令的 `!ejit.may_const` metadata annotation（嵌入在 `CGExpr.cpp` 中）

### 5.2 IR Metadata 格式

根据 PASS1–PASS6 设计文档中的 IR 格式约定，生成以下 metadata：

#### 5.2.1 函数级 metadata: `!ejit.metadata`

每种 EJIT 属性在**函数级别**附加一个统一的 `!ejit.metadata` Named Metadata Node。示例：

```llvm
; ejit_entry 函数（无 period_arr_ind 参数，仅依赖 static）
; !ejit.metadata = distinct !{!0}
; !0 = !{!"ejit_entry"}

; ejit_entry 函数（单 period_arr_ind 参数）
; !ejit.metadata = distinct !{!0, !1}
; !0 = !{!"ejit_entry"}
; !1 = !{!"ejit_period_arr_ind", !"cell", i32 0}

; ejit_entry 函数（无参数，不依赖 static 之外的时间窗）
; !ejit.metadata = distinct !{!0}
; !0 = !{!"ejit_entry"}
```

#### 5.2.2 全局变量 metadata: `!ejit.metadata`

```llvm
; @g_boardCfg = ... | !ejit.metadata = distinct !{!0}
; !0 = !{!"ejit_period", !"static"}

; @g_cellCfg = ... | !ejit.metadata = distinct !{!1}
; !1 = !{!"ejit_period_arr", !"cell", i32 16}
```

#### 5.2.3 Load metadata: `!ejit.may_const`

```llvm
; 非 volatile 的 ejit_may_const 字段 load → 标注 !ejit.may_const
%v = load i32, ptr %field, !ejit.may_const !{}
```

`!{}` 是一个空的 metadata node，仅作为标记位——PASS6 通过 `hasMetadata("ejit.may_const")` 检查。

### 5.3 CodeGenFunction 中的 Load Metadata 注入

#### 5.3.1 修改 CGExpr.cpp — EmitLoadOfScalar / EmitLoadOfLValue

在 `CGExpr.cpp` 中，load 指令生成位置（约第 1700–2100 行之间），对于 `ejit_may_const` 字段的 load 添加 metadata：

```cpp
// 在 EmitLoadOfScalar 或同等位置
LValue LV = ...;
llvm::LoadInst *Load = Builder.CreateLoad(ConvertTypeForMem(E->getType()),
                                          Addr, ...);
// 现有: MD_tbaa, MD_noundef, MD_range, MD_nonnull 等

// === EmbeddedJIT: 注入 !ejit.may_const metadata ===
if (LV.getBaseInfo() && E->getType().hasMayConstAttr()) {
  // 仅对非 volatile 的 may_const 字段 load 加 metadata
  llvm::MDNode *MayConstMD = llvm::MDNode::get(getLLVMContext(), {});
  Load->setMetadata("ejit.may_const", MayConstMD);
}
```

**判断逻辑**：需要从 `MemberExpr` 回溯到 `FieldDecl`，检查是否有 `EjitMayConstAttr`。由于 Clang 的 `FieldDecl` 上已挂载属性，CodeGen 可通过以下方式检查：

```cpp
// 辅助函数: 检查 Expr 的基是否为 ejit_may_const 字段
static bool isEjitMayConstLoad(const Expr *E) {
  if (const auto *ME = dyn_cast<MemberExpr>(E->IgnoreParens())) {
    if (const auto *FD = dyn_cast<FieldDecl>(ME->getMemberDecl())) {
      return FD->hasAttr<EjitMayConstAttr>() &&
             !FD->getType().isVolatileQualified();
    }
  }
  // 处理数组下标 + 成员访问组合: g_cells[idx].field
  if (const auto *ASE = dyn_cast<ArraySubscriptExpr>(E->IgnoreParens())) {
    // 递归检查 base
  }
  return false;
}
```

更准确的方案是在 `LValue` 对象中传递 may_const 标记。在 `LValue` 类中添加：

```cpp
// clang/lib/CodeGen/LValue.h (或类似位置)
class LValue {
  // ...
  bool isEjitMayConst_ = false;
public:
  void setEjitMayConst(bool v) { isEjitMayConst_ = v; }
  bool isEjitMayConst() const { return isEjitMayConst_; }
};
```

在构建 LValue 时（`EmitLValueForField` 等位置），检查 FieldDecl 是否有 `EjitMayConstAttr`，设置标记。

#### 5.3.2 修改 CodeGenModule.cpp — 函数/全局变量 metadata 生成

在 `EmitGlobalFunctionDefinition` 和 `EmitGlobalVarDefinition` 完成后，调用对应的 metadata 生成函数。

```cpp
// CGEJIT.cpp 中的 metadata 生成函数

namespace clang {
namespace CodeGen {

/// 为 ejit_entry / ejit_period_lc 函数生成 !ejit.metadata
void emitEjitFunctionMetadata(CodeGenModule &CGM,
                              const FunctionDecl *FD,
                              llvm::Function *F) {
  llvm::LLVMContext &Ctx = CGM.getLLVMContext();
  SmallVector<llvm::Metadata *, 8> entries;

  // 收集 metadata 条目
  if (FD->hasAttr<EjitEntryAttr>()) {
    entries.push_back(llvm::MDNode::get(Ctx,
        llvm::MDString::get(Ctx, "ejit_entry")));
  }

  for (const auto *LCA : FD->specific_attrs<EjitPeriodLcAttr>()) {
    entries.push_back(llvm::MDNode::get(Ctx, {
        llvm::MDString::get(Ctx, "ejit_period_lc"),
        llvm::MDString::get(Ctx, LCA->getPeriodName())
    }));
  }

  // 遍历参数，收集 ejit_period_arr_ind
  for (unsigned i = 0; i < FD->getNumParams(); ++i) {
    const ParmVarDecl *PD = FD->getParamDecl(i);
    if (const auto *IdxAttr = PD->getAttr<EjitPeriodArrIndAttr>()) {
      entries.push_back(llvm::MDNode::get(Ctx, {
          llvm::MDString::get(Ctx, "ejit_period_arr_ind"),
          llvm::MDString::get(Ctx, IdxAttr->getPeriodName()),
          llvm::ConstantAsMetadata::get(
              llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx), i))
      }));
    }
  }

  if (!entries.empty()) {
    llvm::MDNode *MD = llvm::MDNode::getDistinct(Ctx, entries);
    F->setMetadata("ejit.metadata", MD);
  }
}

/// 为 ejit_period / ejit_period_arr 全局变量生成 !ejit.metadata
void emitEjitGlobalMetadata(CodeGenModule &CGM,
                            const VarDecl *VD,
                            llvm::GlobalVariable *GV) {
  llvm::LLVMContext &Ctx = CGM.getLLVMContext();
  SmallVector<llvm::Metadata *, 2> entries;

  if (const auto *PA = VD->getAttr<EjitPeriodAttr>()) {
    entries.push_back(llvm::MDNode::get(Ctx, {
        llvm::MDString::get(Ctx, "ejit_period"),
        llvm::MDString::get(Ctx, PA->getPeriodName())
    }));
  }

  if (const auto *PAA = VD->getAttr<EjitPeriodArrAttr>()) {
    uint64_t size = 0;
    if (const ConstantArrayType *CAT =
            CGM.getContext().getAsConstantArrayType(VD->getType())) {
      size = CAT->getSize().getZExtValue();
    }
    entries.push_back(llvm::MDNode::get(Ctx, {
        llvm::MDString::get(Ctx, "ejit_period_arr"),
        llvm::MDString::get(Ctx, PAA->getPeriodName()),
        llvm::ConstantAsMetadata::get(
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx), size))
    }));
  }

  if (!entries.empty()) {
    llvm::MDNode *MD = llvm::MDNode::getDistinct(Ctx, entries);
    GV->setMetadata("ejit.metadata", MD);
  }
}

} // namespace CodeGen
} // namespace clang
```

### 5.4 CodeGen 调用点

在 CodeGenModule 的 `EmitGlobalFunctionDefinition` 函数体末尾（函数对象创建后），插入：

```cpp
// === EmbeddedJIT: 函数 metadata 生成 ===
if (FD->hasAttr<EjitEntryAttr>() || FD->hasAttr<EjitPeriodLcAttr>()) {
  emitEjitFunctionMetadata(*this, FD, F);
}
```

在 CodeGenModule 的 `EmitGlobalVarDefinition` 函数体末尾（全局变量创建后），插入：

```cpp
// === EmbeddedJIT: 全局变量 metadata 生成 ===
if (VD->hasAttr<EjitPeriodAttr>() || VD->hasAttr<EjitPeriodArrAttr>()) {
  emitEjitGlobalMetadata(*this, VD, GV);
}
```

### 5.5 外部函数声明生成

`ejit_auto_register` 函数是 AOT 编译时由 Pass（PASS1/PASS2）生成的，不是 Clang CodeGen 的责任。但 CodeGen 需要确保 `ejit_period_arr_ind` metadata 被正确生成，以便晚期 AOT Pass (PASS3) 读取。

因此 Clang CodeGen 不直接生成 `ejit_auto_register` 或 `ejit_register_*` 调用。只负责：

1. ✅ 生成 `!ejit.metadata`（函数级 + 全局变量级）
2. ✅ 生成 `!ejit.may_const`（load 指令级）
3. ❌ 不生成 `ejit_auto_register`（PASS1/PASS2 的职责）
4. ❌ 不生成 `@__ejit_bitcode`（PASS1 的职责）

---

## 6. 防呆检测：写 may_const 字段检测

### 6.1 检测目标

在 Sema 阶段检测：对 `ejit_may_const` 字段的写操作是否在 `ejit_period_lc` 标记的函数中进行。

### 6.2 实现方案

在 `SemaChecking.cpp`（或 `SemaEJIT.cpp`）中的 `CheckAssignment`/`CheckBinaryOperator` 阶段插入检测：

```cpp
/// SemaEJIT.cpp

/// checkEjitMayConstWrite - 检查对 may_const 字段的写操作
/// 若写入操作不在 ejit_period_lc 函数中，发出 warning
void Sema::checkEjitMayConstWrite(Expr *LHS, SourceLocation Loc) {
  // 1. 从 LHS 解析出被写的 FieldDecl
  const FieldDecl *FD = getFieldFromExpr(LHS);
  if (!FD || !FD->hasAttr<EjitMayConstAttr>())
    return;

  // 2. 获取当前所在的函数
  FunctionDecl *CurFn = dyn_cast<FunctionDecl>(CurContext);
  if (!CurFn)
    return;

  // 3. 检查当前函数是否标记了 ejit_period_lc
  bool hasLc = CurFn->hasAttr<EjitPeriodLcAttr>();

  if (!hasLc) {
    Diag(Loc, diag::warn_ejit_may_const_modified_without_lc)
        << FD << FD->getParent();
  }
}

/// 从赋值表达式获取被写的 FieldDecl
static const FieldDecl *getFieldFromExpr(Expr *E) {
  E = E->IgnoreParenCasts();
  if (auto *ME = dyn_cast<MemberExpr>(E)) {
    return dyn_cast<FieldDecl>(ME->getMemberDecl());
  }
  if (auto *ASE = dyn_cast<ArraySubscriptExpr>(E)) {
    // g_cells[idx].field = ... → 递归查 base
    return getFieldFromExpr(ASE->getBase());
  }
  return nullptr;
}
```

**调用点**：需要在 Clang Sema 检查赋值操作的位置调用。例如在 `Sema::CheckAssignmentConstraints` 或 `Sema::CheckBinOp` 中插入。具体的 hook 位置可以是：

- 在 `Sema::ActOnBinOp` 中检测 `BO_Assign` 的 LHS
- 在 `Sema::CheckCompoundAssignment` 中检测 LHS
- 在 `Sema::ActOnUnaryOp` 中检测 `UO_PreInc`/`UO_PostInc` 等

**限制**：
- 仅对直接字段访问 (`obj.field = ...`) 有效
- 通过指针间接修改 (`ptr->field = ...`) 需要更复杂的数据流分析，当前版本暂不追踪
- 这是 **Warning** 级别，不阻止编译

---

## 7. Attribute 文档定义 (AttrDocs.td)

在 `clang/include/clang/Basic/AttrDocs.td` 中添加 6 个文档定义：

```tablegen
def EjitMayConstDocs : Documentation {
  let Category = DocCatVariable;
  let Content = [{
EmbeddedJIT: 标记结构体成员在时间窗内可视为常量。

当全局变量属于某个时间窗且时间窗处于激活状态时，标记了 ``ejit_may_const``
的成员可被视为编译期常量，JIT 编译器会将其替换为运行时实际值并执行常量传播、
死代码消除和分支折叠等优化。

支持的字段类型：整型、布尔型、浮点型、嵌套结构体。``volatile`` 字段不视为常量。

.. code-block:: c

  struct Sample {
      __attribute__((ejit_may_const)) uint32_t a;
      uint32_t xx;
  };
  }];
}

def EjitPeriodDocs : Documentation {
  let Category = DocCatVariable;
  let Content = [{
EmbeddedJIT: 定义全局变量所属的时间窗。

内置时间窗 ``"static"`` 代表在 JIT 运行期不变的全局变量，永远处于生效状态，
无需调用 ``ejit_activate``。

自定义时间窗名称（非 ``"static"``）为未来扩展，表示变量在运行期可能变化、
但在业务逻辑保证的时间段内保持不变。

仅用于标记非数组的全局变量。

.. code-block:: c

  __attribute__((ejit_period("static"))) struct BoardConfig g_boardCfg;
  }];
}

def EjitPeriodArrDocs : Documentation {
  let Category = DocCatVariable;
  let Content = [{
EmbeddedJIT: 定义全局数组所属的时间窗数组。

时间窗数组管理具有相同业务概念但状态独立的多个实例，每个实例可独立控制
时间窗状态。多个不同数组可使用相同的名称。

约束：数组长度固定且小于 100。

.. code-block:: c

  __attribute__((ejit_period_arr("cell"))) struct CellConfig g_cellCfg[16];
  }];
}

def EjitPeriodArrIndDocs : Documentation {
  let Category = DocCatFunction;
  let Content = [{
EmbeddedJIT: 标记函数参数为特化维度参数。

标记此属性的参数被识别为指定时间窗数组的下标，JIT 编译时根据该参数的实际值
确定特化哪个时间窗实例。

参数类型必须为整数类型。单个函数最多支持 4 个标记此属性的参数。

.. code-block:: c

  __attribute__((ejit_entry))
  void process_task(__attribute__((ejit_period_arr_ind("cell"))) uint8_t cellIdx);
  }];
}

def EjitEntryDocs : Documentation {
  let Category = DocCatFunction;
  let Content = [{
EmbeddedJIT: 标记函数将进行 JIT 优化。

编译器将在函数入口插入 JIT 编译触发代码，为该函数生成包含必要符号的 IR
并嵌入二进制文件。JIT 成功时调用特化版本，失败时执行原 AOT 代码。

不支持递归函数。

.. code-block:: c

  __attribute__((ejit_entry))
  void my_jit_function(int param);
  }];
}

def EjitPeriodLcDocs : Documentation {
  let Category = DocCatFunction;
  let Content = [{
EmbeddedJIT: 标记函数为时间窗生命周期管理函数。

标记此属性的函数表明会修改指定时间窗内的数据。编译器将在函数入口插入
``ejit_deactivate``、函数出口插入 ``ejit_activate``，确保数据一致性。

必须与 ``ejit_period_arr_ind`` 配合使用。

.. code-block:: c

  __attribute__((ejit_period_lc("cell")))
  void update_config(__attribute__((ejit_period_arr_ind("cell"))) uint8_t idx);
  }];
}
```

---

## 8. 文件清单

### 8.1 需要修改的现有文件

| 文件 | 修改内容 |
|------|---------|
| `clang/include/clang/Basic/Attr.td` | 添加 6 个属性 TableGen 定义 |
| `clang/include/clang/Basic/AttrDocs.td` | 添加 6 个文档定义 |
| `clang/include/clang/Basic/DiagnosticSemaKinds.td` | 添加 8 个诊断消息 |
| `clang/include/clang/Basic/DiagnosticGroups.td` | 添加 `EmbeddedJIT` 警告组 |
| `clang/lib/Sema/SemaDeclAttr.cpp` | 在 switch-case dispatch 中添加 6 个分支 |
| `clang/lib/Sema/CMakeLists.txt` | 添加 `SemaEJIT.cpp` |
| `clang/lib/CodeGen/CGExpr.cpp` | 在 load 指令生成时注入 `!ejit.may_const` metadata |
| `clang/lib/CodeGen/CodeGenModule.cpp` | 在函数/全局变量 emit 后调用 metadata 生成函数 |
| `clang/lib/CodeGen/CMakeLists.txt` | 添加 `CGEJIT.cpp` |

### 8.2 需要新建的文件

| 文件 | 内容 |
|------|------|
| `clang/lib/Sema/SemaEJIT.cpp` | 6 个属性 handler + may_const 写检测 |
| `clang/lib/CodeGen/CGEJIT.cpp` | metadata 生成: `emitEjitFunctionMetadata`, `emitEjitGlobalMetadata` |
| `clang/test/Sema/ext_attr_ejit.cpp` | Sema 语义分析测试 |
| `clang/test/CodeGen/ejit_metadata.c` | CodeGen LLVM IR 输出测试 |

---

## 9. 测试策略

### 9.1 Sema 测试 (`clang/test/Sema/ext_attr_ejit.cpp`)

```cpp
// RUN: %clang_cc1 -fsyntax-only -verify %s

// 正确用法 - 应该无诊断
struct CellConfig {
  __attribute__((ejit_may_const)) int cellType;   // ok
  int xx;                                          // ok
};

__attribute__((ejit_period("static"))) struct CellConfig g_board;  // ok
__attribute__((ejit_period_arr("cell"))) struct CellConfig g_cells[16]; // ok

__attribute__((ejit_entry))
void process_task(__attribute__((ejit_period_arr_ind("cell"))) int idx);  // ok

// 错误用法 - 应有诊断
__attribute__((ejit_period("cell"))) int g_bad_array[10];  // expected-error {{ejit_period attribute cannot be used on array variable}}
__attribute__((ejit_period_arr("cell"))) int g_not_array;   // expected-error {{ejit_period_arr attribute requires an array type}}

// 归属冲突
__attribute__((ejit_period("one"))) // expected-error {{variable g_conflict cannot have multiple}}
__attribute__((ejit_period("two"))) int g_conflict;

// ejit_entry 递归
__attribute__((ejit_entry)) // expected-error {{ejit_entry function 'recursive_func' cannot be recursive}}
void recursive_func() { recursive_func(); }

// ejit_period_lc 无对应 ind
__attribute__((ejit_period_lc("cell"))) // expected-error {{requires a corresponding ejit_period_arr_ind}}
void bad_lc(int x);
```

### 9.2 CodeGen 测试 (`clang/test/CodeGen/ejit_metadata.c`)

```c
// RUN: %clang_cc1 -S -emit-llvm -o - %s | FileCheck %s

struct CellConfig {
  __attribute__((ejit_may_const)) int cellType;
  int xx;
};

__attribute__((ejit_period("static"))) struct CellConfig g_boardCfg;
// CHECK: @g_boardCfg = {{.*}} !ejit.metadata ![[PERIOD_META:[0-9]+]]

__attribute__((ejit_period_arr("cell"))) struct CellConfig g_cellCfg[16];
// CHECK: @g_cellCfg = {{.*}} !ejit.metadata ![[ARR_META:[0-9]+]]

__attribute__((ejit_entry))
void jit_entry(__attribute__((ejit_period_arr_ind("cell"))) int cellIdx) {
  // CHECK: define void @jit_entry({{.*}} !ejit.metadata ![[ENTRY_META:[0-9]+]]
  if (g_cellCfg[cellIdx].cellType == 2) {
    // CHECK: load {{.*}} !ejit.may_const ![[MAYCONST:[0-9]+]]
  }
}

// CHECK-DAG: ![[PERIOD_META]] = distinct !{![[PERIOD:[0-9]+]]}
// CHECK-DAG: ![[PERIOD]] = !{!"ejit_period", !"static"}
// CHECK-DAG: ![[ARR_META]] = distinct !{![[ARR:[0-9]+]]}
// CHECK-DAG: ![[ARR]] = !{!"ejit_period_arr", !"cell", i32 16}
// CHECK-DAG: ![[ENTRY_META]] = distinct !{![[ENTRY:[0-9]+]], ![[IND:[0-9]+]]}
// CHECK-DAG: ![[ENTRY]] = !{!"ejit_entry"}
// CHECK-DAG: ![[IND]] = !{!"ejit_period_arr_ind", !"cell", i32 0}
// CHECK-DAG: ![[MAYCONST]] = !{}
```

---

## 10. 实施顺序

| 步骤 | 内容 | 文件 | 预估 |
|------|------|------|------|
| 1 | TableGen 定义 (6 属性) | `Attr.td` | 0.5d |
| 2 | 诊断消息定义 (8 diagnostics) | `DiagnosticSemaKinds.td` | 0.5d |
| 3 | 文档定义 (6 docs) | `AttrDocs.td` | 0.5d |
| 4 | Sema handler 实现 (6 handlers) | `SemaEJIT.cpp` + `SemaDeclAttr.cpp` | 2d |
| 5 | 防呆检测：写 may_const 检测 | `SemaEJIT.cpp` | 0.5d |
| 6 | CodeGen metadata 生成 | `CGEJIT.cpp` + `CodeGenModule.cpp` | 1.5d |
| 7 | CodeGen load metadata 注入 | `CGExpr.cpp` | 0.5d |
| 8 | CMakeLists.txt 更新 | `Sema/CMakeLists.txt` + `CodeGen/CMakeLists.txt` | 0.5d |
| 9 | Sema 测试编写 | `test/Sema/ext_attr_ejit.cpp` | 1d |
| 10 | CodeGen 测试编写 | `test/CodeGen/ejit_metadata.c` | 0.5d |

**总计**: 约 8 人天。

---

*文档版本: 1.0*
*创建日期: 2026-05-03*
*关联: SPEC4.md, PLAN4.md*
