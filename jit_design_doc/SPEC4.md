# EmbeddedJIT 需求规格说明书

## 1. 项目概述

**项目名称**: EmbeddedJIT
**项目类型**: 嵌入式场景 JIT 编译系统
**核心功能**: 基于时间窗常量的运行时特化编译器，用于优化嵌入式设备的代码执行效率
**目标用户**: 嵌入式系统开发者，使用 BiSheng Embedded CPU 的 clang/llvm 编译器的业务场景

EmbeddedJIT 的核心思想：在 `ejit_activate` 与 `ejit_deactivate` 定义的时间窗内，将运行时数据视为编译期常量，通过 JIT 特化生成高效代码。当时间窗失效时，自动回退到 AOT 代码。

---

## 2. 用户标注接口

> **属性拼写约定**: 所有带字符串参数的属性（`ejit_period`/`ejit_period_arr`/`ejit_period_arr_ind`/`ejit_period_lc`）使用 `#x` 宏字符串化。用户代码通过宏使用无引号标识符（如 `ejit_period(static)`），宏展开后变为 `__attribute__((ejit_period("static")))`。代码示例中统一使用宏形式。

### 2.1 数据不变性标注：ejit_may_const

```c
__attribute__((ejit_may_const))
#define ejit_may_const __attribute__((ejit_may_const))
```

**语义**: 标记结构体成员在时间窗内的不变性。当全局变量属于某个时间窗且时间窗处于激活状态时，标记了 `ejit_may_const` 的成员可被视为常量。

**示例**:

```c
struct Sample {
    ejit_may_const uint32_t a;
    uint32_t xx;
    struct SampleInner {
        ejit_may_const uint32_t b;
        uint32_t xxx;
    } inner;
};
```

**约束**:
- 支持整型、布尔型、浮点类型、嵌套结构体字段，及对应类型的数组
- `volatile` 字段不视为常量

### 2.2 时间窗标注

#### 2.2.1 ejit_period(name) — 标量全局变量

```c
__attribute__((ejit_period("static")))
#define ejit_period(x) __attribute__((ejit_period(#x)))
```

> **拼写说明**: 参数为字符串字面量。`static` 是 C 保留关键字，必须以 `"static"` 形式传入。

**语义**: 定义全局变量所属的时间窗。`name` 为开发者定义的具有业务含义的时间窗名称。

**约束**: 仅用于标记非数组的全局变量。

**内置时间窗 `static`**: 代表在 JIT 运行期不变的全局变量，永远处于生效状态，无需调用 `ejit_activate`。

**自定义时间窗 [未来扩展]**: `name` 为开发者自定义的名称（非 static），表示该变量在运行期可能变化、但在业务逻辑保证的时间段内保持不变。当前版本不支持函数对自定义标量时间窗的依赖声明，仅支持 `static` 和 `ejit_period_arr`。

```c
struct BoardConfig {
    ejit_may_const boardType;
    uint32_t xx;
};

// static 时间窗：JIT 运行期不变
ejit_period(static) struct BoardConfig g_boardCfg;
```

#### 2.2.2 ejit_period_arr(name) — 数组全局变量

```c
__attribute__((ejit_period_arr("cell")))
#define ejit_period_arr(x) __attribute__((ejit_period_arr(#x)))
```

**语义**: 定义全局变量数组所属的时间窗数组。时间窗数组管理具有相同业务概念但状态独立的多个实例，每个实例可独立控制时间窗状态。多个不同数组可使用相同的 `name`（即同一时间窗名称），此时 `ejit_activate(name, idx)` 会同时激活该名称下所有数组的第 idx 个实例；`ejit_activate_array(name, ptr, idx)` 仅激活指定数组。

```c
struct CellConfig {
    ejit_may_const cellType;
    uint32_t xx;
};

// 每个数组元素为独立的时间窗实例
ejit_period_arr(cell) struct CellConfig g_cellCfg[N];
```

**约束**:
- 数组长度 N 固定且较小 (<100)（仅数组类型；指针类型无大小约束，由用户保证越界安全）
- 整个程序最多支持 1024 个 `ejit_period_arr` 数组

**指针类型支持 (v1.6)**: `ejit_period_arr` 也支持指向结构体的指针变量：

```c
ejit_period_arr(cell) struct CellConfig *g_cellPtr;
// g_cellPtr = malloc(N * sizeof(struct CellConfig));
// 用户保证在 ejit_entry 调用前指针已赋值，下标不越界
```

PASS6 在 JIT 编译期通过读取 `*(void**)&GV` 获取指针值作为数组基地址。指针必须指向结构体/类类型。

### 2.3 JIT 函数标注

#### 2.3.1 ejit_entry — JIT 优化函数

```c
__attribute__((ejit_entry))
```

**语义**: 标记函数将进行 JIT 优化。编译器将：
1. 在函数入口插入 JIT 编译触发代码
2. 为该函数生成包含必要符号的 IR 并嵌入二进制文件

**约束**:
- 不支持递归
- 异步编译模式下，首次调用执行 AOT 路径（fallback），Cache 命中后执行特化版本。因此 `ejit_entry` 函数的**副作用必须是幂等的**——相同参数下多次执行的副作用效果一致。若函数包含非幂等操作（如修改全局计数器、写入硬件寄存器），应使用同步编译模式或避免标记为 `ejit_entry`

**默认依赖**: 每个 `ejit_entry` 函数自动依赖 `static` 时间窗。对 `ejit_period_arr` 时间窗的依赖通过 `ejit_period_arr_ind` 在函数参数上显式声明。

```c
ejit_entry void jit_entry(void)
{
    // JIT 优化时，g_boardCfg.boardType 被视为常量进行分支优化
    if (g_boardCfg.boardType == UBBPg) {
        doSomeThing();
    }
}
```

#### 2.3.2 ejit_period_arr_ind(name) — 特化维度参数

```c
__attribute__((ejit_period_arr_ind("cell")))
#define ejit_period_arr_ind(x) __attribute__((ejit_period_arr_ind(#x)))
```

**语义**: 标记函数参数为指定时间窗数组的下标，表明函数依赖该时间窗数组的特定实例，具体使用哪个实例由该参数值决定。仅适用于 `ejit_period_arr(name)` 数组变量。

```c
ejit_entry void jit_entry(ejit_period_arr_ind(cell) uint8_t cellIndex)
{
    // JIT 优化时，g_cellCfg[cellIndex].cellType 被视为常量
    if (g_cellCfg[cellIndex].cellType == FDD) {
        doSomeThing();
    }
}
```

**约束**:
- 参数类型必须为整数类型
- 单个函数最多支持 4 个 `ejit_period_arr_ind` 参数
- 每个维度最多关联 1024 个数组

**Cache Key 格式** (v1.8: uint64_t):

```
┌──────────────────────┬──────────┬──────────┬──────────┬──────────┐
│    funcIdx (32b)     │  d[3]    │  d[2]    │  d[1]    │  d[0]    │
│                      │  (8b)    │  (8b)    │  (8b)    │  (8b)    │
└──────────────────────┴──────────┴──────────┴──────────┴──────────┘
 bit 63-32              31-24      23-16      15-8       7-0
```

- **funcIdx**：FNV-1a 32-bit hash of 函数名，AOT 编译期计算，运行时一致
- **d[0..3]**：每个维度 index 占 8-bit（完整 uint8_t，与 wrapper 中 d[i] 参数一致），按 `ejit_period_arr_ind` 参数顺序排列
- 无维度时低 32 位为 0
- 相比旧方案（字符串拼接 `"fnName|cell=3,trp=1"`），uint64_t 直接比较/哈希，无内存分配

#### 2.3.3 ejit_period_lc(name) — 时间窗生命周期管理函数

```c
__attribute__((ejit_period_lc("cell")))
#define ejit_period_lc(x) __attribute__((ejit_period_lc(#x)))
```

**语义**: 标记函数会修改指定时间窗内的数据。编译器将在函数入口插入 `ejit_deactivate`、函数出口插入 `ejit_activate`，确保数据一致性。

**约束**: 必须与 `ejit_period_arr_ind` 配合使用，标记函数参数对应的数组下标。

```c
ejit_period_lc(cell)
void change_cell_cfg(ejit_period_arr_ind(cell) uint8_t cellIndex);
```

---

## 3. 运行时 API

### 3.1 核心类型定义

```c
/* 错误码 */
typedef enum {
    EJIT_OK = 0,
    EJIT_ERR_INVALID_PARAM = -1,
    EJIT_ERR_NOT_ACTIVE = -2,
    EJIT_ERR_COMPILE_FAILED = -3,
    EJIT_ERR_CACHE_FULL = -4,
    EJIT_ERR_MEMORY = -5,
    EJIT_ERR_BITCODE_NOT_FOUND = -6
} ejit_status_t;

/* 编译模式 */
typedef enum {
    EJIT_COMPILE_SYNC,    /* 同步编译：阻塞等待 JIT 完成 */
    EJIT_COMPILE_ASYNC    /* 异步编译：立即返回，后台编译 */
} ejit_compile_mode_t;

/* 优化等级 */
typedef enum {
    EJIT_OPT_L1 = 1,      /* 保守：常量传播 + 死代码消除 */
    EJIT_OPT_L2 = 2,      /* 中等：+ 函数内联 + CFG 简化 */
    EJIT_OPT_L3 = 3       /* 激进：+ 循环展开 */
} ejit_opt_level_t;

/* 维度编码在 cacheKey 中 —— 不再需要 ejit_dim_t 结构 */```

### 3.2 初始化与配置

| API | 说明 |
|-----|------|
| `ejit_init(const ejit_config_t* config)` | 初始化 JIT 运行时，NULL 使用默认配置 |
| `ejit_shutdown(void)` | 终止 JIT 运行时 |
| `ejit_set_compile_mode(ejit_compile_mode_t mode)` | 设置编译模式 |
| `ejit_get_compile_mode(void)` | 获取当前编译模式 |

**配置结构体**:

```c
typedef struct {
    ejit_compile_mode_t mode;       /* 编译模式：同步/异步 */
    ejit_opt_level_t opt_level;     /* 优化等级：1/2/3 */
    size_t max_cache_size;          /* 最大缓存大小 (字节) */
} ejit_config_t;
```

### 3.3 生命周期管理

| API | 说明 |
|-----|------|
| `ejit_activate(period_name, cell_idx)` | 激活指定时间窗名称下所有数组的第 `cell_idx` 个实例 |
| `ejit_deactivate(period_name, cell_idx)` | 标记指定时间窗名称下所有数组的第 `cell_idx` 个实例失效 |
| `ejit_activate_array(period_name, array_ptr, cell_idx)` | 激活指定时间窗名称下**指定数组**的第 `cell_idx` 个实例 |
| `ejit_deactivate_array(period_name, array_ptr, cell_idx)` | 标记指定时间窗名称下指定数组的第 `cell_idx` 个实例失效 |
| `ejit_activate_all(period_name)` | 激活指定时间窗名称下所有数组的所有实例 |
| `ejit_deactivate_all(period_name)` | 标记指定时间窗名称下所有数组的所有实例失效 |
| `ejit_is_active(period_name, cell_idx)` | 检查时间窗实例是否激活 |

**`ejit_activate` vs `ejit_activate_array`**: 同一时间窗名称可关联多个 `ejit_period_arr` 数组。`ejit_activate` 按 name + index 激活该名称下所有数组的对应实例；`ejit_activate_array` 通过额外传入数组指针，仅激活指定数组。

**数组越界行为**: `ejit_activate(period_name, cell_idx)` 激活该名称下所有数组的第 `cell_idx` 个实例。若不同数组大小不同（如 `g_cellCfg[16]` 和 `g_cellCfg2[8]`），`cell_idx` 对较小数组越界时，该数组实例被**静默跳过**（不报错，不激活），仅激活大小足够的数组实例。`cell_idx` 为负数时返回 `EJIT_ERR_INVALID_PARAM`。

**状态流**:

```
inactive → ejit_activate() → active (可 JIT)
active  → ejit_deactivate() → invalid (缓存失效)
invalid → ejit_activate()   → active (可 JIT)
```

### 3.4 符号注册

```c
void ejit_register_symbol(const char *name, void *addr);
```

用于裸核/嵌入式场景，将外部符号（库函数、全局变量）注册到 JIT 引擎，使 JIT 编译时可以解析。PASS1 在编译期自动扫描 ejit_entry 闭包引用的外部符号并生成注册调用，用户通常无需手动调用。

| 参数 | 说明 |
|------|------|
| `name` | 符号名称（与 bitcode 中声明一致） |
| `addr` | 符号运行时地址 |

**自动注册**: PASS1 `EJitRegisterBitcodePass` 扫描闭包中的外部函数调用和全局变量引用，在 `ejit_auto_register` 构造函数中自动生成 `ejit_register_symbol` 调用。Constructor 阶段暂存到 `EJitRegistrationStore`，`ejit_init` 时消费并转发给 JIT 引擎。

### 3.5 编译接口

```c
void* ejit_compile_or_get(uint64_t cacheKey,
                          int count, void** out_pfn);
```

| 参数 | 说明 |
|------|------|
| `funcIdx` | FNV-1a 32-bit hash of 函数名（AOT 编译期计算） |
| `cacheKey` | 预计算的 uint64_t key = funcIdx(32b) \| dims(4x8b) |
| `count` | 维度数量（无维度时为 0，dims 为 NULL） |
| `out_pfn` | 保留扩展参数，当前传 NULL |

**返回值**:
- 非 NULL：特化函数指针，调用方可直接调用
- NULL：编译失败或异步模式尚未完成，调用方应 fallback 到原函数

**行为说明**:
- 同步模式：首次调用阻塞等待编译完成，返回特化函数指针
- 异步模式：首次调用触发后台编译，立即返回 NULL，下次调用命中 Cache

---

## 4. 行为规格

### 4.1 AOT 编译期行为

编译器在 AOT 阶段需完成以下工作：

1. **插桩**: 为 `ejit_entry` 函数生成 wrapper 逻辑，在函数入口插入对 `ejit_compile_or_get` 的调用
2. **Bitcode 收集**: 为 `ejit_entry` 函数生成包含必要符号的 IR，嵌入二进制文件
3. **元数据生成**: 记录 JIT 优化所需的元数据（函数依赖的时间窗、变量、常量成员等）
4. **结构体布局注册**: 生成结构体字段布局信息的注册代码，供运行时字段偏移计算使用

**插桩行为要求**:
- 采用单函数混合方案：wrapper 逻辑直接插入原函数入口
- JIT 成功 → 调用特化函数并返回
- JIT 失败 → 继续执行原函数逻辑（fallback）

**生命周期函数插桩要求**:
- 对标记 `ejit_period_lc` 的函数，函数入口插入 `ejit_deactivate_array`，函数出口插入 `ejit_activate_array`

### 4.2 JIT 运行期行为

JIT 编译的整体流程：

1. 按需加载 Bitcode → Module
2. 根据 Cache Key 查找 Code Cache，命中则直接返回
3. 未命中则进入特化流程：
   - 替换 `ejit_period_arr_ind` 参数为运行时常量
   - 替换 `ejit_may_const` 成员 load 为运行时值（从激活实例中读取当前值）
   - 使用结构体布局信息计算正确的字段偏移
4. 运行可配置的 IR 优化 pipeline
5. 生成 Machine Code，存入 Code Cache
6. 返回函数指针

**缓存失效与重新编译**:
- `ejit_deactivate` 触发缓存失效：依赖于该时间窗实例的所有 Code Cache 条目被标记为无效
- 再次 `ejit_activate` 后，下次函数调用时发现缓存失效，重新触发 JIT 编译流程
- 被标记无效的缓存条目在其关联的特化函数不再被任何线程执行后，由运行时回收

### 4.3 编译模式

| 模式 | 行为 | 适用场景 |
|------|------|----------|
| 同步编译 | 首次调用阻塞等待 JIT 编译完成 | 对性能要求高，可容忍首次阻塞 |
| 异步编译 | 首次调用触发后台编译，立即返回 NULL | 对首次延迟敏感 |

**配置方式**: 支持编译时配置（CMake 宏）和运行时配置（`ejit_init`）两种方式。

### 4.4 优化能力

优化通过 `ejit_opt_level_t` 配置等级控制：

| 优化类型 | 说明 | L1 (保守) | L2 (中等) | L3 (激进) |
|---------|------|-----------|-----------|-----------|
| 结构体访存消除 | 将 `g_xxx[idx].field` 替换为常量 | ✓ | ✓ | ✓ |
| 常量传播 | 跨函数常量传播 | ✓ | ✓ | ✓ |
| 死代码消除 | 删除不可达代码 | ✓ | ✓ | ✓ |
| 分支折叠 | 常量条件消除不可达分支 | ✓ | ✓ | ✓ |
| 函数内联 | 内联被调用函数 | | ✓ | ✓ |
| CFG 简化 | 控制流图简化 | | ✓ | ✓ |
| 循环展开 | 极端情况下完全展开 | | | ✓ |

---

## 5. 应用场景案例

### 场景 1: 运行期不变全局变量

```c
struct BoardConfig {
    ejit_may_const boardType;
    uint32_t xx;
};

ejit_period(static) struct BoardConfig g_boardCfg;

ejit_entry void jit_entry(void);
```

### 场景 2: 单时间窗依赖

```c
struct BoardConfig {
    ejit_may_const boardType;
    uint32_t xx;
};
struct CellConfig {
    ejit_may_const cellType;
    uint32_t xx;
};

ejit_period(static) struct BoardConfig g_boardCfg;
ejit_period_arr(cell) struct CellConfig g_cellCfg[N];

// 依赖 static + cell[cellIndex]
ejit_entry void jit_entry(ejit_period_arr_ind(cell) uint8_t cellIndex);

// 生命周期管理
ejit_period_lc(cell)
void change_cell_cfg(ejit_period_arr_ind(cell) uint8_t cellIndex);
```

### 场景 3: 多时间窗依赖

```c
struct BoardConfig  { ejit_may_const boardType; uint32_t xx; };
struct CellConfig   { ejit_may_const cellType;  uint32_t xx; };
struct TrpConfig    { ejit_may_const trpType;   uint32_t xx; };

ejit_period(static) struct BoardConfig g_boardCfg;
ejit_period_arr(cell) struct CellConfig g_cellCfg[N];
ejit_period_arr(trp) struct TrpConfig g_trpCfg[M];

// 显式依赖 static + cell[cellIndex] + trp[trpIndex]
ejit_entry void jit_entry(
    ejit_period_arr_ind(cell) uint8_t cellIndex,
    ejit_period_arr_ind(trp) uint8_t trpIndex);

// 方式一：同时变更多个时间窗
ejit_period_lc(cell) ejit_period_lc(trp)
void change_cfg(ejit_period_arr_ind(cell) uint8_t cellIndex,
                ejit_period_arr_ind(trp) uint8_t trpIndex);

// 方式二：分别变更
ejit_period_lc(cell)
void change_cell_cfg(ejit_period_arr_ind(cell) uint8_t cellIndex);
ejit_period_lc(trp)
void change_trp_cfg(ejit_period_arr_ind(trp) uint8_t trpIndex);
```

### 场景 4: 多时间窗隐式依赖 [未来扩展]

函数通过结构体内的关联关系间接依赖多个时间窗，当前版本暂不支持。

---

## 6. 目标硬件平台

| 属性 | 规格 |
|------|------|
| CPU 架构 | ARM Cortex-A (高性能 CPU) |
| 内存约束 | RAM 100KB - 2MB (中等受限) |
| 存储约束 | Flash 对应中等受限 |
| 实时性 | 软实时 (ms 级)，可容忍首次 JIT 编译延迟 |

---

## 7. 约束与限制

| 约束项 | 规格 |
|--------|------|
| 并发模型 | 主线程执行用户代码，异步编译在独立后台线程完成，使用轻量级同步机制 |
| 外部函数 | `ejit_entry` 函数调用纯内部函数 (static) |
| 函数指针 | 暂不特化函数指针调用 |
| 编译器 | 仅支持 Clang |
| 版本管理 | 不需要，编译时固定结构体定义 |
| 长期运行 | 可接受重启，无需特殊内存管理 |
| 单函数代码量 | 需限制单个特化函数最大代码量，防止循环展开导致膨胀 |
| Code Cache | 需要淘汰策略 (如 LRU) + 大小限制 |
| 数组大小 | 时间窗数组长度 <100，全局最多 1024 个数组 |
| 特化维度 | 单函数最多 4 个 `ejit_period_arr_ind` 参数 |
| 函数递归 | `ejit_entry` 函数不支持递归 |

---

## 8. 错误处理

| 错误场景 | 处理策略 |
|---------|---------|
| Bitcode 解析失败 | 记录日志，fallback 到 AOT |
| 特化过程出错 | 记录详细日志，fallback 到 AOT |
| 内存不足 | 触发缓存淘汰，fallback 到 AOT |
| cellIdx 越界 | 边界检查失败时记录日志，fallback 到 AOT |
| 空指针访问 | 空指针检查，检测到时 fallback |

**核心原则**: 所有 JIT 失败场景必须 fallback 到 AOT，不能导致程序崩溃。

---

## 9. 防呆设计

### 9.1 全局变量归属冲突检测

检查同一变量是否被多个 `ejit_period` 或 `ejit_period_arr` 标记。**错误级别**，阻止编译。

```c
// 错误：同一变量被多个时间窗标记
ejit_period(cell) struct CellConfig g_cellCfg;
ejit_period(trp)  struct CellConfig g_cellCfg;
```

### 9.2 时间窗修改点检测

检查修改 `ejit_may_const` 成员但未标记生命周期管理函数的情况。**警告级别**，不阻止编译。

```c
void unsafe_function(void) {
    g_cellCfg.cellType = TDD;  // 警告：未标记 ejit_period_lc
}
```

### 9.3 生命周期调用链检测 [暂不支持]

检查 `ejit_entry` 函数调用了生命周期管理函数但未正确处理的情况。**警告级别**。

### 9.4 元数据一致性检测

检查标注声明与实际使用是否一致。**警告级别**，不阻止编译。

```c
ejit_period(cell) struct CellConfig g_cellCfg[N];
ejit_entry void jit_entry(ejit_period_arr_ind(trp) uint8_t trpIndex) {
    // 警告：标记依赖 trp 但实际使用 cell 数据
    use_cell_data(trpIndex);
}
```

---

## 10. 非功能性需求

| 类别 | 需求 |
|------|------|
| JIT 首次编译延迟 | < 100ms (软实时可接受) |
| 特化函数执行性能 | 显著优于 AOT 代码 |
| 内存占用 | Code Cache 上限受内存约束限制 |
| 可靠性 | 所有 JIT 失败必须 fallback，不崩溃 |
| 日志 | JIT 编译失败时记录详细日志 |
| 模块化 | 核心模块逻辑独立，易于维护 |
| 调试支持 | 后期实现源码级调试 |

---

## 11. 部署方式

| 组件 | 方式 |
|------|------|
| Bitcode | 嵌入到可执行文件的独立段，静态链接 |
| 运行时库 | 独立提供 (libejit.a) |
| 加载时机 | 按需加载，首次调用时加载 Bitcode |

---

## 12. 关键设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 内存模式 | 按需加载 Bitcode | 减少启动时内存占用 |
| 优化等级 | 可配置 (1/2/3 级) | 适应不同场景需求 |
| 缓存策略 | LRU 淘汰 + 大小限制 | 防止内存耗尽 |
| 错误策略 | 记录日志后 fallback | 保证系统稳定性 |
| 编译模式 | 同步/异步可选 | 适应不同实时性需求 |
| 插桩方案 | 单函数混合方案 | 无需单独 fallback 函数 |
| 调试支持 | 后期实现 | 优先功能完善 |

---

## 13. 未来扩展 [暂不实现]

- 自定义 `ejit_period(name)` 标量变量的函数依赖声明机制
- 多时间窗隐式依赖
- 函数指针特化
- 源码级调试功能
- 多线程安全支持
- 其他编译器 (GCC) 兼容

---

## 14. 术语表

| 术语 | 定义 |
|------|------|
| AOT | Ahead-of-Time，预先编译 |
| JIT | Just-in-Time，运行时编译 |
| Bitcode | LLVM IR 的二进制格式 |
| 特化 (Specialization) | 根据特定参数生成专用代码 |
| 时间窗常量 | 在特定时间段内保持不变的运行时数据 |
| Code Cache | JIT 编译产物的代码缓存 |
| Fallback | JIT 失败时回退到 AOT 代码执行 |

---

*文档版本: 1.1*
*创建日期: 2026-04-11*
*更新日期: 2026-04-24*
