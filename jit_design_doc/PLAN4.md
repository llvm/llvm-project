# EmbeddedJIT 技术方案 v2.1

**版本**: 2.1
**日期**: 2026-04-15
**关联文档**: SPEC4.md
**基于**: PLAN3.md (v2.0)

---

## 0. 背景

基于 easyJIT 的 Bitcode 嵌入、JIT 编译流程、Cache 管理、异步编译机制进行复用和扩展。easyJIT 基于参数值特化；EmbeddedJIT 基于 **时间窗常量 + 结构体字段索引** 特化，新增时间窗生命周期管理和结构体字段常量替换能力。

**可复用组件**: Bitcode 提取与嵌入 (`RegisterBitcodePass`)、Bitcode 加载器 (`BitcodeTracker`)、JIT 编译流程框架 (`Function::Compile`)、LLVM 内置 `ExtractGVPass`。

---

## 1. 目录结构

```
llvm/
├── lib/
│   ├── Transforms/
│   │   └── EmbeddedJIT/           // EmbeddedJIT AOT Pass 实现
│   │       ├── EJitWrapperGen.cpp          // Wrapper 插桩代码生成 Pass (新增)
│   │       ├── EJitPeriodHandler.cpp       // 生命周期标记处理 Pass (新增)
│   │       ├── EJitRegisterBitcode.cpp     // Bitcode 注册 Pass (参考 easyJIT)
│   │       ├── EJitRegisterPeriod.cpp      // period 变量注册 Pass (**新增**)
│   │       ├── EJitAotModulePass.cpp       // AOT 协调 Pass
│   │       └── CMakeLists.txt
│   │
│   └── ExecutionEngine/
│       └── EJIT/                   // 运行时库 (含 JIT Pipeline Pass)
│           ├── EJit.cpp           // 主实现
│           ├── EJitCache.cpp      // Code Cache 管理 (增强版 LRU)
│           ├── EJitRuntime.cpp    // 运行时状态管理 (新增 activate/deactivate)
│           ├── EJitCompiler.cpp   // 编译协调器
│           ├── EJitStructFieldPass.cpp  // 结构体字段特化 (JIT Pipeline, 新增)
│           ├── EJitAsyncCompiler.cpp    // 异步编译器 (工作线程 + 请求队列)
│           ├── EJitModuleLoader.cpp // Bitcode 加载器 (参考 easyJIT)
│           ├── EJitOptimizer.cpp    // 优化 pipeline
│           ├── EJitLogger.cpp      // 日志系统
│           └── CMakeLists.txt
│
├── include/
│   └── llvm/
│       └── ExecutionEngine/
│           └── EJIT/               // 运行时库头文件
│               ├── EJit.h              // 主 API 头文件
│               ├── EJitError.h         // 错误类型定义
│               ├── EJitOptions.h       // 配置选项
│               └── EJitRuntime.h       // C 运行时接口
│
├── test/
│   ├── Transforms/
│   │   └── EmbeddedJIT/           // AOT Pass Lit 测试
│   │       ├── test_wrapper_gen.ll
│   │       ├── test_period_handler.ll
│   │       ├── test_register_bitcode.ll
│   │       └── test_register_layout.ll
│   │
│   └── ExecutionEngine/
│       └── EJIT/                   // Runtime + JIT Pass 测试
│           ├── EJitCacheTest.cpp
│           ├── EJitStructFieldTest.cpp
│           ├── EJitCompilerTest.cpp
│           └── CMakeLists.txt
│
├── examples/
│   └── EmbeddedJIT/               // 示例代码
│       ├── demo.c
│       └── CMakeLists.txt
│
└── docs/
    └── EmbeddedJIT/               // 设计文档
        └── design.md
```

**Clang 属性支持** (需添加属性定义)

```
clang/test/
├── Sema/
│   └── ext_attr_ejit_*.cpp       // 语义分析测试
└── CodeGen/
    └── ejit_*.c                   // IR 生成测试
```

**注意**: Clang 属性基本实现在

```
clang/include/clang/Basic/Attr.td
clang/include/clang/Basic/AttrDocs.td
clang/lib/CodeGen/CGExpr.cpp
clang/lib/CodeGen/CodeGenModule.cpp
```

不需要特殊隔离

---

## 2. 整体架构

### 2.1 端到端流程

```
┌──────────────────────────────────────────────────────────────────────────┐
│  阶段 1: AOT 编译                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ Clang 编译用户代码 → 生成 IR (含 !ejit.may_const metadata)        │   │
│  │                                                                   │   │
│  │ [早期 Pass — 标准优化之前]                                        │   │
│  │ - EJitRegisterBitcodePass 提取原始 bitcode (metadata 完整)       │   │
│  │                                                                   │   │
│  │ [AOT Pass — 标准优化之前 (非LTO) / 之后 (LTO)]                    │   │
│  │ - EJitRegisterPeriodPass 生成 period 数组/static 变量注册代码     │   │
│  │ - EJitWrapperGenPass 生成 Wrapper 插桩代码                        │   │
│  │ - EJitPeriodHandlerPass 处理 ejit_period_lc 生命周期              │   │
│  │                                                                   │   │
│  │ [标准优化 Pipeline — O2/O3] (非LTO 路径，优化 wrapper IR)         │   │
│  │ - SROA / InstCombine / GVN / Inline / ...                        │   │
│  │                                                                   │   │
│  │ - AOT 编译为可执行文件 + 嵌入 bitcode                              │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  阶段 2: 运行时                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ 激活时间窗: ejit_activate(periodName, cellIdx)                    │   │
│  │ - 记录时间窗基地址映射                                              │   │
│  │ - 设置时间窗状态为 Active                                          │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                   ↓                                      │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ 用户调用 ejit_entry 函数                                          │   │
│  │ - 实际执行 Wrapper 插桩代码                                        │   │
│  │ - 查询 Code Cache (封装在 ejit_compile_or_get 内部)               │   │
│  │ - 命中 → 直接调用特化函数                                          │   │
│  │ - 未命中 → JIT 编译 → Cache → 调用                                 │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件交互

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  EJitCache   │ ←── │  EJitRuntime    │ ←── │ ejit_compile_    │
│  (函数缓存)   │     │  (时间窗管理)    │     │ or_get()         │
└──────────────┘     └─────────────────┘     └──────────────────┘
        ↑                    ↑                        ↑
   存储特化函数          activate/deactivate     JIT 编译触发
   返回函数指针         管理时间窗基地址映射
                        ↓
              ┌──────────────────────┐
              │ PeriodArrayRegistry  │ ← AOT 注册代码填入
              │ (数组→时间窗映射)      │
              └──────────────────────┘
                        ↑
              JIT 编译时读取: baseAddr + offset → 常量值
```

### 2.3 Wrapper 调用流程

```
用户调用:                          运行时:
process_task_multi(idx, iter);  →  Wrapper 构建 dims
                                       ↓
                                 ejit_compile_or_get()
                                       ↓
                                 [内部: 查Cache → 命中 → 返回]
                                 [内部: 未命中 → JIT编译 → 存Cache → 返回]
                                       ↓
                                 pfn != NULL → 调用特化函数
                                 pfn == NULL → fallback 到 AOT
```

**关键点**：

- Cache 查询封装在 `ejit_compile_or_get` 内部，调用者不感知缓存逻辑
- 根据 **funcIdx（FNV-1a hash）** + **ejit_period_arr_ind 参数值** 构建 Cache key
- **常量读取时机**：JIT 编译时按需读取 `ejit_may_const` 字段值，从进程内存直接读取
- 用户调用原函数名，实际执行 Wrapper 插桩代码

---

## 3. 核心数据模型

### 3.1 运行时状态模型

根据 SPEC4.md 的时间窗设计：

```cpp
// 运行时值类型 (支持多种常量类型)
union RuntimeValue {
    int32_t intVal;
    int64_t longVal;
    float floatVal;
    double doubleVal;         // v1.6: 双精度浮点
    bool boolVal;
    void* ptrVal;             // v1.5: 指针类型 (如函数指针)
};

// 时间窗状态枚举
enum class PeriodState {
    Inactive,   // 未激活
    Active,     // 已激活，可 JIT
    Invalid     // 已失效，缓存待清理
};
```

#### 3.1.1 AOT 注册数据结构

AOT 阶段由 `EJitRegisterPeriodPass` 生成注册代码，在 `ejit_init` 时自动执行，将全局变量运行时地址桥接到 JIT 编译环境。

```cpp
// ejit_period_arr 数组注册信息
// 由 AOT 生成的注册代码调用 ejit_register_period_array() 填入
struct PeriodArrayInfo {
    std::string varName;       // 全局变量名 (如 "g_cellCfg")
    std::string periodName;    // 时间窗名称 (如 "cell")
    void* baseAddr;            // 数组基地址 (注册时填入 &g_cellCfg)
    size_t arraySize;          // 数组长度
};

// static 时间窗变量注册信息
struct StaticVarInfo {
    std::string varName;       // 全局变量名 (如 "g_boardCfg")
    void* varAddr;             // 变量地址
};

// 时间窗数组注册表
// periodName → 该名称下所有 ejit_period_arr 数组
class PeriodArrayRegistry {
    std::unordered_map<std::string, std::vector<PeriodArrayInfo>> arraysByPeriod_;
    std::vector<StaticVarInfo> staticVars_;
public:
    // AOT 注册代码调用
    void registerArray(const std::string& periodName, const std::string& varName,
                       void* baseAddr, size_t size);
    void registerStaticVar(const std::string& varName, void* varAddr);
    // JIT 编译时使用
    const std::vector<PeriodArrayInfo>* getArrays(const std::string& periodName) const;
    const std::vector<StaticVarInfo>& getStaticVars() const;
};
```

**AOT 生成的注册代码示例**:

```c
// 由 EJitRegisterPeriodPass 自动生成
// 调用时机: ejit_init 内部自动调用，或在 main 之前通过 __attribute__((constructor)) 执行
static void ejit_auto_register(void) {
    // 注册数组时间窗
    ejit_register_period_array("cell", "g_cellCfg", &g_cellCfg, N);
    ejit_register_period_array("trp",  "g_trpCfg",  &g_trpCfg,  M);
    // 注册 static 时间窗
    ejit_register_static_var("g_boardCfg", &g_boardCfg);
}
```

#### 3.1.2 运行时激活状态

```cpp
// 时间窗激活状态
// 支持两种粒度:
//   - period 级: ejit_activate(name, idx) 激活该名称下所有数组
//   - array 级: ejit_activate_array(name, ptr, idx) 激活指定数组
struct PeriodInstance {
    std::string periodName;
    int cellIdx;
    PeriodState state;
    uint64_t activateTime;

    // 按数组粒度激活时使用:
    // period 级激活时 arrayInfo == nullptr (表示该名称下所有数组)
    // array 级激活时 arrayInfo 指向具体数组
    PeriodArrayInfo* arrayInfo = nullptr;
};
```

**时间窗模型说明**：

| 时间窗类型  | 标记                      | 激活方式                           | 说明         |
| ------ | ----------------------- | ------------------------------ | ---------- |
| static | `ejit_period(static)`   | 永远激活                           | 运行期不变的全局变量 |
| 数组     | `ejit_period_arr(name)` | `ejit_activate(name, cellIdx)` | 独立控制的数组实例  |

注：自定义 `ejit_period(name)` 标量变量为未来扩展，当前版本不支持。参见 SPEC4.md §2.2.1。

**激活流程**:

```
ejit_activate("cell", 3):
  1. 从 PeriodArrayRegistry 查找 periodName="cell" 的所有数组
  2. 记录 (periodName="cell", cellIdx=3, arrayInfo=null) → Active
  3. 含义: 该名称下所有数组在 idx=3 都视为已激活

ejit_activate_array("cell", &g_cellCfg, 3):
  1. 从 PeriodArrayRegistry 查找 periodName="cell" 且 baseAddr=&g_cellCfg 的数组
  2. 记录 (periodName="cell", cellIdx=3, arrayInfo=&g_cellCfg_info) → Active
  3. 含义: 仅 g_cellCfg 在 idx=3 视为已激活

JIT 编译时检查:
  - 函数依赖 "cell" + "trp" 两个 period
  - 查找 PeriodInstance: 两个 period 对应的 cellIdx 是否都是 Active
  - 全部 Active → 允许 JIT 编译，读取常量
  - 任一非 Active → 返回 NULL，fallback 到 AOT

JIT 编译时读取常量:
  1. 从 PeriodArrayRegistry 获取数组 baseAddr
  2. 从 GEP 链通过 accumulateConstantOffset 计算 byteOffset（地址计算用）
  3. 从进程内存读取: *(baseAddr + cellIdx * sizeof(Element) + fieldOffset)
  4. 替换 IR 中的 load 为常量
```

### 3.2 代码缓存模型

根据 SPEC4.md 的 Cache key 格式设计：

```cpp
// Cache Key: uint64_t = funcIdx(32b) | dim[3](8b) | dim[2](8b) | dim[1](8b) | dim[0](8b)
// funcIdx 由 FNV-1a hash 确定（AOT 编译期计算，运行时一致）。
// 无维度时低 32 位为 0。

// 缓存条目 — LRU list iterator 嵌入 Entry，getOrNull 一次 hash 完成 LRU bump
struct Entry {
    void* funcPtr;
    size_t codeSize;
    std::list<uint64_t>::iterator lruIt;  // embedded → O(1) splice/erase
    SmallVector<std::string, 4> periodDeps;
};

// Code Cache 管理器 — 线程安全，iterator-embedded LRU
class EJitCache {
    std::unordered_map<uint64_t, Entry> cache_;
    std::list<uint64_t> lruList_;         // key, LRU 顺序
    // 不再需要 lruIter_ 反向映射 — iterator 嵌在 Entry 里

    std::unordered_map<std::string, std::set<uint64_t>> periodIndex_;

    size_t maxEntries_        = 4096;      // 缓存条目上限
    size_t maxTotalSize_      = 32 MB;     // 总代码大小上限
    size_t maxSingleFuncSize_ = 512 KB;    // 单函数上限

public:
    void* getOrNull(uint64_t cacheKey);    // unique_lock (splice 是写操作)
    bool put(uint64_t cacheKey, void* fn, size_t codeSize,
             ArrayRef<std::string> periodDeps = {});
    void invalidateByPeriod(const std::string& periodName, uint8_t cellIdx);
    void clear();

    static uint64_t buildCacheKey(uint32_t funcIdx,
        const std::pair<std::string, uint8_t>* dims, unsigned count);
};

// 构建 Cache key
uint64_t buildCacheKey(uint32_t funcIdx,
                       const std::pair<std::string, uint8_t>* dims,
                       unsigned count) {
    uint64_t key = static_cast<uint64_t>(funcIdx) << 32;
    for (unsigned i = 0; i < count && i < 4; ++i)
        key |= static_cast<uint64_t>(dims[i].second) << (i * 8);
    return key;
}
```

**Cache key 格式说明** (SPEC4.md §2.3.2):

| 维度数 | Cache key (uint64_t) | 示例 |
| --- | --- | --- |
| 0 (static only) | `funcIdx << 32` | funcIdx=1 → `0x00000001_00000000` |
| 1 维 | `funcIdx(32b) + dim[0](8b)` | funcIdx=1, cellIdx=3 → `0x00000001_00000003` |
| 2 维 | `funcIdx(32b) + dim[0](8b) + dim[1](8b)` | cell=1, trp=5 → `0x00000001_00000501` |

### 3.2.1 运行时 Wrapper 调用模型

根据 SPEC4.md §4.1，Wrapper 由 AOT 编译时生成，每个 `ejit_entry` 函数有独立的 Wrapper 插桩代码。

**Wrapper 生成方式**：

- `EJitWrapperGenPass` 在 AOT 编译时为每个 `ejit_entry` 函数生成独立的 Wrapper
- Wrapper 包含 Cache 查询和 `ejit_compile_or_get()` 调用逻辑
- 用户直接调用原函数名，实际执行的是 Wrapper 代码
- `PFN_xxx` 是 Pass 生成的函数指针 typedef，签名与原函数一致，用于类型安全地转换 `ejit_compile_or_get` 返回的 `void*`

**单维度 Wrapper 示例 (单函数混合方案)**：

```c
// 原代码
void process_task_multi(uint8_t cellIdx, int iterations) { ... }

// AOT 编译后生成的 Wrapper (单函数混合方案)
void process_task_multi(uint8_t cellIdx, int iterations) {
    // === Wrapper 逻辑 (JIT 入口) ===
    // 维度编码在 cacheKey 中
    uint64_t key = (funcIdx << 32) | cellIdx;
    PFN_process_task_multi pfn = (PFN_process_task_multi)ejit_compile_or_get(key, NULL);

    if (pfn) {
        return pfn(cellIdx, iterations);  // JIT 成功，调用 specialized 版本
    }
    // Fallback: 继续执行原函数逻辑
    // ...
}
```

**多维度 Wrapper 示例 (单函数混合方案)**：

```c
// 原代码
void process_task_multi(uint8_t cellIdx, uint8_t trpIdx, int iterations) { ... }

// AOT 编译后生成的 Wrapper (单函数混合方案)
void process_task_multi(uint8_t cellIdx, uint8_t trpIdx, int iterations) {
    // === Wrapper 逻辑 (JIT 入口) ===
    // 维度编码在 cacheKey 中
        {"cell", cellIdx},   // cell 维度: g_cellCfg[cellIdx]
        {"trp", trpIdx}      // trp 维度: g_trpCfg[trpIdx]
    };
    PFN_process_task_multi pfn = (PFN_process_task_multi)ejit_compile_or_get(key, NULL);

    if (pfn) {
        return pfn(cellIdx, trpIdx, iterations);  // JIT 成功，调用 specialized 版本
    }
    // Fallback: 继续执行原函数逻辑
    // ...
}
```

**仅依赖 static 的 Wrapper 示例 (单函数混合方案)**：

```c
// 原代码 (无参数，仅依赖 static 时间窗)
void process_static_data(void) { ... }

// AOT 编译后生成的 Wrapper (单函数混合方案)
void process_static_data(void) {
    // === Wrapper 逻辑 (JIT 入口) ===
    // 无维度参数，dims 为 NULL，count 为 0
    PFN_process_static_data pfn = (PFN_process_static_data)ejit_compile_or_get(funcIdx << 32, NULL);

    if (pfn) {
        return pfn();  // JIT 成功，调用 specialized 版本
    }
    // Fallback: 继续执行原函数逻辑
    // ...
}
```

**关键点**：

| 项目         | 说明                                        |
| ---------- | ----------------------------------------- |
| Wrapper 生成 | AOT 编译时由 `EJitWrapperGenPass` 生成，单函数混合方案  |
| Wrapper 数量 | 每个 `ejit_entry` 函数只有一个函数                  |
| 参数传递       | Wrapper 在寄存器中计算 cacheKey（funcIdx<<32|dims），零栈分配    |
| 用户调用       | 直接调用原函数名，实际执行 wrapper + 原函数逻辑             |
| Fallback   | 单函数混合方案：JIT 失败时继续执行原函数逻辑，无需单独 fallback 函数 |
| out_pfn    | 保留作为 future 扩展，用于跨平台适配或状态查询               |

### 3.3 特化上下文模型

```cpp
// 特化参数上下文
// 注意: 使用 std::string 而非 const char*，
// 异步模式时请求在线程间传递，原始 C 字符串生命周期不足
struct SpecializationContext {
    std::string fnName;                     // 入口函数名
    int period_count;                       // 时间窗维度数量 (0-4，0 表示仅依赖 static)
    struct {
        std::string periodName;             // 时间窗名称 (如 "cell", "trp")
        int cellIdx;                        // 数组下标
    } dimensions[4];                        // 最多 4 个维度
    OptimizationLevel optLevel;             // 优化等级
};

// 优化等级配置 (对应 SPEC4.md 3.4.1)
enum class OptimizationLevel {
    Level1 = 1,  // 保守：SCCP + DCE
    Level2 = 2,  // 中等：+ Inline + CFGSimplify
    Level3 = 3   // 激进：+ LoopUnroll
};
```

### 3.4 属性元数据模型

根据 SPEC4.md 3.2 的属性定义：

```cpp
// 函数元数据 (对应 SPEC4.md §4.1)
struct EjitFuncMeta {
    const char* name;                       // 函数名
    const void* bitcode_ptr;                // 嵌入的 bitcode 指针
    size_t bitcode_size;                    // bitcode 大小
    int param_count;                        // 参数数量
    int dependency_count;                   // 依赖的时间窗数量 (0-4，不含隐式的 static)
    const char* dependencies[4];            // 依赖的时间窗名称列表
};
```

**属性类型说明** (SPEC4.md 3.2):

| 属性                          | 标记位置  | 说明               |
| --------------------------- | ----- | ---------------- |
| `ejit_may_const`            | 结构体成员 | 时间窗内可视为常量的成员     |
| `ejit_period(static)`       | 全局变量  | 运行期不变的全局变量       |
| `ejit_period_arr(name)`     | 全局数组  | 按索引独立控制的时间窗数组    |
| `ejit_entry`                | 函数    | JIT 优化入口函数       |
| `ejit_period_arr_ind(name)` | 函数参数  | 特化维度参数，关联对应时间窗数组 |
| `ejit_period_lc(name)`      | 函数    | 时间窗生命周期管理函数      |

### 3.5 ejit_may_const 识别模型

`ejit_may_const` 通过 `load` 指令上的 metadata 标注识别（仿造 `volatile` 的传参链，但用 metadata 而非 instruction bit）：

```llvm
; Clang CodeGen 为 ejit_may_const 字段的 load 标注 !ejit.may_const
%v = load i32, ptr %field, !ejit.may_const !{}
```

PASS6 在 JIT 时直接检查 `load->hasMetadata("ejit.may_const")`，不再需要 offset 计算 + 注册表查找。metadata 被优化 pass 丢弃时仅导致错过一次 JIT 优化（fallback 到 AOT），不影响正确性。

### 3.6 Pass 架构模型

Pass 分为两条独立的 Pipeline：AOT（标准优化前）、JIT（运行时）。

**Pipeline 时序 (非LTO 路径)**:

```
Clang CodeGen → [PASS1] → [PASS2-4] → O2/O3 标准优化 → 目标代码生成
                 ↓            ↓                              ↓
         提取原始 bitcode  wrapper 插桩              优化 wrapper IR
         (metadata 完整)  + period 注册             (fallback 路径优化等)
```

**Pipeline 时序 (LTO 路径)**:

```
Clang CodeGen → [PASS1] → O2/O3 标准优化 → [PASS2-4] → 目标代码生成
                 ↓                              ↓
         提取原始 bitcode              wrapper 插桩 + period 注册
         (metadata 完整)               (LTO 后的最终 IR)
```

> **为什么非 LTO 路径 PASS2-4 放在优化前**: PASS3 生成的 wrapper IR (jit_entry/jit_dispatch/jit_fallback 块)
> 会被 O2 优化，fallback 路径被优化（分支变 select、PHI 合并返回值等），生成更高效的机器码。
> PASS3 添加的 `noinline` 属性也在 inliner 运行前生效，防止 wrapper 被内联到 caller 中。
>
> **为什么 LTO 路径 PASS2-4 放在优化后**: LTO 已经完成了完整的优化流程，PASS2-4 在优化后插入 wrapper，
> 此时 IR 已是最终形态，无需再次优化。

```
AOT Pipeline:
┌───────────────────────────────────────────────┐
│  PASS1: EJitRegisterBitcode (始终在优化前)     │
│  PASS2: EJitRegisterPeriod                    │
│  PASS3: EJitWrapperGen                        │
│  PASS4: EJitPeriodHandler                     │
└───────────────────────────────────────────────┘
非LTO: PASS2-4 在 O2/O3 之前运行
LTO:   PASS2-4 在 O2/O3 之后运行

JIT Pipeline (ejit_compile_or_get 内部执行):
┌───────────────────────────────────────────────┐
│  1. 参数预处理: 替换 ejit_period_arr_ind 参数   │
│  2. InstCombine: 折叠常量链                    │
│  3. Inline: 内联被调用函数                      │
│  4. EJitStructFieldPass: 字段 load → 常量      │
│  5. 标准 LLVM 优化 (按 L1/L2/L3 编排)          │
└───────────────────────────────────────────────┘
输出: 特化机器码 (存入 Code Cache)
```

**PASS1 — 早期 AOT (始终在标准优化之前)**:

| Pass 名称               | 类型          | 职责                                           |
| --------------------- | ----------- | -------------------------------------------- |
| `EJitRegisterBitcode` | Module Pass | 提取 ejit_entry 函数 bitcode，运行 AOT 预优化（Inline+Mem2Reg+EarlyCSE+InstCombine+SimplifyCFG），`!ejit.may_const` 由固定 metadata kind + copyMetadataForLoad 白名单 + GV offset 回退三重保证，自动注册外部符号 |

**PASS2-4 — AOT (非LTO: 优化前; LTO: 优化后)**:

| Pass 名称               | 类型          | 职责                                           |
| --------------------- | ----------- | -------------------------------------------- |
| `EJitRegisterPeriod`  | Module Pass | 生成 period 数组/static 变量注册代码 (已移除 layout 注册) |
| `EJitWrapperGen`      | Module Pass | 为 ejit_entry 函数生成 Wrapper 插桩代码               |
| `EJitPeriodHandler`   | Module Pass | 为 ejit_period_lc 函数插入 deactivate/activate 调用 |
| `EJitAotModulePass`   | Module Pass | 协调上述 3 个子 Pass，按序执行                          |

**JIT Pipeline**:

| 步骤  | 组件                    | 职责                                                     |
| --- | --------------------- | ------------------------------------------------------ |
| 1   | 参数预处理 (非 Pass)        | `replaceAllUsesWith` 替换 ejit_period_arr_ind 参数为实际值     |
| 2   | InstCombine           | 折叠 `zext`/GEP 常量链，消除 PHI 节点                            |
| 3   | Inline                | 内联被调用函数，使 PASS6 可追溯到全局变量的 GEP 链                       |
| 4   | `EJitStructFieldPass` | 替换 ejit_may_const 字段 load 为运行时常量                       |
| 5   | 标准 LLVM Pass 编排       | 按 L1/L2/L3 等级运行 SCCP、DCE、CFGSimplify、LoopUnroll         |

> **为什么 Inline 在 PASS6 之前**: 被调用函数内部的 `ejit_may_const` load 的指针操作数是函数参数而非全局变量，PASS6 无法直接追溯到根 GV。Inline 后 GEP 链展开为 `@g_cellCfg → GEP 0, constIdx → ...`，可被 PASS6 正常处理。

> **为什么 InstCombine 在 PASS6 之前**: 参数替换 (`replaceAllUsesWith(ConstantInt)`) 后，`zext i8 3 to i64` 等指令仍是变量形式。InstCombine 将它们折叠为常量，使 `accumulateConstantOffset` 计算可靠。

**JIT 编译流程** (ejit_compile_or_get 内部):

```cpp
CompiledFunction EJitCompiler::compile(Module& M, Function* targetFunc,
                                        const SpecializationContext& ctx) {
    // 1. 参数预处理: 替换 ejit_period_arr_ind 参数为实际值
    for (int i = 0; i < ctx.period_count; ++i) {
        unsigned paramIdx = findParamIndex(targetFunc, ctx.dimensions[i].periodName);
        Argument* arg = targetFunc->getArg(paramIdx);
        Constant* constVal = ConstantInt::get(arg->getType(), ctx.dimensions[i].cellIdx);
        arg->replaceAllUsesWith(constVal);
    }

    // 2. InstCombine: 折叠 zext/GEP 常量链，清理 PHI
    runInstCombine(M);

    // 3. Inline: 内联被调用函数，展开跨函数 GEP 链
    runInline(M);

    // 4. 结构体字段特化 (FunctionPass, 逐函数处理)
    EJitStructFieldPass structField(reg);
    FunctionAnalysisManager FAM;
    for (Function &F : M.functions()) {
      if (!F.isDeclaration())
        structField.run(F, FAM);
    }

    // 5. 标准 LLVM 优化 (按等级编排)
    runOptimizationPipeline(M, ctx.optLevel);

    // 6. 生成机器码
    return codeGen(M, targetFunc);
}
```

**AOT 协调 Pass**:

```cpp
class EJitAotModulePass : public PassInfoMixin<EJitAotModulePass> {
public:
    PreservedAnalyses run(Module& M, ModuleAnalysisManager& AM) {
        EJitRegisterPeriodPass().run(M, AM);
        EJitWrapperGenPass().run(M, AM);
        EJitPeriodHandlerPass().run(M, AM);
        return PreservedAnalyses::none();
    }
    static StringRef name() { return "ejit-aot-module"; }
};
```

> **注意**: `EJitRegisterBitcodePass` 不再包含在 `EJitAotModulePass` 内。它作为独立的早期 Pass，在标准优化 Pipeline 之前单独注册。

### 3.7 异步编译模型 (对应 SPEC3 §4.3)

```
同步模式 (EJIT_COMPILE_SYNC):
  ejit_compile_or_get() → 查Cache → 未命中 → 阻塞JIT编译 → 存Cache → 返回pfn
  首次调用有延迟，后续调用命中Cache

异步模式 (EJIT_COMPILE_ASYNC):
  ejit_compile_or_get() → 查Cache → 未命中 → 提交编译请求 → 立即返回NULL
  后台线程完成编译 → 存入Cache
  下次调用命中Cache → 返回pfn
```

**异步编译组件**:

```cpp
// 编译请求
struct CompileRequest {
    std::string funcName;
    std::vector<std::pair<std::string, int>> dims; // (periodName, cellIdx)
    std::string cacheKey;
};

// 异步编译器
class EJitAsyncCompiler {
    std::thread workerThread_;
    std::queue<CompileRequest> requestQueue_;
    std::mutex queueMutex_;
    std::condition_variable queueCV_;
    std::atomic<bool> running_{false};

public:
    void start();
    void stop();
    void submitRequest(const CompileRequest& req);

private:
    void workerLoop(); // 取请求 → compile() → cache_.put()
};
```

**ejit_compile_or_get 行为差异**:

| 步骤 | 同步模式 | 异步模式 |
|------|---------|---------|
| 查 Cache | 命中则返回 pfn | 命中则返回 pfn |
| 未命中 | 阻塞执行 JIT 编译，返回 pfn | 提交异步请求，返回 NULL |
| 后续调用 | 命中 Cache | 命中 Cache (后台已完成编译) |
| Fallback | 编译失败返回 NULL | 首次返回 NULL，fallback 到 AOT |

**线程安全要求**: Cache 的 get/put 操作需加锁或使用并发安全数据结构。

## 4. 接口定义

### 4.1 C++ 主 API

```cpp
// include/llvm/ExecutionEngine/EJIT/EJit.h

namespace llvm::ejit {

// 编译模式
enum class CompileMode {
    Sync,    // 同步：阻塞等待 JIT 完成
    Async    // 异步：立即返回，后台编译
};

// 初始化配置
struct Config {
    CompileMode compileMode = CompileMode::Sync; // 编译模式
    size_t maxCacheSize = 512 * 1024;     // 默认 512KB
    size_t maxSingleFunctionSize = 64 * 1024; // 默认 64KB
    size_t maxCacheEntries = 100;         // 最大缓存条目数
    OptimizationLevel optLevel = OptimizationLevel::Level2; // 默认优化等级
    bool enableLogging = true;
    std::string logLevel = "info";        // debug/info/warn/error
};

// 主类
class EJit {
public:
    // 构造函数
    explicit EJit(const Config& config = Config{});
    ~EJit();

    // 生命周期 API (用户可见，对应 SPEC4.md 3.3)
    // 激活指定时间窗实例，记录状态变更
    void activate(const char* periodName, int cellIdx);
    void deactivate(const char* periodName, int cellIdx);

    // 激活数组指定实例 (显式传入数组指针)
    void activate_array(const char* periodName, void* arrayPtr, int cellIdx);
    void deactivate_array(const char* periodName, void* arrayPtr, int cellIdx);

    // 激活时间窗所有实例
    void activate_all(const char* periodName);
    void deactivate_all(const char* periodName);

    // 预热：提前编译指定实例 (可选优化)
    void warmup(const char* periodName, int cellIdx);

    // 清除缓存
    void clearCache();
    void invalidate(const char* periodName, int cellIdx);

    // 获取统计信息
    struct Stats {
        size_t cacheHits;
        size_t cacheMisses;
        size_t totalCompileTimeMs;
        size_t currentCacheSize;
    };
    Stats getStats() const;

    // 设置优化等级 (运行时动态调整)
    void setOptimizationLevel(OptimizationLevel level);

    // 获取最近错误
    const Error& getLastError() const;

private:
    // 内部实现...
};

// 错误类型
enum class ErrorCode {
    Success = 0,
    BitcodeLoadFailed,
    SpecializationFailed,
    OptimizationFailed,
    OutOfMemory,
    NullPointer,
    InvalidCellIdx,
    CodeSizeExceeded
};

// 错误信息
struct Error {
    ErrorCode code;
    std::string message;
    std::string details;  // 详细堆栈信息
};

} // namespace llvm::ejit
```

**API 说明** (SPEC4.md 3.3):

| API                                               | 说明                               |
| ------------------------------------------------- | -------------------------------- |
| `activate(periodName, cellIdx)`                   | 激活指定时间窗实例，记录基地址映射                |
| `deactivate(periodName, cellIdx)`                 | 去激活指定时间窗实例，触发缓存失效                |
| `activate_array(periodName, arrayPtr, cellIdx)`   | 激活指定时间窗名称下指定数组的指定实例              |
| `deactivate_array(periodName, arrayPtr, cellIdx)` | 标记指定时间窗名称下指定数组的指定实例进入 invalid 状态 |
| `activate_all(periodName)`                        | 激活指定时间窗名称下的所有数组实例                |
| `deactivate_all(periodName)`                      | 标记指定时间窗名称下的所有数组实例进入 invalid 状态   |
| `warmup(periodName, cellIdx)`                     | 预热：提前编译指定实例，减少首次调用延迟             |
| `clearCache()`                                    | 清除所有缓存                           |
| `invalidate(periodName, cellIdx)`                 | 使指定实例的缓存失效                       |
| `getStats()`                                      | 获取缓存命中率、编译时间等统计信息                |
| `setOptimizationLevel()`                          | 运行时动态调整优化等级                      |

### 4.2 C 运行时 API

```c
// include/llvm/ExecutionEngine/EJIT/EJitRuntime.h

#ifdef __cplusplus
extern "C" {
#endif

// 错误码 (对应 SPEC3 §3.1)
typedef enum {
    EJIT_OK = 0,
    EJIT_ERR_INVALID_PARAM = -1,
    EJIT_ERR_NOT_ACTIVE = -2,
    EJIT_ERR_COMPILE_FAILED = -3,
    EJIT_ERR_CACHE_FULL = -4,
    EJIT_ERR_MEMORY = -5,
    EJIT_ERR_BITCODE_NOT_FOUND = -6
} ejit_status_t;

// 编译模式 (对应 SPEC3 §3.1)
typedef enum {
    EJIT_COMPILE_SYNC,    /* 同步编译：阻塞等待 JIT 完成 */
    EJIT_COMPILE_ASYNC    /* 异步编译：立即返回，后台编译 */
} ejit_compile_mode_t;

// 优化等级 (对应 SPEC3 §3.1)
typedef enum {
    EJIT_OPT_L1 = 1,      /* 保守：常量传播 + 死代码消除 */
    EJIT_OPT_L2 = 2,      /* 中等：+ 函数内联 + CFG 简化 */
    EJIT_OPT_L3 = 3       /* 激进：+ 循环展开 */
} ejit_opt_level_t;

// 初始化 (C 接口，对应 SPEC3 §3.2)
typedef struct {
    ejit_compile_mode_t mode;       /* 编译模式：同步/异步 */
    ejit_opt_level_t opt_level;     /* 优化等级：L1/L2/L3 */
    size_t max_cache_size;          /* 最大缓存大小 (字节) */
    /* 以下为扩展字段 (SPEC3 未定义) */
    size_t max_single_function_size; /* 单函数最大代码量 */
    unsigned int max_cache_entries;  /* 最大缓存条目数 */
    int enable_logging;             /* 是否启用日志 */
} ejit_config_t;

// 初始化/销毁
ejit_status_t ejit_init(const ejit_config_t* config);
void ejit_shutdown(void);

// 编译模式控制 (对应 SPEC3 §3.2)
void ejit_set_compile_mode(ejit_compile_mode_t mode);
ejit_compile_mode_t ejit_get_compile_mode(void);

// 生命周期 API (用户可见，对应 SPEC4.md 3.3)
void ejit_activate(const char* periodName, int cellIdx);
void ejit_deactivate(const char* periodName, int cellIdx);

ejit_status_t ejit_activate_array(const char* periodName, void* arrayPtr, int cellIdx);
void ejit_deactivate_array(const char* periodName, void* arrayPtr, int cellIdx);

ejit_status_t ejit_activate_all(const char* periodName);
void ejit_deactivate_all(const char* periodName);

bool ejit_is_active(const char* periodName, int cellIdx);

// 维度信息结构 (用于 ejit_compile_or_get)
// 包含维度名称和索引值，用于 JIT 特化
typedef struct {
    const char* name;    /* 维度名称，如 "cell", "trp" */
    uint8_t index;       /* 参数值 */

// 内部使用: Wrapper 入口和编译函数
// 返回特化函数指针，NULL 表示编译失败继续执行原函数逻辑
// cacheKey = (funcIdx << 32) | dims
// out_pfn 保留作为 future 扩展，用于跨平台适配或状态查询
void* ejit_compile_or_get(uint64_t cacheKey,
                          int count, void** out_pfn);

// 缓存管理
void ejit_clear_cache(void);
void ejit_invalidate(const char* periodName, int cellIdx);

// 统计信息
typedef struct {
    size_t cacheHits;
    size_t cacheMisses;
    size_t totalCompileTimeMs;
    size_t currentCacheSize;
} ejit_stats_t;

int ejit_get_stats(ejit_stats_t* stats);

// 错误信息
typedef struct {
    int code;
    const char* message;
} ejit_error_t;

const ejit_error_t* ejit_get_last_error(void);

#ifdef __cplusplus
}
#endif
```

### 4.3 LLVM Pass 接口

根据 §3.6 的 Pass 架构，定义各 Pass 接口：

```cpp
// include/llvm/ExecutionEngine/EJIT/EJitPasses.h

#include <llvm/IR/PassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Function.h>

namespace llvm {

// 前置声明
class SpecializationContext;

// ===== AOT Pipeline Passes =====

// ===== 早期 AOT Pipeline (标准优化之前) =====

// EJitRegisterBitcodePass - 在标准优化前提取 ejit_entry 函数 bitcode
// 作为独立早期 Pass 注册，不包含在 EJitAotModulePass 内
class EJitRegisterBitcodePass : public PassInfoMixin<EJitRegisterBitcodePass> {
public:
    PreservedAnalyses run(Module& M, ModuleAnalysisManager& AM);
    static StringRef name() { return "ejit-register-bitcode"; }
};

// ===== AOT Pipeline (PASS2-4) =====
// 非 LTO: 在标准优化之前 (wrapper IR 被 O2 优化)
// LTO: 在标准优化之后 (LTO 已完成优化)

// 1. EJitRegisterPeriodPass - 生成 period 数组/static 变量注册代码
class EJitRegisterPeriodPass : public PassInfoMixin<EJitRegisterPeriodPass> {
public:
    PreservedAnalyses run(Module& M, ModuleAnalysisManager& AM);
    static StringRef name() { return "ejit-register-period"; }
};

// 2. EJitWrapperGenPass - 为 ejit_entry 函数生成 Wrapper 插桩代码
class EJitWrapperGenPass : public PassInfoMixin<EJitWrapperGenPass> {
public:
    PreservedAnalyses run(Module& M, ModuleAnalysisManager& AM);
    static StringRef name() { return "ejit-wrapper-gen"; }
};

// 3. EJitPeriodHandlerPass - 为 ejit_period_lc 函数插入 deactivate/activate 调用
class EJitPeriodHandlerPass : public PassInfoMixin<EJitPeriodHandlerPass> {
public:
    PreservedAnalyses run(Module& M, ModuleAnalysisManager& AM);
    static StringRef name() { return "ejit-period-handler"; }
};

// AOT 协调 Pass
class EJitAotModulePass : public PassInfoMixin<EJitAotModulePass> {
public:
    PreservedAnalyses run(Module& M, ModuleAnalysisManager& AM);
    static StringRef name() { return "ejit-aot-module"; }
};

// ===== JIT Pipeline Passes =====

// EJitStructFieldPass - 替换 load(!ejit.may_const) 为运行时常量
// 运行在 ejit_compile_or_get 内部，Inline 之后
// v1.2: may_const 识别通过 load 上的 !ejit.may_const metadata
// v1.3: 支持直接 GlobalVariable load（无 GEP），内联后的函数也可处理
class EJitStructFieldPass : public PassInfoMixin<EJitStructFieldPass> {
public:
    EJitStructFieldPass(PeriodArrayRegistry &reg);
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
private:
    PeriodArrayRegistry &registry_;
};

} // namespace llvm
```

**EJitStructFieldPass 核心工作流程**:

```
输入: 加载的 bitcode Module (参数已替换为常量 + InstCombine 已折叠 + Inline 已展开)

1. 解析 Module metadata，收集 ejit_period_arr 全局变量信息
   metadata: !ejit.period_arr = !{!"g_cellCfg", !"cell", i32 N}
   → 得到: 全局变量名 → 时间窗名称 + 数组大小

2. 遍历 Module 中所有函数的 load 指令:
   a. 检查 load->hasMetadata("ejit.may_const") → 无则跳过
   b. 从指针操作数获取 byteOffset:
      - 直接 GlobalVariable: byteOffset = 0 (标量全局变量)
      - GEPOperator: accumulateConstantOffset → byteOffset
      - 其他 (PHI/Select): 跳过 (Inline+InstCombine 已消除绝大多数)
   c. 检查根全局变量是否属于 ejit_period_arr (从 gvPeriodMap 查)

3. 替换: 对每个命中的 load:
   a. 从 PeriodArrayRegistry 获取数组 baseAddr
   b. 计算: addr = baseAddr + byteOffset
   c. 从进程内存读取实际值: value = *(*(typeof(field))*)addr
   d. 创建 Constant: 按 load->getType() 构造 ConstantInt/ConstantFP/...
   e. 替换: loadInst->replaceAllUsesWith(constant)
   f. 删除: loadInst->eraseFromParent()

输出: 所有 ejit_may_const 字段的 load 被替换为运行时常量值
```

**Pass 执行顺序** (对应 §3.6):

**PASS1 — 早期 AOT** (始终在标准优化之前):

| 顺序  | Pass                      | 说明                                           |
| --- | ------------------------- | -------------------------------------------- |
| 1   | `EJitRegisterBitcodePass` | 在标准优化前提取 ejit_entry 函数 bitcode，确保 metadata 完整 |

**PASS2-4 — AOT** (非LTO: 标准优化之前; LTO: 标准优化之后):

| 顺序  | Pass                      | 说明                                           |
| --- | ------------------------- | -------------------------------------------- |
| 1   | `EJitRegisterPeriodPass`  | 生成 period 数组/static 变量注册代码                    |
| 2   | `EJitWrapperGenPass`      | 为 ejit_entry 函数生成 Wrapper 插桩代码               |
| 3   | `EJitPeriodHandlerPass`   | 为 ejit_period_lc 函数插入 deactivate/activate 调用 |

**JIT Pipeline**:

| 顺序  | 步骤                    | 说明                                                     |
| --- | --------------------- | ------------------------------------------------------ |
| 1   | 参数预处理                 | 替换 ejit_period_arr_ind 参数为实际值 (非 Pass)                 |
| 2   | InstCombine           | 折叠 zext/GEP 常量链，清理 PHI 节点                             |
| 3   | Inline                | 内联被调用函数，展开跨函数 GEP 链                                    |
| 4   | `EJitStructFieldPass` | 替换 ejit_may_const 字段 load 为运行时常量                       |
| 5   | 标准 LLVM 优化            | 按 L1/L2/L3 等级运行 SCCP、DCE、CFGSimplify、LoopUnroll         |

### 4.4 Clang 属性接口

根据 SPEC4.md 3.2 的属性定义，实现以下 6 种属性：

> **注意**: 以下为伪代码，实际 TableGen 语法使用 `def` 定义，Spellings/Subjects 语法有差异。

```cpp
// include/clang/Basic/Attr.td - Attribute definitions (伪代码)

// 1. ejit_may_const - 结构体成员属性
class EjitMayConst : InheritableAttr {
    string Spellings = [Clang<"ejit_may_const">];
    string Subjects = [SubjectList<"FieldDecl", Error>];
    string DocComment = "标记结构体成员在时间窗内可视为常量";
};

// 2. ejit_period - 全局变量时间窗标记
class EjitPeriod : InheritableAttr {
    string Spellings = [Clang<"ejit_period">];
    string Subjects = [SubjectList<"VarDecl", Error>];

    // 时间窗名称参数 (如 "static", "cell", "trp")
    ArgumentKind Kind = String;

    // 判断是否为 static 时间窗 (运行期不变)
    bool isStatic() const { return getArgument() == "static"; }

    string DocComment = "定义全局变量所属的时间窗";
};

// 3. ejit_period_arr - 全局数组时间窗标记
class EjitPeriodArr : InheritableAttr {
    string Spellings = [Clang<"ejit_period_arr">];
    string Subjects = [SubjectList<"VarDecl", Error>];

    // 时间窗数组名称 (如 "cell", "trp")
    ArgumentKind Kind = String;

    std::string getPeriodName() const { return getArgument(); }

    string DocComment = "定义全局数组所属的时间窗数组，按索引独立控制";
};

// 4. ejit_period_arr_ind - 函数参数特化维度标记
class EjitPeriodIdx : InheritableAttr {
    string Spellings = [Clang<"ejit_period_arr_ind">];
    string Subjects = [SubjectList<"ParmVarDecl", Error>];

    // 关联的时间窗数组名称 (如 "cell", "trp")
    ArgumentKind Kind = String;

    std::string getPeriodName() const { return getArgument(); }

    string DocComment = "标记 JIT 特化维度参数，关联对应时间窗数组";
};

// 5. ejit_entry - JIT 优化入口函数标记
class EjitEntry : InheritableAttr {
    string Spellings = [Clang<"ejit_entry">];
    string Subjects = [SubjectList<"FunctionDecl", Error>];

    string DocComment = "标记函数将进行 JIT 优化";

    // 限制：不支持递归
    bool isRecursive() const;
};

// 6. ejit_period_lc - 时间窗生命周期管理函数标记
class EjitPeriodLc : InheritableAttr {
    string Spellings = [Clang<"ejit_period_lc">];
    string Subjects = [SubjectList<"FunctionDecl", Error>];

    // 管理的时间窗名称
    ArgumentKind Kind = String;

    std::string getPeriodName() const { return getArgument(); }

    string DocComment = "标记时间窗生命周期管理函数，表示该函数会修改指定时间窗内的数据";
};
```

---

### 属性处理流程

```cpp
// clang/lib/Sema/SemaEJIT.cpp - 属性语义检查

namespace clang::sema {

class EJitAttributeHandler {
public:
    // 处理 ejit_may_const
    void HandleEjitMayConst(Decl* D, const ParsedAttr& AL) {
        // 1. 检查是否标记在 FieldDecl 上
        // 2. 检查类型：仅支持整型、布尔型、浮点型、嵌套结构体
        // 3. 检查是否为 volatile (volatile 不视为常量)
    }

    // 处理 ejit_period
    void HandleEjitPeriod(Decl* D, const ParsedAttr& AL) {
        // 1. 检查是否标记在 VarDecl 上
        // 2. 检查是否为数组 (数组应使用 ejit_period_arr)
        // 3. 提取时间窗名称
        // 4. 检查 static 时间窗的特殊规则
    }

    // 处理 ejit_period_arr
    void HandleEjitPeriodArr(Decl* D, const ParsedAttr& AL) {
        // 1. 检查是否标记在 VarDecl 上
        // 2. 检查是否为数组类型
        // 3. 提取时间窗数组名称
        // 4. 注册到时间窗数组表
    }

    // 处理 ejit_period_arr_ind
    void HandleEjitPeriodIdx(Decl* D, const ParsedAttr& AL) {
        // 1. 检查是否标记在 ParmVarDecl 上
        // 2. 检查类型：必须为整数类型
        // 3. 提取关联的时间窗数组名称
        // 4. 验证该时间窗数组已定义
    }

    // 处理 ejit_entry
    void HandleEjitEntry(Decl* D, const ParsedAttr& AL) {
        // 1. 检查是否标记在 FunctionDecl 上
        // 2. 检查递归调用
        // 3. 生成元数据 (bitcode 收集)
    }

    // 处理 ejit_period_lc
    void HandleEjitPeriodLc(Decl* D, const ParsedAttr& AL) {
        // 1. 检查是否标记在 FunctionDecl 上
        // 2. 检查是否有对应的 ejit_period_arr_ind 参数
        // 3. 提取管理的时间窗名称
    }
};

} // namespace clang::sema
```

---

### IR 元数据生成

```cpp
// clang/lib/CodeGen/CGEJIT.cpp - 属性到 LLVM IR 的转换

namespace clang::cg {

class CodeGenEJIT {
public:
    // 生成 ejit_may_const metadata
    void emitMayConstMetadata(llvm::Module& M, const FieldDecl* FD) {
        // 生成类似：!ejit.may_const = !{!{i32 偏移，字符串 "fieldName"}}
    }

    // 生成 ejit_period metadata
    void emitPeriodMetadata(llvm::Module& M, const VarDecl* VD) {
        // 生成类似：!ejit.period = !{!{字符串 "varName", 字符串 "periodName"}}
    }

    // 生成 ejit_period_arr metadata
    void emitPeriodArrMetadata(llvm::Module& M, const VarDecl* VD) {
        // 生成类似：!ejit.period_arr = !{!{字符串 "varName", 字符串 "periodName", i32 数组大小}}
    }

    // 生成 ejit_period_arr_ind metadata
    void emitPeriodIdxMetadata(llvm::Module& M, const FunctionDecl* FD) {
        // 生成类似：!ejit.period_idx = !{!{字符串 "funcName", i32 参数索引，字符串 "periodName"}}
    }

    // 生成 ejit_entry metadata
    void emitEntryMetadata(llvm::Module& M, const FunctionDecl* FD) {
        // 生成类似：!ejit.entry = !{!{字符串 "funcName", !bitcode_ref}}
    }

    // 生成 ejit_period_lc metadata
    void emitPeriodLcMetadata(llvm::Module& M, const FunctionDecl* FD) {
        // 生成类似：!ejit.period_lc = !{!{字符串 "funcName", 字符串 "periodName"}}
    }
};

} // namespace clang::cg
```

---

### 属性约束检查表

| 属性                          | 可标记位置         | 类型约束           | 特殊规则                                     |
| --------------------------- | ------------- | -------------- | ---------------------------------------- |
| `ejit_may_const`            | FieldDecl     | 整型、布尔、浮点、嵌套结构体 | volatile 不视为常量                           |
| `ejit_period(static)`       | VarDecl (非数组) | 任意结构体/基本类型     | static 时间窗永远激活                           |
| `ejit_period_arr(name)`     | VarDecl (数组)  | 数组类型           | 数组大小固定且 < 100                            |
| `ejit_period_arr_ind(name)` | ParmVarDecl   | 整数类型           | 最多 4 个参数，需验证 name 对应已定义的 ejit_period_arr |
| `ejit_entry`                | FunctionDecl  | 函数             | 不支持递归                                    |
| `ejit_period_lc(name)`      | FunctionDecl  | 函数             | 必须有对应的 ejit_period_arr_ind 参数            |

### 防呆设计实现方案 (对应 SPEC3 §9)

| 检查项 | 级别 | 实现位置 | 说明 |
|--------|------|---------|------|
| 全局变量归属冲突 | 错误 (阻止编译) | Clang Sema (`SemaEJIT.cpp`) | 同一 VarDecl 被多个 `ejit_period`/`ejit_period_arr` 标记时报错 |
| 时间窗修改点检测 | 警告 | Clang Sema (`SemaEJIT.cpp`) | 修改 `ejit_may_const` 字段但所在函数未标记 `ejit_period_lc` 时发出警告 |
| 生命周期调用链检测 | 暂不支持 | — | SPEC3 §9.3 明确标注暂不支持 |
| 元数据一致性检测 | 警告 | LLVM Pass (`EJitAotModulePass` 末尾) | 检查 `ejit_entry` 函数实际引用的全局变量是否与其声明的 `ejit_period_arr_ind` 一致 |

**实现要点**:

1. **归属冲突**: 在 `HandleEjitPeriod` / `HandleEjitPeriodArr` 中维护已注册变量表，重复注册时报错
2. **修改点检测**: 在 AST 遍历中识别对 `ejit_may_const` 字段的写操作，检查所在函数是否标记了 `ejit_period_lc`
3. **元数据一致性**: Pass 阶段分析 `ejit_entry` 函数的 IR，收集实际引用的 `@global_var`，与 metadata 中声明的依赖做交叉校验

---

## 5. 实施阶段

### 阶段 1: 基础框架 (4 周)

**目标**: 建立项目骨架，实现核心数据结构和基础 API

| 周次  | 任务                           | 交付物                 | 复用/新增 |
| --- | ---------------------------- | ------------------- | ----- |
| 1   | 项目目录结构搭建，CMake 构建配置          | 目录结构、CMakeLists.txt | 新增    |
| 2   | Config、Error、Logger 基础组件     | 基础组件代码              | 新增    |
| 3   | EJit 主类骨架、C++ API 定义         | EJit.h/cpp 骨架       | 新增    |
| 4   | C 运行时接口定义和实现 (含 6 个生命周期 API) | EJitRuntime.h/cpp   | 新增    |

**里程碑**: 编译通过，基础框架可运行

---

### 阶段 2: Clang 属性支持 (3 周)

**目标**: 在 Clang 中实现 6 种属性的解析和代码生成

| 周次  | 任务                                                                                          | 交付物             |
| --- | ------------------------------------------------------------------------------------------- | --------------- |
| 5   | 定义 6 种属性类 (EjitMayConst, EjitPeriod, EjitPeriodArr, EjitPeriodIdx, EjitEntry, EjitPeriodLc) | 属性类定义 (Attr.td) |
| 6   | 属性解析 (Sema) 和语义检查 (ejit_period_arr_ind 关联验证，ejit_period_lc 参数检查)                            | SemaEJIT.cpp    |
| 7   | 属性到 LLVM IR 的 codegen (metadata 生成)                                                         | CGEJIT.cpp      |

**里程碑**: 属性可正确编译，IR 中可见 metadata

---

### 阶段 3: LLVM Pass 与 JIT 核心引擎 (6 周)

**目标**: 实现 AOT 4 Pass + JIT 1 Pass 和 JIT 编译核心逻辑

| 周次  | 任务                                                   | 交付物                   | 复用/新增      |
| --- | ---------------------------------------------------- | --------------------- | ---------- |
| 8   | Bitcode 加载器实现，按需加载                                   | EJitModuleLoader.cpp  | **新增** |
| 9   | EJitWrapperGenPass 实现 (为 ejit_entry 生成 Wrapper)      | WrapperGenPass.cpp    | **新增**     |
| 10  | EJitPeriodHandlerPass 实现 (插入 activate/deactivate 调用) | PeriodHandlerPass.cpp | **新增**     |
| 11  | EJitRegisterBitcodePass + EJitRegisterPeriodPass 实现  | RegisterPasses.cpp    | **新增** |
| 12  | EJitStructFieldPass 实现 (g_pdc[idx].field 替换为常量)      | StructFieldPass.cpp   | **新增**     |
| 13  | 优化 pipeline 实现与 L1/L2/L3 配置                        | EJitOptimizer.cpp     | **新增**     |

**里程碑**: 端到端 JIT 编译流程可工作

---

### 阶段 4: Code Cache 管理 (3 周)

**目标**: 实现代码缓存、LRU 淘汰、统计功能 (支持复合键)

| 周次  | 任务                            | 交付物                  | 复用/新增         |
| --- | ----------------------------- | -------------------- | ------------- |
| 14  | 缓存数据结构 (复合键：fnName            | periodName=cellIdx)  | EJitCache.cpp |
| 15  | LRU 淘汰策略 + 大小限制               | 淘汰逻辑                 | **新增**        |
| 16  | 统计信息、运行时状态管理 (activate/deactivate 状态跟踪) | Stats API、Runtime 状态 | **新增**        |

**里程碑**: 缓存功能完整，LRU 正确工作

---

### 阶段 5: 错误处理与容错 (2 周)

**目标**: 实现 fallback 机制和日志系统

| 周次  | 任务                  | 交付物    |
| --- | ------------------- | ------ |
| 17  | 错误分类、处理、fallback 逻辑 | 错误处理系统 |
| 18  | 日志系统集成，调试信息输出       | 完整日志系统 |

**里程碑**: JIT 失败时平滑降级到 AOT

---

### 阶段 6: 测试与集成 (4 周)

**目标**: 单元测试、集成测试、回归测试

| 周次  | 任务                            | 交付物    |
| --- | ----------------------------- | ------ |
| 19  | 单元测试：Cache、Specializer、Passes | 单元测试代码 |
| 20  | 集成测试: 完整 JIT 流程 (含多维度多数组)     | 集成测试   |
| 21  | 回归测试: JIT vs AOT 一致性          | 回归测试框架 |
| 22  | 性能测试: 优化效果评估                  | 性能测试报告 |

**里程碑**: 测试覆盖核心路径，功能稳定

---

### 阶段 7: 示例与文档 (2 周)

**目标**: 完善示例代码和设计文档

| 周次  | 任务                    | 交付物            |
| --- | --------------------- | -------------- |
| 23  | demo 示例编写 (含多维度多数组场景) | demo.c 完整示例    |
| 24  | 设计文档完善                | docs/design.md |

**里程碑**: 可交付状态

---

## 6. 依赖关系

```
阶段 1 ─────┬─────> 阶段 2
            │
            ├─────> 阶段 3
            │
            ├─────> 阶段 4
            │
            └─────> 阶段 5
                         │
                         └─────> 阶段 6 ─────> 阶段 7
```

- 阶段 1 是所有阶段的前置依赖
- 阶段 2 (属性) 是阶段 3 (Pass) 的前置依赖
- 阶段 3/4/5 可并行开发
- 阶段 6 依赖所有前序阶段

---

## 7. 关键风险与缓解

| 风险               | 影响    | 缓解措施     |
| ---------------- | ----- | -------- |
| LLVM Pass 开发复杂度  | 进度延迟  | 预留 1 周缓冲 |
| Bitcode 兼容性      | 运行时错误 | 严格版本检查   |
| 内存超出限制           | 系统不稳定 | 严格淘汰策略   |
| 优化导致代码膨胀         | 内存溢出  | 单函数大小限制  |
| 与 easyJIT 代码复用冲突 | 集成问题  | 明确接口边界   |

---

*文档版本: 2.1*
*创建日期: 2026-04-15*
*基于版本: 2.0*
