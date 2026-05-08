# EmbeddedJIT X86 Mini: LLVM C API 运行时方案

> **目标**: 用 LLVM C API 重写 EJIT 运行时（纯 C），先 X86 验证，再移植 ARM bare-metal。
> **核心问题**: LLVM 是 C++ 编译的，裸核环境需要提供 libc++/libc++abi 的子集。

## 1. 架构

```
┌──────────────────────────────────────────────────┐
│              EJIT 应用（C 代码）                    │
│  ejit_init() / ejit_activate() / ejit_compile()   │
└──────────────────┬───────────────────────────────┘
                   │ 纯 C 接口
┌──────────────────▼───────────────────────────────┐
│         EJIT C Runtime (libejit_c.a)              │
│  调用 LLVM C API: llvm-c/LLJIT.h, Orc.h, Core.h   │
│  手写 C Pass: 常量替换 / InstCombine / Inline       │
└──────────────────┬───────────────────────────────┘
                   │ C ABI
┌──────────────────▼───────────────────────────────┐
│            LLVM 静态库 (C++ 编译)                   │
│  libLLVMCore.a libLLVMOrcJIT.a libLLVMCodeGen.a   │
│  ... + libc++.a + libc++abi.a + libunwind.a       │
└──────────────────────────────────────────────────┘
```

**关键**: EJIT C Runtime 是纯 C 代码，调用 LLVM C API。LLVM 库本身是 C++ 编译的预构建 `.a`，部署时作为系统依赖提供。

## 2. C API 能力映射

### 2.1 LLVM C API 覆盖范围

| EJIT 当前使用的功能 | LLVM C API | 可用性 |
|------|------|------|
| 创建 LLJIT | `LLVMOrcCreateLLJIT()` | `llvm-c/LLJIT.h` |
| 加载 bitcode Module | `LLVMParseBitcode2()` → `LLVMOrcLLJITAddLLVMIRModule()` | `llvm-c/BitReader.h` |
| IR Transform Layer | `LLVMOrcLLJITGetIRTransformLayer()` + `LLVMOrcIRTransformLayerSetTransform()` | `llvm-c/Orc.h` |
| 符号查找 | `LLVMOrcLLJITLookup()` | `llvm-c/LLJIT.h` |
| 运行 Pass | `LLVMRunPasses()` | `llvm-c/Transforms/PassBuilder.h` |
| 创建常量 | `LLVMConstInt()`, `LLVMConstReal()` | `llvm-c/Core.h` |
| 替换指令 | `LLVMReplaceAllUsesWith()` | `llvm-c/Core.h` |
| IR 遍历 (Function/BasicBlock/Instruction) | `LLVMGetFirstFunction()`, `LLVMGetFirstBasicBlock()`, `LLVMGetFirstInstruction()` | `llvm-c/Core.h` |

### 2.2 C API 缺失的能力

以下功能 LLVM C API **未暴露**，需要在 EJIT C Runtime 中自行实现：

| 功能 | 当前实现 | C API 替代方案 |
|------|---------|---------------|
| `EJitStructFieldPass` (may_const load→常量) | C++ FunctionPass, 访问 PeriodArrayRegistry | **用 C 遍历 IR 指令, 手动替换 load** |
| Metadata 解析 (`!ejit.may_const`, `!ejit.metadata`) | `MDNode::getOperand()` | `LLVMGetMetadata()` / `LLVMGetMDKindID()` |
| GEP 链追踪 (计算字段偏移) | `GetElementPtrInst` 递归分析 | 手动遍历 GEP operand 链 |
| OptimizationLevel enum 控制 Pass 管线 | `PassBuilder::buildFunctionSimplificationPipeline()` | `LLVMRunPasses()` 用字符串描述管线 |

**结论**: 特化管线（参数替换 → InstCombine → Inline → StructFieldPass → 标准优化）的核心逻辑需要用 C 重新实现。但工作量可控——总共约 500-800 行 C 代码（`EJitStructFieldPass` 逻辑清晰：扫描 metadata → 追踪 GEP → 读运行时值 → 替换 load）。

### 2.3 Transform Callback 设计

```c
// EJIT C Runtime 的核心：IR 变换回调
static LLVMOrcThreadSafeModuleRef
ejit_transform(void *ctx, LLVMOrcThreadSafeModuleRef tsm,
               LLVMOrcMaterializationResponsibilityRef mr) {
    LLVMModuleRef mod = LLVMOrcThreadSafeModuleGetModule(tsm);

    // 1. 参数替换: ejit_period_arr_ind → const
    ejit_c_replace_period_indices(mod, (ejit_ctx_t*)ctx);

    // 2. InstCombine (走 LLVM C API 的 RunPasses)
    LLVMRunPasses(mod, "instcombine", NULL, NULL, NULL);

    // 3. Inline
    LLVMRunPasses(mod, "always-inline", NULL, NULL, NULL);

    // 4. StructFieldPass: may_const load → 常量
    ejit_c_replace_may_const_loads(mod, (ejit_ctx_t*)ctx);

    // 5. 标准优化管线
    const char *pipeline = "sccp,adce,simplifycfg,mem2reg,loop-unroll";
    LLVMRunPasses(mod, pipeline, NULL, NULL, NULL);

    return tsm;
}
```

## 3. libc++.a 依赖分析

### 3.1 符号来源分类

通过对当前 LLVMEJIT + 所有依赖库的 `nm --undefined-only` 分析，共 89 个 C++ 运行时符号，分为 4 类：

#### A. libc++abi 符号 (5个)

| 符号 | 提供方 | 说明 |
|------|--------|------|
| `__cxa_guard_acquire` / `__cxa_guard_release` | `cxa_guard.o` | 静态局部变量线程安全初始化 |
| `__cxa_atexit` / `__cxa_finalize` | `cxa_atexit.o` | 全局对象析构注册 |
| `__cxa_pure_virtual` | `cxa_pure_virtual.o` | 纯虚函数调用 |

**裸核替代**: 
- `cxa_guard` → 单线程环境下可用简单 flag 替换（50 行）
- `cxa_atexit` → 裸核无 exit 概念，空实现
- `cxa_pure_virtual` → `abort()` 或死循环

#### B. libc++ 容器/工具类符号 (~30个)

| 符号类别 | 示例 | libc++ 中的 .o |
|---------|------|---------------|
| `std::ostream` | `operator<<(int)`, `tellp()` | `iostream.o`, `ios.o`, `locale.o` |
| `std::string` / `std::__string` | string 的 refcount / SSO 操作 | `string.o` |
| `std::_Rb_tree_*` (std::map/set) | 红黑树节点操作 | `algorithm.o` → 通常在头文件中 |
| `std::__detail::_List_node_base` (std::list/unordered_map) | 链表钩子 | `algorithm.o` |
| `std::chrono` | `steady_clock::now()`, `system_clock::now()` | `chrono.o` |
| `std::random_device` | `_M_fini()`, `_M_getval()` | `random.o` |
| `std::thread` | `join()`, `detach()`, `hardware_concurrency()` | `thread.o` |
| `std::condition_variable` | `wait()`, `notify_all()` | `condition_variable.o` |
| `std::mutex` / `std::lock` | (header-only, 但依赖 pthread) | — |
| `std::future` / `std::promise` | `future_category()`, `_Result_base` | `future.o` |

#### C. libc++abi 异常相关符号 (10个)

| 符号 | 说明 |
|------|------|
| `std::__throw_bad_alloc()` | operator new 失败 |
| `std::__throw_bad_array_new_length()` | new[] 失败 |
| `std::__throw_bad_function_call()` | std::function 空调用 |
| `std::__throw_system_error(int)` | std::system_error |
| `std::__throw_future_error(int)` | std::future_error |
| `std::terminate()` | 未捕获异常 |
| `std::rethrow_exception()` | 异常重抛 |
| `std::__exception_ptr::*` (2个) | exception_ptr 引用计数 |

**LLVM 使用了 `-fno-exceptions`**，这些符号理论上不应被调用。但 `std::__throw_bad_alloc` 可能被 operator new 路径引用（LLVM 使用 `new (std::nothrow)` 或自定义分配器）。`std::terminate` 是 `-fno-exceptions` 下的异常兜底。

**裸核策略**: 全部提供空实现 → `abort()` 或死循环。正常路径不会走到这里。

#### D. pthread 符号 (17个)

| 符号 | 说明 |
|------|------|
| `pthread_mutex_lock/unlock` | std::mutex 底层 |
| `pthread_rwlock_*` (3个) | 读写锁 |
| `pthread_create/join/detach` | std::thread 底层 |
| `pthread_once` | call_once |
| `pthread_self` | this_thread::get_id |
| `pthread_attr_*` (3个) | 线程属性 |
| `pthread_setname/getname` (2个) | 线程命名 |
| `pthread_setschedparam` | 线程调度 |
| `pthread_sigmask` | 信号屏蔽 |

**裸核替代策略**:
- 如果不需要异步编译 → 禁用线程，所有 pthread 符号空实现
- `LLVMOrcLLJITBuilder` 可设置 `setNumCompileThreads(0)` → 单线程编译
- `pthread_once` → 裸核单线程无需实现

### 3.2 libc++.a 需要链接的 .o 文件

```
# libc++ 核心 (必选)
string.o              # std::string
iostream.o            # std::ostream (LLVM 日志/错误输出)
ios.o                 # iostream 基础
locale.o              # std::locale (iostream 依赖)
chrono.o              # std::chrono::clock
random.o              # std::random_device (LLVM 随机数生成)
condition_variable.o  # std::condition_variable
mutex.o               # std::mutex (header-only, 但可能需要 .o)
thread.o              # std::thread
future.o              # std::future/promise

# libc++ 可选 (如裁掉日志功能可省略)
fstream.o             # 文件流 (仅调试用)
sstream.o             # 字符串流

# libc++abi (必选)
cxa_guard.o           # 静态局部变量初始化
cxa_atexit.o          # 全局析构注册
cxa_pure_virtual.o    # 纯虚函数
cxa_default_handlers.o # terminate/unexpected
cxa_handlers.o        # 异常处理器
cxa_eh_globals.o      # 异常全局变量
private_typeinfo.o    # type_info (即使 -fno-rtti)

# libc++abi (异常相关, 裸核空实现可替代)
cxa_throw.o           # 不会被调用, 空实现即可
cxa_exception.o       # 同上
cxa_vector.o          # operator new[] / delete[]
cxa_new_delete.o      # operator new / delete
```

### 3.3 裸核自实现的最小符号集

以下符号可以在裸核上手工实现（<200 行 C），不需要链接 libc++abi 的对应 .o：

```c
// ejit_baremetal_stubs.c

// ── operator new/delete ──
void *operator new(size_t sz) { return malloc(sz); }
void *operator new[](size_t sz) { return malloc(sz); }
void operator delete(void *p) noexcept { free(p); }
void operator delete[](void *p) noexcept { free(p); }

// ── cxa_guard (单线程简化版) ──
extern "C" int __cxa_guard_acquire(uint64_t *g) {
    return !(*g);  // 若未初始化则返回 1
}
extern "C" void __cxa_guard_release(uint64_t *g) {
    *g = 1;
}

// ── cxa_atexit (裸核无 exit) ──
extern "C" int __cxa_atexit(void (*f)(void*), void *p, void *d) {
    return 0;  // 不做任何事
}

// ── cxa_pure_virtual ──
extern "C" void __cxa_pure_virtual() {
    for (;;) {}  // 死循环, 或 LED 闪烁
}

// ── std::terminate ──
namespace std {
    void terminate() { for (;;) {} }
}

// ── throw 函数 (不应被调用) ──
namespace std {
    void __throw_bad_alloc() { for (;;) {} }
    void __throw_system_error(int) { for (;;) {} }
    void __throw_bad_function_call() { for (;;) {} }
    // ... 其他
}
```

## 4. 分层构建策略

```
Phase 1: X86 验证 (当前)
  ├─ BUILD_SHARED_LIBS=OFF, 静态链接
  ├─ EJIT C Runtime 调用 LLVM C API
  ├─ 链接完整 LLVM 静态库 + 系统 libc++/libpthread
  ├─ 单元测试验证功能正确性
  └─ 体积测量

Phase 2: 依赖分析
  ├─ 用 --gc-sections + -ffunction-sections 链接
  ├─ 分析 linker map file, 提取实际需要的 .o 列表
  ├─ 分析 libc++.a 实际链接了哪些 .o
  └─ 生成最小依赖清单

Phase 3: 裸核适配
  ├─ 用 arm-linux-gnueabihf- 或 aarch64-none-elf- 交叉编译
  ├─ 提供 ejit_baremetal_stubs.c (new/delete/cxa_guard/...)
  ├─ 提供 minimal libc (picolibc 或手写 memcpy/memset/malloc)
  ├─ 处理 MMU/MPU 设置 (可执行内存页)
  └─ 集成测试
```

## 5. EJIT C Runtime 文件结构

```
llvm/lib/ExecutionEngine/EJIT/
├── EJit.h / EJitRuntime.h          (现有, C++ API)
├── EJitCRuntime.h                  (新增, 纯 C API)
├── EJitOrcEngine.cpp               (现有, C++ OrcJIT wrapper)
│
├── c_runtime/                      (新增, 纯 C 运行时)
│   ├── ejit_c_runtime.c            # ejit_init / ejit_shutdown / ejit_activate
│   ├── ejit_c_compile.c            # ejit_compile_or_get → 查缓存 / 触发编译
│   ├── ejit_c_transform.c          # IR 变换回调 (参数替换 + may_const + 优化)
│   ├── ejit_c_state.c              # 激活状态管理 (periodName → cellIdx → bool)
│   ├── ejit_c_cache.c              # LRU 函数指针缓存
│   ├── ejit_c_internal.h           # 内部类型定义
│   └── ejit_c_stubs_template.c     # 裸核符号模板 (Phase 3)
│
└── CMakeLists.txt                  # 新增 libejit_c 目标
```

## 6. X86 验证计划的 CMake 改动

```cmake
# Phase 1: 构建纯 C 运行时 + 链接现有 LLVM 静态库
add_library(ejit_c STATIC
  c_runtime/ejit_c_runtime.c
  c_runtime/ejit_c_compile.c
  c_runtime/ejit_c_transform.c
  c_runtime/ejit_c_state.c
  c_runtime/ejit_c_cache.c
)

target_include_directories(ejit_c PUBLIC
  ${LLVM_MAIN_INCLUDE_DIR}
  ${LLVM_BINARY_DIR}/include
)

target_link_libraries(ejit_c PUBLIC
  LLVMCore LLVMOrcJIT LLVMJITLink LLVMBitReader
  LLVMExecutionEngine LLVMSupport LLVMPasses
  LLVMScalar LLVMInstCombine LLVMipo
  LLVMTransformUtils LLVMAnalysis
  LLVMCodeGen LLVMSelectionDAG LLVMGlobalISel
  LLVMMC LLVMTarget LLVMRuntimeDyld
  LLVMObject LLVMBinaryFormat LLVMTargetParser
)

# 单元测试 (C test, 链接 ejit_c + LLVM)
add_executable(ejit_c_test
  c_runtime/ejit_c_test.c
)
target_link_libraries(ejit_c_test ejit_c)
```

## 7. 体积预估

| 组件 | X86 (当前) | ARM (估计) |
|------|------|------|
| LLVM 核心库 (静态, --gc-sections) | ~7 MB | ~5 MB |
| LLVM CodeGen (AArch64) | — | ~2 MB |
| libc++ (最小子集) | ~200 KB | ~150 KB |
| libc++abi (最小子集) | ~50 KB | ~30 KB |
| EJIT C Runtime | ~20 KB | ~15 KB |
| **总计** | **~7.5 MB** | **~7.2 MB** |

> 注: 如果裁掉 `std::thread` (同步编译模式)、`std::ostream` (禁用日志)、`std::random_device` (用简单 LCG)，可进一步节省 ~1-2MB。

## 8. 风险与缓解

| 风险 | 缓解 |
|------|------|
| LLVM C API 对 Pass 管线不够灵活 | `LLVMRunPasses()` 支持全部标准 Pass 字符串 |
| C 实现 StructFieldPass 遗漏边界情况 | 先对照现有 C++ Pass 写 test case |
| libc++ 依赖链比预期多 | Phase 2 通过 linker map 精确分析 |
| 裸核无 MMU, 无法 mmap 可执行内存 | 用静态分配的 executable section 或 MPU 配置 |
| LLVM C API 版本不稳定 | Pin LLVM 版本, 不追 API 变更 |

---

*文档版本: 0.1*
*关联: SPEC4.md, PLAN4.md*
