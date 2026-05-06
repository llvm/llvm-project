# EmbeddedJIT 测试用例

基于 [SPEC4.md](/workspaces/jit_design_doc/SPEC4.md) 用户标注接口规范。

## 目录结构

```
ejit_test/
  README.md            — 本文件
  build_demo.sh        — 一键编译+链接+体积测量脚本
  ejit_demo.c          — 最小 EJIT Runtime C API 示例
  ejit_attr_test.c     — 6 种 EJIT 属性基础功能测试
  ejit_trace_test.c    — 运行时 trace 测试 (验证 JIT dispatch/fallback/生命周期)
  ejit_complex_test.c  — 复杂场景测试 (4 维度/多数组共享/多 return/外部输入)
```

## 环境要求

- **编译器**: `build/bin/clang` (debug 版本，支持 EJIT 属性)
- **链接库**: `build_x86/lib/*.a` (release 静态库，-Os 优化)
- 工作目录为 `/workspaces/llvm-project`

## 快速开始

```bash
# 所有命令以 /workspaces/llvm-project 为当前目录

# === 编译 + 链接 + 运行 ===

# 1. ejit_demo — 最小示例，验证 EJIT Runtime C API
build/bin/clang -O0 -c ejit_test/ejit_demo.c -o /tmp/demo.o
LIBS=$(ls build_x86/lib/*.a)
clang++ -Os -Wl,--gc-sections -Wl,--strip-all /tmp/demo.o \
  -Wl,--whole-archive $LIBS -Wl,--no-whole-archive \
  -lpthread -ldl -o /tmp/ejit_demo
/tmp/ejit_demo

# 2. ejit_attr_test — 基础属性测试 (所有 SPEC4 场景)
build/bin/clang -O2 -c ejit_test/ejit_attr_test.c -o /tmp/attr_test.o
clang++ -Os -Wl,--gc-sections -Wl,--strip-all /tmp/attr_test.o \
  -Wl,--whole-archive $LIBS -Wl,--no-whole-archive \
  -lpthread -ldl -o /tmp/ejit_attr_test
/tmp/ejit_attr_test

# 3. ejit_complex_test — 复杂场景，cellIdx 从命令行传入
build/bin/clang -O2 -c ejit_test/ejit_complex_test.c -o /tmp/complex.o
clang++ -Os -Wl,--gc-sections -Wl,--strip-all /tmp/complex.o \
  -Wl,--whole-archive $LIBS -Wl,--no-whole-archive \
  -lpthread -ldl -o /tmp/ejit_complex
/tmp/ejit_complex 0 1 2 3    # cell=0 trp=1 slice=2 carrier=3
/tmp/ejit_complex 3 0        # cell=3 trp=0 (slice/carrier 默认 0)

# 4. 仅 Sema 检查 (不链接)
build/bin/clang -fsyntax-only ejit_test/ejit_attr_test.c
build/bin/clang -fsyntax-only ejit_test/ejit_complex_test.c

# 5. 生成 IR 查看插桩结果
build/bin/clang -O2 -S -emit-llvm ejit_test/ejit_complex_test.c -o - | grep 'ejit_'
```

## 测试文件说明

### ejit_demo.c — 最小 Runtime API 示例

| 目的 | 验证 EJIT Runtime C API 基础生命周期 |
|------|-------------------------------------|
| 对应 | SPEC4 §3 (运行时 API) |
| 内容 | `ejit_init` → `ejit_activate` → `ejit_get_stats` → `ejit_shutdown` |
| 运行 | 无输出即成功 (所有 API 返回 OK) |

### ejit_attr_test.c — 属性功能测试

| 目的 | 验证 6 种 EJIT 属性在 Clang 前端+后端全流程 |
|------|---------------------------------------------|
| 对应 | SPEC4 §2 (用户标注接口), §5 (应用场景案例) |
| 覆盖 | |
| | `ejit_may_const` — 整型/布尔/浮点/嵌套结构体字段 |
| | `ejit_period("static")` — 标量全局变量时间窗 |
| | `ejit_period_arr("cell"/"trp"/"nested")` — 数组时间窗 |
| | `ejit_period_arr_ind` — 0/1/2 维度参数 |
| | `ejit_entry` — JIT 入口函数 (6 个函数) |
| | `ejit_period_lc` — 单/多时间窗生命周期函数 |
| 编译 | 全部优化等级 `-O0 -O1 -O2 -O3 -Os -Oz` |
| 验证点 | Wrapper 插桩 / may_const metadata / Bitcode 嵌入 / PASS4 生命周期 hook |

### ejit_trace_test.c — 运行时 Trace 测试

| 目的 | 带 printf trace 验证 JIT 优化流程每步正确性 |
|------|---------------------------------------------|
| 对应 | SPEC4 §3, §4 (运行时行为规格) |
| 内容 | 调用 ejit_entry → 观察 AOT fallback 结果 → 生命周期修改 → 重新调用 |
| 验证点 | 分支折叠 / dispatch-fallback 路径 / 生命周期 deactivate-activate 配对 |

### ejit_complex_test.c — 复杂场景测试

| 目的 | 边界条件 + 真实业务模式 + 外部输入 cellIdx |
|------|---------------------------------------------|
| 对应 | SPEC4 §5 场景 3, §7 约束 (4 维度上限) |
| cellIdx 来源 | `argv[1..4]` — 编译期不可知，模拟真实运行时输入 |
| 场景 | |
| | **A: 4 维度极限** — `cell + trp + slice + carrier` 同时特化 |
| | **B: 循环+switch+early return** — 遍历全部 TRP × cellType switch |
| | **C: Loop unroll 边界** — switch 内嵌 for 循环 (L3 优化) |
| | **D: 多 return 生命周期** — 3 个 return 点的 deactivate/activate |
| | **E: 仅改非 may_const 字段** — 仍触发生命周期 hook |
| 特性 | 多数组共享时间窗 / 三层结构体嵌套 / 混合类型 may_const / switch-case |

## IR 验证清单

编译后可检查以下 LLVM IR 特征：

| 检查项 | grep 命令 | 期望结果 |
|--------|----------|---------|
| Wrapper jit_entry 块 | `jit_entry:` | 每个 ejit_entry 函数都有一个 |
| jit_fallback 块 | `jit_fallback:` | 原函数体移到此块 |
| jit_dispatch 块 | `jit_dispatch:` | JIT 成功后调用特化函数 |
| ejit_compile_or_get 调用 | `ejit_compile_or_get` | 在 jit_entry 块中 |
| may_const metadata | `ejit.may_const` | 所有标记字段的 load 指令 |
| 函数/全局 metadata | `ejit.metadata` | 对应 !{!"ejit_entry"/"ejit_period"/...} |
| Bitcode 嵌入 | `ejit.bitcode` | `@__ejit_bitcode` 位于独立段 |
| 自动注册 | `ejit_auto_register` | `@llvm.global_ctors` 中 |
| 生命周期 deactivate | `ejit_deactivate_array` | 在 ejit_period_lc 函数入口 |
| 生命周期 activate | `ejit_activate_array` | 在 ejit_period_lc 函数出口 (每个 return 前) |
| 生命周期参数值 | `ejit_deactivate_array(ptr ..., ptr ..., i32 %cellIdx)` | 使用实际参数值，非硬编码 0 |

## 体积测量

```bash
# 使用 build_demo.sh
cd /workspaces/llvm-project && ./ejit_test/build_demo.sh

# 预期: ~10.2 MB (--whole-archive, -Os, --gc-sections)
# 其中 .text 段约 6.9 MB (67%)
```

## 关联文档

- [SPEC4.md](/workspaces/jit_design_doc/SPEC4.md) — 需求规格说明书
- [CLANG_ATTR_DESIGN.md](/workspaces/jit_design_doc/CLANG_ATTR_DESIGN.md) — Clang 属性实现方案
- [PASS3_EJitWrapperGen.md](/workspaces/jit_design_doc/PASS3_EJitWrapperGen.md) — 插桩 pass 设计
- [PASS4_EJitPeriodHandler.md](/workspaces/jit_design_doc/PASS4_EJitPeriodHandler.md) — 生命周期 pass 设计
