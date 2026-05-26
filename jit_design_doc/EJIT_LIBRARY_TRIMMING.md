# EmbeddedJIT 运行时库裁剪设计文档

**版本**: 3.0
**日期**: 2026-05-26
**关联**: SPEC4.md, PASS7_EJitRuntime_OrcJITLink.md
**目标**: 提供 X86_64 / AArch64 裸核环境下 EJIT 运行时的最小单一产物

---

## 1. 裸核环境约束

| 约束 | 说明 |
|---|---|
| 无 OS | 无 mmap/munmap/mprotect，无 dlopen/dlsym，无 pthread |
| 无文件系统 | 无 fopen/read/write，bitcode 从内存加载 |
| 静态链接 | 所有符号编译期确定，无动态库 |
| 单线程 | 默认同步编译模式，无后台线程 |
| 内存受限 | RAM 100 KB – 2 MB，Flash 对应中等受限 |

---

## 2. 裁剪历程与最终结果

### 2.1 总览

```
起点: 53 个 .a (117 MB)  →  44 个 .a (CMake 瘦身)  →  1 个 ejit.o (37 MB)
```

| 阶段 | 产物 | 大小 | 方法 |
|---|---|---|---|
| 原始 | 53 个 `.a` | 117 MB | 通配符全量链接 |
| Phase 1 | 44 个 `.a` | — | EJitPassBuilder + IPO/Scalar LINK_COMPONENTS 裁剪 |
| Phase 2 | 1 个 `.a` | ~99 MB | lipo extract（linker map + nm -u 依赖追踪） |
| Phase 3 | 1 个 `.a` | ~59 MB | lipo gc-merge（ld -r --gc-sections） |
| Phase 4 | 1 个 `ejit.o` | **37 MB** | ld -r -T merge.ld（段合并 + .group DISCARD） |
| 最终链接 | 二进制 | **21 MB** | `-Os` + `--gc-sections` + strip |

### 2.2 编译选项

| 选项 | 值 | 说明 |
|---|---|---|
| `CMAKE_BUILD_TYPE` | Release | |
| `CMAKE_CXX_FLAGS_RELEASE` | `-Os -DNDEBUG -ffunction-sections -fdata-sections` | 体积优先 + 段级裁剪 |
| `LLVM_TARGETS_TO_BUILD` | X86 或 AArch64 | 单架构 |
| `EJIT_BARE_METAL` | ON | 去除 mutex/chrono/logger/async |

---

## 3. EJIT 源文件裁剪

### 3.1 已删除

| 文件 | 原因 |
|---|---|
| `EJitJITLinkMemoryManager.cpp/.h` | 死代码桩，`allocate()` 调用 `report_fatal_error` |

### 3.2 宏隔离（`EJIT_BARE_METAL`）

| 文件 | 效果 |
|---|---|
| `EJitAsyncCompiler.cpp/.h` | 裸核排除了 std::thread/mutex/condition_variable |
| `EJitSyncCompiler.cpp/.h` | 只被 Async 调用，一并排除 |
| `EJitLogger.cpp` | no-op 桩 |
| `EJitCache.cpp` | std::shared_mutex → BareMetalMutex |
| `EJitRegistrationStore.cpp` | std::mutex → BareMetalMutex |
| `EJitRuntimeState.cpp` | std::mutex → BareMetalMutex |
| `EJitCompileDriver.cpp` | chrono 计时代码排除 |

### 3.3 EJitPassBuilder

创建 `EJitPassBuilder` 替代 `PassBuilder`，只注册 EJIT 需要的 ~20 个 analysis（原 ~40 个），去除 Passes 组件依赖（19 个 LINK_COMPONENTS）。

---

## 4. Lipo 裁剪管线

### 4.1 三步流程

```
extract                 gc-merge                  merge
  │                        │                        │
  │ linker map +           │ ld -r --gc-sections    │ ld -r -T merge.ld
  │ nm -u 依赖追踪          │ 死代码段消除             │ 段合并 + .group DISCARD
  │                        │                        │
  ▼                        ▼                        ▼
libejit_lipo_x86.a      libejit_lipo_x86_gc.a    ejit.o
  ~99 MB                  ~59 MB                  37 MB
  (1062 .o)               (1 merged .o)           (1 merged .o, 无 shstrtab)
```

### 4.2 使用

```bash
# 三步生成 ejit.o
python3 ejit_test/lipo/lipo.py extract  --arch=x86 --build-dir=build_release_x86_os
python3 ejit_test/lipo/lipo.py gc-merge --input=ejit_test/lipo/libejit_lipo_x86.a --build-dir=build_release_x86_os
python3 ejit_test/lipo/lipo.py merge    --input=ejit_test/lipo/libejit_lipo_x86_gc.a --build-dir=build_release_x86_os

# 测试
./ejit_test/build.sh --run --lipo
```

### 4.3 extract 原理

1. **linker map 初提取**：用 `--print-map` 生成成功链接的完整 map，匹配 `libLLVM*.a(member.o)` 模式，提取被拉入的 `.o` 文件。
2. **nm -u 依赖追踪**：对每个已提取的 `.o`，用 `nm -u` 列出未定义符号，在符号索引（`nm --print-armap` 建立，~110K 条）中查找定义者，迭代提取（通常 4-5 轮收敛）。
3. **同名冲突**：多个 `.a` 中同名 `.o`（如 `Local.cpp.o`），用 `archive__member.o` 唯一命名。

### 4.4 gc-merge 原理

`ld -r --gc-sections --entry=ejit_init -u <EJIT_API_symbols>` 对 1062 个 `.o` 做部分链接，以 `ejit_init` 为根裁剪未引用 function-sections。添加 `--allow-multiple-definition` 处理 COMDAT 重复符号。

### 4.5 merge 原理

`ld -r -T merge.ld --whole-archive` 用链接脚本将所有 per-function 段合并为单一 `.text`/`.rodata`/`.data` 段，DISCARD `.group`，消除 `.shstrtab`（8 MB → 0）。

### 4.6 架构支持

`lipo.py` 通过 `--arch=x86|aarch64` 区分 Target 库：

| 架构 | Target 库 |
|---|---|
| x86 | `X86CodeGen`, `X86Desc`, `X86Info` |
| aarch64 | `AArch64CodeGen`, `AArch64Desc`, `AArch64Info`, `AArch64Utils` |

---

## 5. 体积数据

### 5.1 X86_64

| 产物 | `-O3` | `-Os` |
|---|---|---|
| extract `.a` | 91 MB | 99 MB |
| gc-merge `.a` | 60 MB | 59 MB |
| **ejit.o** | **39 MB** | **37 MB** |
| 最终二进制 | 29 MB | **21 MB** |
| 最终 `.text` | 21.4 MB | **15.2 MB** |

### 5.2 ejit.o 内部构成（37 MB, -Os）

| 段 | 大小 | 说明 |
|---|---|---|
| `.text` | 22.7 MB | 合并后的代码 |
| `.strtab` | 5.9 MB | 符号名字符串 |
| `.rodata` | 5.9 MB | 只读数据 |
| `.symtab` | 1.9 MB | 符号表 |
| `.data` + `.bss` | 1.2 MB | 数据段 |
| `.rela.*` | ~0 MB | 合并后残留 |
| `.group` | 0 | 已 DISCARD |

### 5.3 最终二进制瘦身路径

```
ejit.o (37 MB)
  → 最终链接 --gc-sections: 裁剪未引用段
  → --strip-all: 去除符号表
  → 21 MB 二进制 / 15.2 MB .text
```

---

## 6. 裁剪效果对比

| 指标 | 原始 | 当前 | 缩减 |
|---|---|---|---|
| `.a`/`.o` 文件数 | 53 | **1** | -98% |
| 部署产物大小 | 117 MB (53 .a) | **37 MB** (1 .o) | -68% |
| 最终二进制 | ~30 MB | **21 MB** | -30% |
| 最终 `.text` | — | **15.2 MB** | — |
| EJIT 自身大小 | 100 KB | 100 KB | — |

---

## 7. 未完成的裁剪方向

### 7.1 源码级裁剪（需割裂 clang/JIT 共享的 `.a`）

| 可裁库 | 大小 | 阻断原因 |
|---|---|---|
| DebugInfoDWARF / CodeView | ~2.5 MB | clang 和 JIT 共用 AsmPrinter，排除会破坏 CFI 汇编 |
| GlobalISel | ~1.9 MB | X86CodeGen LINK_COMPONENTS 硬依赖 |
| Instrumentation | ~0.3 MB | X86CodeGen LINK_COMPONENTS 硬依赖 |
| RuntimeDyld | ~0.8 MB | OrcJIT Layer.cpp 核心引用 |
| OrcTargetProcess | ~0.01 MB | LLJIT SelfExecutorProcessControl 依赖 |

### 7.2 编译选项

| 选项 | 预计收益 | 状态 |
|---|---|---|
| `-Os` | -28% 二进制 | ✅ 已落地 |
| ThinLTO | -10~30% | 未测试 |

### 7.3 裸核专用 ExecutorProcessControl

自定义 EPC + JITLinkMemoryManager 可解锁 OrcTargetProcess/RuntimeDyld 裁剪，是裸核部署的必经之路。

---

## 8. 构建命令参考

```bash
# 配置（-Os 体积优化）
cmake -S llvm -B build_release_x86 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE='-Os -DNDEBUG -ffunction-sections -fdata-sections' \
  -DCMAKE_C_FLAGS_RELEASE='-Os -DNDEBUG -ffunction-sections -fdata-sections' \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DBUILD_SHARED_LIBS=OFF -DLLVM_OPTIMIZED_TABLEGEN=ON \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_ENABLE_ZLIB=OFF -DLLVM_ENABLE_ZSTD=OFF \
  -DEJIT_BARE_METAL=ON

# 构建
ninja -C build_release_x86 clang LLVMEJIT lld

# 生成 ejit.o
python3 ejit_test/lipo/lipo.py extract  --arch=x86 --build-dir=build_release_x86
python3 ejit_test/lipo/lipo.py gc-merge --input=ejit_test/lipo/libejit_lipo_x86.a --build-dir=build_release_x86
python3 ejit_test/lipo/lipo.py merge    --input=ejit_test/lipo/libejit_lipo_x86_gc.a --build-dir=build_release_x86

# 测试
./ejit_test/build.sh --run --lipo
```

---

*文档版本: 3.0*
*创建日期: 2026-05-24*
*更新日期: 2026-05-26*
