# EmbeddedJIT SRE 机器码内存池设计（EJIT_SRE_CODE_POOL）

**状态**: 已实现（方案 B，完整接管 JITLink 代码内存分配 + lookup 处 enable_ex 封固）
**开关**: `-DEJIT_SRE_CODE_POOL=ON`（默认 OFF，上游默认行为不变）
**关联代码**:
- `llvm/include/llvm/ExecutionEngine/EJIT/EJitCodePool.h` / `llvm/lib/ExecutionEngine/EJIT/EJitCodePool.cpp`
- `llvm/include/llvm/ExecutionEngine/EJIT/EJitCodePoolMemoryManager.h` / `.cpp`
- `llvm/include/llvm/ExecutionEngine/EJIT/EJitSrePlatform.h` / `.cpp`
- `llvm/lib/ExecutionEngine/EJIT/EJitOrcEngine.cpp`（接入点）
- 测试: `llvm/unittests/ExecutionEngine/EJIT/EJitCodePoolTest.cpp`、`EJitCodePoolMemoryManagerTest.cpp`

---

## 1. 背景与目标

目标平台（SRE 类）的执行权限原语 `enable_ex` 有两条硬约束：

1. **只能按 2MiB 粒度**设置执行权限：`enable_ex(1, va)` 中 `va` 必须 2MiB 对齐。
2. 一旦某 2MiB 区域被置为 **RX（可执行）**，该区域**不能再写**。

EmbeddedJIT 需要在运行期把 JIT 生成的机器码变为可执行。直接的做法（wrap 全局
`mprotect`）会破坏 ORC/JITLink 的后续写入，因此本方案让 EmbeddedJIT **自己拥有
JIT 代码内存**：以 2MiB 池为单位分配、写入阶段保持 RW、在把函数指针交还调用者之前
对该池调用 `enable_ex` 封固为 RX，封固后的池永不再写。

---

## 2. 为什么不能 wrap 全局 mprotect

- ORC/JITLink 的 in-process 内存管理器在 finalize 阶段会对各 segment 调用
  `mprotect` 切换到 R-X。如果我们 wrap 全局 `mprotect` 把它转成 `enable_ex`，那么：
  - `enable_ex` 是 **2MiB 粒度**的，而 JITLink 的 segment 通常是页粒度、且多个分配
    可能落在同一个 2MiB 区域里。把整块 2MiB 置为 RX 后，**同一 2MiB 内后续分配的
    写入会失败**（W^X 冲突），导致下一个模块链接/重定位崩溃。
  - JITLink 在 finalize 之后、甚至在 allocation actions 中仍可能写入工作内存；全局
    wrap 无法区分"这次 mprotect 该转 enable_ex"还是"这页之后还要写"。
- 全局 wrap 还会影响**业务侧**所有 `mprotect` 调用，污染范围过大、不可控。

因此我们**不 wrap 全局 mprotect**，而是在 EmbeddedJIT 自己的 JIT 代码分配/封固链
条里精确处理。

---

## 3. 为什么 enable_ex 必须在"JIT 写完之后、返回函数指针之前"

- **之前**（写入期间）：JITLink 需要把机器码拷进内存并应用重定位，这要求该内存
  **可写（RW）**。如果过早 `enable_ex` 置 RX，写入/重定位会失败。
- **之后**（调用之前）：调用者拿到函数指针就会执行它，这要求该内存
  **可执行（RX）**。
- 因此唯一安全的封固点是：**JITLink finalize 完成、即将把函数指针交还调用者的那一刻**。
  在本实现里就是 `EJitOrcEngine::lookup()` 取得解析地址之后、`return` 之前。

时序：

```
allocate(RW, 来自 2MiB 池)
   -> JITLink 写机器码 + 重定位（池保持 RW）
   -> finalize（本实现不做 mprotect，池仍 RW）
   -> lookup 得到函数地址
   -> enable_ex(1, pool->base) 封固该 2MiB 池为 RX，标记 sealed
   -> 返回函数指针（此时指向 RX 内存，可安全执行）
```

> 注意：`enable_ex` 内部已完成权限/cache 同步，所以本实现在 SRE 模式下
> **不调用 `__builtin___clear_cache`**。

---

## 4. 为什么采用 2MiB 池

- `enable_ex` 的最小粒度就是 2MiB，因此**池的对齐与封固单位必须是 2MiB**，否则一次
  封固会波及相邻无关内存。
- 以 2MiB 为单位 bump 分配，使"封固一个池"与"该池内所有函数都已写完"一一对应，
  从而保证封固后绝不再写。
- 2MiB 大页对 icache/iTLB 局部性也更友好。

每个池的原始申请大小为 `poolSize + 2MiB - 1`，再把基址向上对齐到 2MiB：
`base = alignUp(raw, 2MiB)`，池仅使用 `[base, base + poolSize)`，`raw` 保留在
`CodePool` 中用于调试/未来回收。

---

## 5. 为什么 sealed 池不再写

- 平台约束：2MiB 区域置 RX 后不可写。封固后若再往里 bump 新代码，写入必然失败。
- 因此分配策略规定：**active 池一旦 sealed（或空间不足被提前 sealed），新分配一律
  进入新的 2MiB 池**。sealed 池只读只执行，生命周期内不回收、不复用。

分配策略（`EJitCodePoolManager::allocateCode`）：

1. 无 active 池 → 申请新 2MiB 池。
2. active 池已 sealed → 申请新 2MiB 池。
3. active 池剩余空间不足 → **先 seal 当前池**，再申请新 2MiB 池。
4. 否则在 active 池内 bump 分配（对齐到 `max(请求对齐, 64)`）。
5. 单次请求 > 池大小 → **clean error**（不静默回退）。
6. 第一版不支持释放小块、不做 free-list、不复用 sealed 池（可靠性优先）。

> 策略 3 的"提前封固"在 EmbeddedJIT 的**同步编译**模型下是安全的：当新的分配到来
> 时，落在当前池里的上一个模块早已 finalize 完成，不会再写。

---

## 6. 方案 B 的优缺点

**优点**
- 可靠：写入期 RW、执行期 RX，封固点唯一且明确，不会出现 W^X 冲突。
- 与业务内存隔离：JIT 机器码只来自 SRE 分配器，不进业务 heap。
- 局部性更好：2MiB 大页利于 icache/iTLB。
- 与上游解耦：通过 JITLink `JITLinkMemoryManager` 接口替换，不改通用 mprotect/malloc。

**缺点**
- 内存按 2MiB 粒度增长；第一版**不释放** code pool（sealed 不回收）。
- 每个池尾部可能有浪费（`wastedBytes` 统计可见）。
- 单个特化函数不能超过一个池大小（默认 2MiB），否则 clean error。

---

## 7. 与普通业务内存的边界

- **机器码字节**：永远来自注入的 raw 分配器（目标平台 `SRE_MemDbgAlloc`，分区
  `EJIT_SRE_CODE_POOL_PTNO`，默认 8），**绝不**来自 `malloc/new`/业务 heap。
- **池描述符（`CodePool` 记账）**：是少量普通宿主内存（`std::vector` 记账），**不
  存放任何可执行代码**。这是记账与代码字节的明确分界。
- `EJitCodePoolManager` **不依赖任何 SRE 头文件**：raw 分配器与 `enable_ex` 封固
  都是注入的回调，因此该类可在宿主上用 mock 完整单测；真实平台适配集中在
  `EJitSrePlatform.cpp`，不把 SRE 头文件塞进通用 LLVM 文件。

---

## 8. 接入点：真正接管了 JITLink 代码内存分配吗？

**是。** 本实现不是"只在 lookup 前 seal 一下"的浅层 hook，而是用一个自定义的
`jitlink::JITLinkMemoryManager`（`EJitCodePoolMemoryManager`）**接管了 JIT segment
的内存分配**：

- `EJitOrcEngine::Create` 在 `EJIT_SRE_CODE_POOL` 下，通过
  `LLJITBuilder::setObjectLinkingLayerCreator` 安装一个使用
  `EJitCodePoolMemoryManager` 的 `orc::ObjectLinkingLayer`。
- `EJitCodePoolMemoryManager::allocate` 仿照 `InProcessMemoryManager`，用
  `BasicLayout` 计算各 segment 大小，但 slab **来自 `EJitCodePoolManager` 的 2MiB
  池**（而非 mmap）。
- `finalize()` **刻意不做任何 mprotect**，让池保持 RW；也不调用
  `InvalidateInstructionCache`（由 `enable_ex` 负责）。
- 真正的 RW→RX 封固由 `EJitOrcEngine::lookup` 在返回函数指针前对包含该地址的池调用
  `enable_ex` 完成（幂等：已封固的池不重复调用）。

换言之：**代码内存来自 EJITCodePoolManager，而非普通 mmap/mprotect**；封固在
lookup 处发生。两部分共同构成完整方案，没有"伪装成内存池"的浅层实现。

---

## 9. 如何开启与配置（CMake）

```bash
cmake -S llvm -B build-ejit-sre-pool \
  -DEJIT_SRE_CODE_POOL=ON \
  -DEJIT_SRE_CODE_POOL_PTNO=8 \
  -DEJIT_DEFAULT_TARGET_TRIPLE=aarch64_be-unknown-linux-gnu \
  <保留项目原本需要的 CMake 参数>
```

| 开关 | 默认 | 说明 |
|------|------|------|
| `EJIT_SRE_CODE_POOL` | OFF | 启用 2MiB 代码池 + enable_ex 封固；同时隐含启用 `EJIT_SRE_ENABLE_EX` |
| `EJIT_SRE_CODE_POOL_SIZE` | 2097152 (2MiB) | 每个池的可用字节数 |
| `EJIT_SRE_CODE_POOL_PTNO` | 8 | 传给 `SRE_MemDbgAlloc` 的分区号 ptNo |
| `EJIT_SRE_ENABLE_EX` | 随 `EJIT_SRE_CODE_POOL` | 关闭后代码仍走池，但不做权限翻转（bring-up/测量用） |

平台接口（在 `EJitSrePlatform.cpp` 内声明，避免污染通用命名空间）：

```cpp
extern "C" unsigned ejit_sre_enable_ex(unsigned startLevel,
                                       unsigned long long va) __asm__("enable_ex");
extern "C" void *SRE_MemDbgAlloc(unsigned int mid, unsigned char ptNo,
                                 unsigned long size, const char *func,
                                 unsigned int line);
```

为保证**宿主无真实 SRE 符号时也能链接/跑通**，`EJitSrePlatform.cpp` 为这两个符号
提供了 **weak 兜底定义**（`enable_ex` 兜底为 no-op 成功、`SRE_MemDbgAlloc` 兜底为
对齐的宿主分配）。真实平台提供的 **strong 符号会覆盖** weak 兜底。

---

## 10. 如何验证

**默认构建（开关 OFF，行为不变）**：

```bash
cmake --build build-ejit --target LLVMEJIT -j4
```

**代码池单元测试（宿主可跑，使用 mock allocator + mock enable_ex，不需真实 SRE）**：

```bash
cmake --build build-ejit --target EJITCodePoolTests -j4
cmake --build build-ejit --target check-ejit-code-pool -j4
```

覆盖：
- `EJitCodePoolTest`：2MiB 对齐、RW 池内 bump、seal 标记 executable、sealed 不复用、
  重复 seal 不重复 enable_ex、enable_ex 失败返回 Error、超池大小 clean reject、
  统计正确、rollover 自动封固、`sealAllWritablePools` 等。
- `EJitCodePoolMemoryManagerTest`：用合成 JITLink `LinkGraph` 驱动内存管理器，验证
  JIT 代码来自池、finalize 后仍 RW、lookup 式封固只翻一次、第二个函数在前一个池
  sealed 后进入新池。

**代码池开关构建（开关 ON）**：

```bash
cmake -S llvm -B build-ejit-sre-pool -DEJIT_SRE_CODE_POOL=ON \
  -DEJIT_SRE_CODE_POOL_PTNO=8 <其余参数>
cmake --build build-ejit-sre-pool --target LLVMEJIT -j4
cmake --build build-ejit-sre-pool --target check-ejit-code-pool -j4
```

---

## 11. 已知限制

1. **不释放 code pool**：池生命周期等于 EJIT runtime/engine 生命周期；sealed 池不回
   收。这是为了可靠性、避免 W/RX 权限冲突（目标平台 free/realloc 也可能不可靠）。
2. **单函数 ≤ 池大小**：超过 `EJIT_SRE_CODE_POOL_SIZE` 的单次请求返回 Error。
3. **端到端原生执行未在本机验证**：本机为 aarch64 宿主但只构建了 X86 后端，无法在
   宿主上真正执行 JIT 机器码。因此所有测试都是**宿主可跑的逻辑/集成测试**（mock
   分配器 + mock enable_ex + 合成 LinkGraph）；真正"编译→执行特化函数"需在与所构建
   后端匹配的目标/宿主上验证。
4. **同步编译假设**：策略 3 的"满则提前封固"依赖 EmbeddedJIT 的同步编译模型
   （`setNumCompileThreads(0)`）。
5. **EJITTests 现状**：上游基线分支上的 `EJITTests` 因无关的历史 API 漂移当前无法
   编译；因此本特性的测试放在独立的 `EJITCodePoolTests` 可执行文件 + `check-ejit-
   code-pool` 目标中，与之解耦。

---

## 12. 后续可扩展

- **free-list / pool reset / LRU**：在 quiesce 点整体 reset 或按 LRU 回收 sealed 池
  （需要平台支持把 RX 区域改回 RW 或释放）。
- **跨进程共享**：只共享 **bitcode / object**，**不直接共享含绝对地址的机器码**
  （机器码内嵌了进程相关的绝对地址，跨进程不安全）。
- **更细的 finalize 段处理**：当前把 standard/finalize 段一起放进池；后续可单独管理
  finalize 段以减少浪费。
