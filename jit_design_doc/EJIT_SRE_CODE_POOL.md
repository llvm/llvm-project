# EmbeddedJIT SRE 机器码内存池设计（EJIT_SRE_CODE_POOL）

**状态**: 已实现（方案 B，完整接管 JITLink 代码内存分配；默认 4K 粒度页封固，可回退整 2MiB 池封固）
**开关**:
- `-DEJIT_SRE_CODE_POOL=ON`（默认 OFF，上游默认行为不变）
- `-DEJIT_CODE_POOL_4K_SEAL=ON`（默认 ON，仅在 `EJIT_SRE_CODE_POOL=ON` 时生效；OFF 回退到旧的整 2MiB 池封固）

**关联代码**:
- `llvm/include/llvm/ExecutionEngine/EJIT/EJitCodePool.h` / `llvm/lib/ExecutionEngine/EJIT/EJitCodePool.cpp`
- `llvm/include/llvm/ExecutionEngine/EJIT/EJitCodePoolMemoryManager.h` / `.cpp`
- `llvm/include/llvm/ExecutionEngine/EJIT/EJitSrePlatform.h` / `.cpp`
- `llvm/lib/ExecutionEngine/EJIT/EJitOrcEngine.cpp`（接入点）
- 测试: `llvm/unittests/ExecutionEngine/EJIT/EJitCodePoolTest.cpp`、`EJitCodePoolMemoryManagerTest.cpp`

> **重要更新（4K 粒度执行权限封固）**：目标平台的**真实大页仍然是 2MiB**，但平台新增
> 了 `split_2m_to_4k(base, size)`，可把一段 2MiB 对齐的 VA 区间**拆成 4K 映射**；其后
> `enable_ex(1, pageVA)` 即可**按 4K 页**设置执行权限。因此 code pool 仍按 **2MiB 对齐**
> 申请大池（在池创建时对整池调用一次 `split_2m_to_4k`），但**封固改为 4K 粒度**：只对某
> 个函数实际写入机器码覆盖到的 4K 页逐页 `enable_ex`，而不是把整个 2MiB 池一次性封固。
> 详见 [§13](#13-4k-粒度执行权限封固ejit_code_pool_4k_seal)。下面的 §1–§8 描述通用的池
> 化/接管机制（两种封固模式共用）；§5 的"整池封固"是 `EJIT_CODE_POOL_4K_SEAL=OFF` 的旧
> 行为。

---

## 1. 背景与目标

目标平台（SRE 类）的执行权限原语有如下约束：

- `enable_ex(1, va)` 为 `va` **所在的 4K 页**设置执行权限（**新接口**；旧接口按 2MiB
  粒度）。
- 真实大页仍是 **2MiB**：必须先用 `split_2m_to_4k(base, size)` 把 2MiB 对齐区间拆成 4K
  映射，之后才能对其中的 4K 页 `enable_ex`。`base` 必须 2MiB 对齐，`size` 为 2MiB 的整数
  倍，且 `split_2m_to_4k` 必须在 `enable_ex` 之前调用。
- 一旦某 4K 页被置为 **RX（可执行）**，该页**不能再写**。

EmbeddedJIT 需要在运行期把 JIT 生成的机器码变为可执行。直接的做法（wrap 全局
`mprotect`）会破坏 ORC/JITLink 的后续写入，因此本方案让 EmbeddedJIT **自己拥有
JIT 代码内存**：以 2MiB 对齐大池为单位分配、写入阶段保持 RW、在把函数指针交还调用者
之前对实际写入的 4K 页调用 `enable_ex` 封固为 RX，封固后的页永不再写。


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
  `BasicLayout` 计算各 segment 大小，但 slab **来自 `EJitCodePoolManager` 的 2MiB 对齐
  池**（而非 mmap）；**allocate 阶段绝不 `enable_ex`**。
- `finalize()` **刻意不做任何 mprotect**；也不调用 `InvalidateInstructionCache`（由
  `enable_ex` 负责）。
- 真正的 RW→RX 封固时机按模式不同：
  - **4K 模式（默认）**：在 `finalize()` 中（所有写入/relocation/fixup 完成后）对该
    allocation 覆盖到的 4K 页逐页 `enable_ex`；任一页失败则 `finalize` 返回 Error，
    `lookup` 不会拿到可调用指针。见 [§13](#13-4k-粒度执行权限封固ejit_code_pool_4k_seal)。
  - **整池模式（`EJIT_CODE_POOL_4K_SEAL=OFF`）**：`finalize` 仍保持 RW，封固由
    `EJitOrcEngine::lookup` 在返回函数指针前对包含该地址的整 2MiB 池调用 `enable_ex`
    完成（幂等：已封固的池不重复调用）。

换言之：**代码内存来自 EJITCodePoolManager，而非普通 mmap/mprotect**；封固在 finalize
（4K 模式）或 lookup（整池模式）处发生。两部分共同构成完整方案，没有"伪装成内存池"
的浅层实现。

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
| `EJIT_SRE_CODE_POOL` | OFF | 启用 2MiB 对齐代码池 + enable_ex 封固；同时隐含启用 `EJIT_SRE_ENABLE_EX` |
| `EJIT_CODE_POOL_4K_SEAL` | ON | 仅在 `EJIT_SRE_CODE_POOL=ON` 时生效。开启：池创建时 `split_2m_to_4k`，封固按 **4K 页** `enable_ex`。关闭：回退到整 2MiB 池封固。**唯一新增布尔开关，无新增数值配置**（2MiB/4KiB 为实现内平台常量） |
| `EJIT_SRE_CODE_POOL_SIZE` | 2097152 (2MiB) | 每个池的可用字节数；4K 模式下向上取整为 2MiB 整数倍 |
| `EJIT_SRE_CODE_POOL_PTNO` | 8 | 传给 `SRE_MemDbgAlloc` 的分区号 ptNo |
| `EJIT_SRE_ENABLE_EX` | 随 `EJIT_SRE_CODE_POOL` | 关闭后代码仍走池，但不做权限翻转（bring-up/测量用） |

平台接口（在 `EJitSrePlatform.cpp` 内**仅声明、不定义**，避免污染通用命名空间）：

```cpp
extern "C" unsigned ejit_sre_enable_ex(unsigned startLevel,
                                       unsigned long long va) __asm__("enable_ex");
extern "C" unsigned ejit_sre_split_2m_to_4k(unsigned long long va,
                                            unsigned long long size) __asm__("split_2m_to_4k");
extern "C" void *SRE_MemDbgAlloc(unsigned int mid, unsigned char ptNo,
                                 unsigned long size, const char *func,
                                 unsigned int line);
```

**EmbeddedJIT runtime 只声明平台符号、不定义平台符号，也不提供 weak 兜底。** 真实平台 /
业务链接环境**必须提供**这些符号的强定义：

- `enable_ex`
- `split_2m_to_4k`
- `SRE_MemDbgAlloc`
- 若启用平台 taskpool 队列（`EJIT_SRE_TASKPOOL_PLATFORM_QUEUE`），还必须提供
  `QueueCreate` / `QueueWrite` / `QueueRead`（见 `EJitSreQueue.cpp`）。

**为什么不做 weak 兜底**：

- 在静态包 / 半链接 / 平台 SDK 场景下，weak 本地定义可能**覆盖或与真实平台符号冲突、
  错误绑定**，造成重复定义或符号归属不清。
- 平台缺符号时应当**在链接期暴露问题**，而不是被一个静默 no-op 兜底掩盖（no-op 的
  `enable_ex` 会让"未真正赋予执行权限"的代码看起来成功）。

**host 单测不依赖任何真实平台符号**：`EJitCodePoolManager` 的 raw 分配器、seal、split
都是**注入的 callback**，单测全部传入 mock；测试也不引用 `makeSreCodePoolManager`，因此
`EJitSrePlatform.cpp` 的外部引用根本不会被拉进宿主测试链接。默认 host taskpool 测试不
定义 `EJIT_SRE_TASKPOOL_PLATFORM_QUEUE`，走 mock ring queue。

> **封固 API 边界**：4K 模式只应通过 `sealCodeRange(start, size)` 封固**实际写入机器码的
> 范围**（由 memory manager 的 finalize 驱动）。`sealPoolContaining` / `sealAllWritablePools`
> 是**整池封固**语义，仅用于旧整池模式；在 4K 模式下它们会**返回 Error**（裸指针无 size，
> 无法判断该封几页；整池"全部封固"在 4K 模式也无正确含义），避免误用。

---

## 10. 如何验证

**默认构建（开关 OFF，行为不变）**：

```bash
cmake --build build-ejit --target LLVMEJIT -j8
```

**代码池单元测试（宿主可跑，使用 mock allocator + mock enable_ex/split_2m_to_4k，不需真实 SRE）**：

```bash
cmake --build build-ejit --target EJITCodePoolTests -j8
cmake --build build-ejit --target check-ejit-code-pool -j8
```

覆盖：
- `EJitCodePoolTest`（整池模式）：2MiB 对齐、RW 池内 bump、seal 标记 executable、
  sealed 不复用、重复 seal 不重复 enable_ex、enable_ex 失败返回 Error、超池大小
  clean reject、统计正确、rollover 自动封固、`sealAllWritablePools` 等。
- `EJitCodePoolTest`（4K 模式 `EJitCodePool4K.*`）：故意非 2MiB 对齐 rawBase → 对齐后
  2MiB 对齐且 usable 不越界、`split_2m_to_4k(alignedBase, usableSize)` 每池只调用一次、
  小代码只封覆盖到的 4K 页、跨多页循环 enable_ex、某页 enable_ex 失败返回 Error、
  后续 allocation 落到下一个 4K 页（不落已封页）、split 失败池创建失败、rollover 新池
  再次 split 且不整池封固、**4K 模式下 `sealPoolContaining` / `sealAllWritablePools`
  返回 Error 且不 enable_ex**。
- `EJitCodePoolMemoryManagerTest`（整池 + 4K 模式 `EJitCodePoolMemMgr4K.*`）：用合成
  JITLink `LinkGraph` 驱动内存管理器，验证 JIT 代码来自池、allocate 阶段不 enable_ex、
  finalize 只封覆盖到的 4K 页、跨多页 finalize 逐页封固、某页 enable_ex 失败时 finalize
  返回 Error（不返回函数指针）、第二个函数复用同一 2MiB 池但落在不同 4K 页。

**代码池开关构建（开关 ON，默认 4K 封固）**：

```bash
cmake -S llvm -B build-ejit-sre-pool -DEJIT_SRE_CODE_POOL=ON \
  -DEJIT_SRE_CODE_POOL_PTNO=8 <其余参数>      # EJIT_CODE_POOL_4K_SEAL 默认 ON
cmake --build build-ejit-sre-pool --target LLVMEJIT -j8
cmake --build build-ejit-sre-pool --target check-ejit-code-pool -j8
```

---

## 11. 已知限制

1. **不释放 code pool（FreeCode 不物理释放）**：池生命周期等于 EJIT runtime/engine
   生命周期；已封固的页/池不回收。`FreeCode` / `deallocate` 只运行 dealloc actions，
   **不归还 code pool 物理内存**。这是为了可靠性、避免 W/RX 权限冲突（目标平台
   free/realloc 也可能不可靠），4K 模式同样如此。
2. **单函数 ≤ 池大小**：超过 `EJIT_SRE_CODE_POOL_SIZE` 的单次请求返回 Error。
3. **4K 模式每个 allocation 至少占 1 个 4K 页**：v1 为可靠起见，每个 JIT allocation
   起点 4K 对齐、大小向上取整到 4K，故最小占用 4K（旧的整 2MiB 粒度则最小占用
   2MiB）。不把多个独立 finalize 的函数塞进同一个 4K 页，避免 RX 后再写。
4. **端到端原生执行未在本机验证**：本机为 aarch64 宿主但只构建了 X86 后端，无法在
   宿主上真正执行 JIT 机器码。因此所有测试都是**宿主可跑的逻辑/集成测试**（mock
   分配器 + mock enable_ex / split_2m_to_4k + 合成 LinkGraph）；真正"编译→执行特化
   函数"需在与所构建后端匹配的目标/宿主上验证。
5. **同步编译假设**：旧整池模式策略 3 的"满则提前封固"、以及 4K 模式"finalize 即封
   固对应页"都依赖 EmbeddedJIT 的同步编译模型（`setNumCompileThreads(0)`）：一个
   allocation 一旦 finalize 完成就不再被写。
6. **EJITTests 现状**：上游基线分支上的 `EJITTests` 因无关的历史 API 漂移当前无法
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

---

## 13. 4K 粒度执行权限封固（EJIT_CODE_POOL_4K_SEAL）

这是当前**默认**的封固模式（`EJIT_CODE_POOL_4K_SEAL=ON`，仅在 `EJIT_SRE_CODE_POOL=ON`
时生效）。它适配目标平台**新的 4K 粒度执行权限接口**，在保留大块 code pool 设计的同时
把封固粒度从整 2MiB 降到 4K。

### 13.1 平台语义

```cpp
extern "C" unsigned int split_2m_to_4k(unsigned long long va, unsigned long long size);
extern "C" unsigned int enable_ex(unsigned int startLevel, unsigned long long va);
```

- **真实大页仍是 2MiB**。`split_2m_to_4k(base, size)` 把一段 **2MiB 对齐**的 VA 区间
  拆成 **4K 映射**：`base` 必须 2MiB 对齐，`size` 必须是 2MiB 的整数倍。
- `split_2m_to_4k` **必须在** `enable_ex` **之前**调用。
- 拆分之后，`enable_ex(1, pageVA)` 对 `pageVA` **所在的 4K 页**设置执行权限。
- 返回 0 表示成功，非 0 表示失败。

### 13.2 2MiB 对齐大池 + alignment slack

池仍按 2MiB 对齐申请，做法（`EJitCodePoolManager::newActivePoolLocked`，4K 模式）：

```
requestedSize = poolSize + 2MiB           // 多申请一个大页做对齐 slack
rawBase       = platform alloc(requestedSize)   // 目标平台 SRE_MemDbgAlloc
alignedBase   = align_up(rawBase, 2MiB)
usableSize    = poolSize                   // poolSize 已向上取整到 2MiB 整数倍
assert(alignedBase + usableSize <= rawBase + requestedSize)   // 不越界检查
split_2m_to_4k(alignedBase, usableSize)    // 每个池创建时只调用一次
```

- `poolSize` 若不是 2MiB 整数倍，构造时**向上取整**到 2MiB 整数倍。
- 2MiB / 4KiB 是**实现内的平台常量**（`EJitSrePlatform.cpp` 的适配器里写死），不做成
  一堆 CMake 数值配置；`EJitCodePoolManager` 本身把它们作为 `Options` 字段，便于宿主用
  小数值做单测（与既有 256B 池测试同一思路）。
- `split_2m_to_4k` 失败 → **池创建失败，返回明确 Error**，不会注册半成品池。每个池
  **只 split 一次**，绝不在每次函数编译时重复 split。

### 13.3 4K 封固粒度（只封写到的页）

`finalize` 完成（所有写入、relocation、link graph fixup 都已结束）后，对该 allocation
实际写入机器码覆盖到的 4K 页**逐页**封固（`EJitCodePoolManager::sealCodeRange`）：

```
pageStart = align_down(codeStart, 4KiB)
pageEnd   = align_up(codeStart + size, 4KiB)
for (page = pageStart; page < pageEnd; page += 4KiB)
    enable_ex(1, page)
```

- 只封固覆盖到的页：一个 ~100 字节的小函数只 `enable_ex` **1 个 4K 页**，而不是整个
  2MiB（旧模式 512 个页）。跨多页的代码会循环逐页 `enable_ex`。
- **任何一页 `enable_ex` 失败 → 返回 Error，绝不返回可调用函数指针。** 封固发生在
  JITLink memory manager 的 `finalize` 中，失败会让 `finalize`（进而 `lookup`）返回
  Error。
- 封固后**不**调用 `__builtin___clear_cache`：`enable_ex` 自身完成权限/cache 同步。

### 13.4 避免写入已 RX 页（分配策略）

采用简单可靠策略（v1）：

- **每个 allocation 起点按 4K 对齐**（`EffAlign = max(reqAlign, 4KiB)`）。
- **allocation size 向上取整到 4K**：bump 游标推进到 `align_up(off + size, 4KiB)`。
- finalize 后封固该 allocation 覆盖的页；**后续 allocation 从下一个 4K 页开始**，因此
  绝不会写入已经 RX 的页。
- **不在 rollover 时整池封固**：4K 模式下池不再"满则整池 seal"。当前池放不下时直接换
  新池（旧池已封固的页保持 RX，未用尾部就是闲置 RW，无害）。同一个池可以服务很多
  4K 函数，**池保持部分可写**。

代价：每个独立 finalize 的函数/section 最少占 1 个 4K 页；不会为了极限省内存把多个独立
finalize 的函数塞进同一个 4K 页（那样会在某页 RX 后还要写它，违反平台约束）。

### 13.5 内存节省效果

- 旧模式：每个"封固 epoch"烧掉一整个 **2MiB** 池（一个池一旦因 lookup 被整池封固便不再
  复用，尾部全部浪费）。N 个小函数最坏占 N×2MiB。
- 4K 模式：每个小函数只占 **4K**（向上取整），且**同一个 2MiB 池可装下最多 512 个**
  这样的函数后才换池。相对旧模式，封固/占用粒度从 2MiB 降到 4K，**约 512×** 的粒度收
  益，小函数密集场景内存占用大幅下降。

### 13.6 接入点小结

| 步骤 | 位置 | 行为 |
|------|------|------|
| 池创建 split | `EJitCodePool.cpp::newActivePoolLocked`（4K 分支） | 对齐 2MiB 后 `split_2m_to_4k(base, poolSize)` 一次 |
| allocate | `EJitCodePoolMemoryManager::allocate` | 从池 bump 取 RW slab，**不** `enable_ex` |
| 4K 封固 | `EJitCodePoolMemoryManager::InFlightAllocImpl::finalize` → `EJitCodePool.cpp::sealCodeRange` | finalize 后对 slab 覆盖的每个 4K 页 `enable_ex(1, pageVA)`；失败返回 Error |
| lookup | `EJitOrcEngine::lookup` | 4K 模式不再整池封固（已在 finalize 完成）；旧整池模式才在此 `sealPoolContaining` |

不 wrap 全局 `mprotect`、不走 mmap/mprotect 老路径；代码内存全部来自注入的 raw 分配器。

### 13.7 封固 API 在两种模式下的语义

| API | 旧整池模式 | 4K 页模式 |
|------|-----------|-----------|
| `sealCodeRange(start, size)` | （可用）逐 4K 页封固覆盖范围 | **主路径**：finalize 处封固写入范围 |
| `sealPoolContaining(ptr)` | 整池封固包含 `ptr` 的池 | **返回 Error**（裸指针无 size，无法判断封几页；应改用 `sealCodeRange`） |
| `sealAllWritablePools()` | 封固所有仍可写的池 | **返回 Error**（4K 模式池有意保持部分可写，"全部封固"无正确含义） |

`EJitOrcEngine::lookup` 已对 `sealPoolContaining` 用 `!usesPageSeal()` 守卫，因此 4K 模式
下不会触发这两个 Error 路径；它们的 Error 仅为防止外部误用。

### 13.8 可定位性 trace（默认无输出）

code pool 与 taskpool 关键路径埋了**默认空展开**的 trace 宏，便于上板时改成 `SRE_printf`：

```cpp
#ifndef EJIT_CODE_POOL_TRACE
#define EJIT_CODE_POOL_TRACE(...) do {} while (0)
#endif
#ifndef EJIT_TASKPOOL_TRACE
#define EJIT_TASKPOOL_TRACE(...) do {} while (0)
#endif
```

- 默认展开为空，**不产生任何输出、不改变任何行为**。
- 上板时可用 `-D'EJIT_CODE_POOL_TRACE(...)=SRE_printf(__VA_ARGS__)'` 之类重定义。
- 参数只用**整数 / 指针 / C 字符串**，不构造 `std::string`、不使用 `raw_ostream`。
- 埋点：code pool 的 `newActivePoolLocked`（enter / rawAlloc / alignedBase / splitRc）、
  `allocateCode`（req / res）、`sealCodeRange`（范围 / 每页 rc）、`sealPoolContaining` 的
  4K 误用；taskpool 的 `compileOrGet`（enter / cacheHit / dedup / queuePush）、`runCompile`
  （begin / compiled / published）、`pollOne`（empty / dequeued）、`freeCode`（cancel/free）。
