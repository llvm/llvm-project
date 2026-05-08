/**
 * EJIT 外部输入 cellIdx 端到端测试
 *
 * 场景：cellIdx 来自命令行参数（外部输入），AOT 编译期不可知。
 * 验证：JIT 运行时能够正确替代参数 → StructField 常量替换 → 分支折叠
 * 最终产生与手工计算一致的正确结果。
 *
 * 编译:
 *   ROOT=/home/ubuntu/github/llvm-project
 *   ${ROOT}/build/bin/clang -O2 \
 *     -I${ROOT}/llvm/include -I${ROOT}/build_x86/include -I${ROOT}/build/include \
 *     -c ejit_external_idx_test.c -o /tmp/ext_idx.o
 *   LIBS=$(ls ${ROOT}/build_x86/lib/*.a)
 *   clang++ -Os -Wl,--gc-sections -Wl,--strip-all /tmp/ext_idx.o \
 *     -Wl,--whole-archive $LIBS -Wl,--no-whole-archive \
 *     -lz -lpthread -ldl -o /tmp/ejit_external_idx
 *
 * 运行:
 *   /tmp/ejit_external_idx 0   # cellIdx=0
 *   /tmp/ejit_external_idx 3   # cellIdx=3
 *   /tmp/ejit_external_idx 7   # cellIdx=7
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

//===-- 场景 A: 单维 cellIdx 外部输入，分支折叠 -----------------------------===//

struct CellCfg {
  __attribute__((ejit_may_const)) uint32_t cellType;
  __attribute__((ejit_may_const)) uint32_t cellId;
  uint32_t trafficLoad;
};

__attribute__((ejit_period(("static")))) uint32_t g_sysVer;

#define N_CELL 8
__attribute__((ejit_period_arr("cell"))) struct CellCfg g_cellCfg[N_CELL];

//===-- 场景 A: 分支折叠验证 ------------------------------------------------===//
// 调用: process_cell_a(cellIdx)
// 验证: g_cellCfg[cellIdx].cellType 被常量替换后分支正确折叠
//       trafficLoad 的变化 = (cellType==特定值 ? 100 : 200)

__attribute__((ejit_entry))
void process_cell_a(
    __attribute__((ejit_period_arr_ind("cell"))) uint8_t cellIdx)
{
  if (g_cellCfg[cellIdx].cellType == 0xFD) {         // 253
    g_cellCfg[cellIdx].trafficLoad += 100;
  } else {
    g_cellCfg[cellIdx].trafficLoad += 200;
  }
}

//===-- 场景 B: 多维 cellIdx + trpIdx 外部输入 -----------------------------===//

struct TrpCfg {
  __attribute__((ejit_may_const)) uint32_t trpType;
  uint32_t activeBeams;
};

#define M_TRP 8
__attribute__((ejit_period_arr("trp"))) struct TrpCfg g_trpCfg[M_TRP];

// 调用: process_cell_trp(cellIdx, trpIdx)
// 验证: 两个 may_const 字段都被替换，复合条件被折叠

__attribute__((ejit_entry))
uint32_t process_cell_trp(
    __attribute__((ejit_period_arr_ind("cell"))) uint8_t cellIdx,
    __attribute__((ejit_period_arr_ind("trp")))  uint8_t trpIdx)
{
  uint32_t ct = g_cellCfg[cellIdx].cellType;
  uint32_t tt = g_trpCfg[trpIdx].trpType;

  // 复合条件: 两个 may_const 字段同时决定分支
  if (ct == 0xFD && tt == 1) {
    return 1000;
  } else if (ct == 0xEC && tt == 2) {
    return 2000;
  } else if (ct == 0xFD && tt == 2) {
    return 3000;
  }
  return 0;
}

//===-- 场景 C: 嵌套结构体 may_const 字段 -----------------------------------===//

struct InnerCfg {
  __attribute__((ejit_may_const)) uint32_t mode;
  uint32_t reserved;
};

struct OuterCfg {
  __attribute__((ejit_may_const)) uint32_t type;
  struct InnerCfg inner;
  uint32_t extra;
};

#define N_OUTER 4
__attribute__((ejit_period_arr("outer"))) struct OuterCfg g_outerCfg[N_OUTER];

// 调用: process_outer(cellIdx)
// 验证: 外层和内层 may_const 字段都能被正确替换

__attribute__((ejit_entry))
uint32_t process_outer(
    __attribute__((ejit_period_arr_ind("outer"))) uint8_t idx)
{
  uint32_t t = g_outerCfg[idx].type;
  uint32_t m = g_outerCfg[idx].inner.mode;

  if (t == 0xAA && m == 0x55) {
    return 777;
  }
  return 0;
}

//===-- 运行时 API ---------------------------------------------------------===//

extern int ejit_init(const void *cfg);
extern int ejit_activate(const char *name, unsigned char idx);
extern int ejit_deactivate(const char *name, unsigned char idx);
extern int ejit_is_active(const char *name, unsigned char idx);
extern void ejit_shutdown(void);

//===-- 测试辅助 -----------------------------------------------------------===//

static int g_failures = 0;

#define CHECK_EQ(actual, expected, name) do {                           \
  if ((actual) != (expected)) {                                         \
    printf("  FAIL %s: got %u, expected %u\n", name,                   \
           (unsigned)(actual), (unsigned)(expected));                   \
    g_failures++;                                                       \
  } else {                                                              \
    printf("  OK   %s = %u\n", name, (unsigned)(actual));              \
  }                                                                     \
} while(0)

// 初始化测试数据
static void init_test_data(uint8_t ci, uint8_t ti) {
  // 所有 cell 初始化为 0xEC
  for (int i = 0; i < N_CELL; i++) {
    g_cellCfg[i].cellType = 0xEC;
    g_cellCfg[i].cellId   = i * 10;
    g_cellCfg[i].trafficLoad = 0;
  }
  // 指定 cellIdx 设为 0xFD
  g_cellCfg[ci].cellType = 0xFD;
  g_cellCfg[ci].cellId   = ci * 10 + 42;

  // trp 初始化
  for (int i = 0; i < M_TRP; i++) {
    g_trpCfg[i].trpType = 0;
    g_trpCfg[i].activeBeams = 0;
  }
  g_trpCfg[ti].trpType = 1;

  // outer 初始化
  for (int i = 0; i < N_OUTER; i++) {
    g_outerCfg[i].type = 0xBB;
    g_outerCfg[i].inner.mode = 0;
    g_outerCfg[i].extra = 0;
  }
  // 指定 idx 设为匹配值
  g_outerCfg[ci % N_OUTER].type = 0xAA;
  g_outerCfg[ci % N_OUTER].inner.mode = 0x55;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Usage: %s <cellIdx> [trpIdx]\n", argv[0]);
    return 1;
  }

  uint8_t ci = (uint8_t)atoi(argv[1]);
  uint8_t ti = (argc >= 3) ? (uint8_t)atoi(argv[2]) : (uint8_t)0;

  printf("=== EJIT External cellIdx Test ===\n");
  printf("cellIdx = %u, trpIdx = %u\n\n", ci, ti);

  // 初始化运行时
  int rc = ejit_init(0);
  if (rc != 0) { printf("ejit_init FAIL: %d\n", rc); return 1; }

  // 初始化测试数据
  init_test_data(ci, ti);

  // 验证初始数据已设置
  printf("[Before JIT]\n");
  printf("  g_cellCfg[%u].cellType = 0x%x\n", ci, g_cellCfg[ci].cellType);
  printf("  g_cellCfg[%u].cellId   = %u\n", ci, g_cellCfg[ci].cellId);
  printf("  g_cellCfg[%u].trafficLoad = %u\n", ci, g_cellCfg[ci].trafficLoad);

  // 激活时间窗
  ejit_activate("cell", ci);
  ejit_activate("trp", ti);
  ejit_activate("outer", ci % N_OUTER);

  printf("\n=== 场景 A: process_cell_a(ci=%u) ===\n", ci);
  // 已知: g_cellCfg[ci].cellType = 0xFD
  // 所以 should take the "then" branch: trafficLoad += 100
  uint32_t before = g_cellCfg[ci].trafficLoad;
  process_cell_a(ci);
  uint32_t after = g_cellCfg[ci].trafficLoad;
  CHECK_EQ(after - before, 100u, "A: trafficLoad delta (then branch)");

  // 修改 cellType 为非 0xFD，测试 else 分支
  // 注：修改 may_const 字段需要通过 ejit_deactivate + 重新激活
  ejit_deactivate("cell", ci);
  g_cellCfg[ci].cellType = 0xA5;  // 不是 0xFD
  ejit_activate("cell", ci);

  printf("\n=== 场景 A2: process_cell_a(ci=%u) after type change ===\n", ci);
  before = g_cellCfg[ci].trafficLoad;
  process_cell_a(ci);
  after = g_cellCfg[ci].trafficLoad;
  CHECK_EQ(after - before, 200u, "A2: trafficLoad delta (else branch)");

  printf("\n=== 场景 B: process_cell_trp(ci=%u, ti=%u) ===\n", ci, ti);
  // g_cellCfg[ci].cellType = 0xA5 (from above), g_trpCfg[ti].trpType = 1
  // Not (0xFD && 1) and not (0xEC && 2) → should return 0
  uint32_t result_b = process_cell_trp(ci, ti);
  CHECK_EQ(result_b, 0u, "B: trp result (no match)");

  // Set cell back to 0xFD, trp to 1
  ejit_deactivate("cell", ci);
  ejit_deactivate("trp", ti);
  g_cellCfg[ci].cellType = 0xFD;
  g_trpCfg[ti].trpType = 1;
  ejit_activate("cell", ci);
  ejit_activate("trp", ti);

  printf("\n=== 场景 B2: process_cell_trp(ci=%u, ti=%u) FD+1 ===\n", ci, ti);
  result_b = process_cell_trp(ci, ti);
  CHECK_EQ(result_b, 1000u, "B2: trp result (FD && 1 → 1000)");

  // Set cell to 0xEC, trp to 2
  ejit_deactivate("cell", ci);
  ejit_deactivate("trp", ti);
  g_cellCfg[ci].cellType = 0xEC;
  g_trpCfg[ti].trpType = 2;
  ejit_activate("cell", ci);
  ejit_activate("trp", ti);

  printf("\n=== 场景 B3: process_cell_trp(ci=%u, ti=%u) EC+2 ===\n", ci, ti);
  result_b = process_cell_trp(ci, ti);
  CHECK_EQ(result_b, 2000u, "B3: trp result (EC && 2 → 2000)");

  // Set cell to 0xFD, trp to 2
  ejit_deactivate("cell", ci);
  ejit_deactivate("trp", ti);
  g_cellCfg[ci].cellType = 0xFD;
  g_trpCfg[ti].trpType = 2;
  ejit_activate("cell", ci);
  ejit_activate("trp", ti);

  printf("\n=== 场景 B4: process_cell_trp(ci=%u, ti=%u) FD+2 ===\n", ci, ti);
  result_b = process_cell_trp(ci, ti);
  CHECK_EQ(result_b, 3000u, "B4: trp result (FD && 2 → 3000)");

  printf("\n=== 场景 C: process_outer(idx=%u) ===\n", (unsigned)(ci % N_OUTER));
  uint32_t result_c = process_outer(ci % N_OUTER);
  CHECK_EQ(result_c, 777u, "C: outer/inner may_const both match");

  // 关闭
  ejit_shutdown();

  printf("\n=== Result: %d failures ===\n", g_failures);
  return g_failures > 0 ? 1 : 0;
}
