/**
 * EJIT 多维数组 may_const 运行时测试
 *
 * 场景: 2D 周期数组，cellIdx 从 argv 外部输入。
 *       g_cells[cellIdx][subIdx].type 等 may_const 字段在 JIT 时
 *       被替换为常量，分支被折叠。
 *
 * 运行:
 *   ./ejit_multidim 0   # cellIdx=0
 *   ./ejit_multidim 2   # cellIdx=2
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

//===-- 2D 周期数组 -----------------------------------------------------------===//

struct SlotCfg {
  __attribute__((ejit_may_const)) uint32_t slotType;
  __attribute__((ejit_may_const)) uint32_t priority;
  uint32_t usageCount;
};

#define N_ROWS  4
#define N_COLS  8
__attribute__((ejit_period_arr("cell"))) struct SlotCfg g_slots[N_ROWS][N_COLS];

//===-- 2D 嵌套结构体数组 -----------------------------------------------------===//

struct Inner2D {
  __attribute__((ejit_may_const)) uint32_t mode;
  uint32_t pad;
};

struct Outer2D {
  struct Inner2D inner;
  uint32_t flags;
};

__attribute__((ejit_period_arr("cell"))) struct Outer2D g_outer2d[N_ROWS][N_COLS];

//===-- JIT entry: 2D flat struct access ------------------------------------===//

__attribute__((ejit_entry))
uint32_t classify_slot(
    __attribute__((ejit_period_arr_ind("cell"))) uint8_t ci)
{
  uint32_t sum = 0;
  // 遍历二维数组的第二维 — JIT 应把 may_const 替换为常量
  for (int s = 0; s < N_COLS; s++) {
    struct SlotCfg *p = &g_slots[ci][s];  // ci 外部输入，s 循环变量
    if (p->slotType == 0xFD)
      sum += p->priority * 10;
    else if (p->slotType == 0xEC)
      sum += p->priority * 5;
    // else: 不加
  }
  return sum;
}

//===-- JIT entry: 2D nested struct access ----------------------------------===//

__attribute__((ejit_entry))
uint32_t check_outer2d(
    __attribute__((ejit_period_arr_ind("cell"))) uint8_t ci)
{
  // 访问 g_outer2d[ci][3].inner.mode
  uint32_t m = g_outer2d[ci][3].inner.mode;
  if (m == 0x55) return 777;
  if (m == 0xAA) return 888;
  return 0;
}

//===-- 运行时 API -----------------------------------------------------------===//

extern int ejit_init(const void *cfg);
extern int ejit_activate(const char *name, unsigned char idx);
extern int ejit_deactivate(const char *name, unsigned char idx);
extern void ejit_shutdown(void);

//===-- 断言 -----------------------------------------------------------------===//

static int g_fail = 0;
#define T(cond, fmt, ...) do {               \
  if (cond) printf("  OK   " fmt "\n", ##__VA_ARGS__); \
  else      printf("  FAIL " fmt "\n", ##__VA_ARGS__), g_fail++; \
} while(0)

int main(int argc, char **argv) {
  uint8_t ci = (argc >= 2) ? (uint8_t)atoi(argv[1]) : 0;

  printf("=== EJIT Multi-Dim Array Test ===\n");
  printf("cellIdx=%u\n\n", ci);

  ejit_init(0);

  //===-- 场景 A: 2D flat struct，遍历第二维 --------------------------------===//
  printf("--- A: 2D classify_slot(ci=%u) ---\n", ci);

  // 初始化 g_slots[ci][*]:
  //   [0]=FD prio=3, [1]=EC prio=7, [2]=FD prio=1, [3]=XX, [4]=EC prio=2, ...
  for (int s = 0; s < N_COLS; s++) {
    if (s % 2 == 0) {
      g_slots[ci][s].slotType = 0xFD;  // FD 分支
      g_slots[ci][s].priority  = (uint32_t)s + 1;
    } else {
      g_slots[ci][s].slotType = 0xEC;  // EC 分支
      g_slots[ci][s].priority  = (uint32_t)s * 2;
    }
  }

  ejit_activate("cell", ci);
  uint32_t r = classify_slot(ci);
  // 偶数 s: FD → priority*10: s=0→1*10=10, s=2→3*10=30, s=4→5*10=50, s=6→7*10=70 → sum=160
  // 奇数 s: EC → priority*5:  s=1→2*5=10, s=3→6*5=30, s=5→10*5=50, s=7→14*5=70 → sum=160
  // total = 160 + 160 = 320
  uint32_t exp_a = 320;
  T(r == exp_a, "classify_slot(%u) = %u (expected %u)", ci, r, exp_a);

  //===-- 场景 A2: 改值后 deactivate + activate → 重新 JIT -------------------===//
  printf("\n--- A2: 改值后重新 JIT (ci=%u) ---\n", ci);
  ejit_deactivate("cell", ci);
  // 全部改成 FD
  for (int s = 0; s < N_COLS; s++) {
    g_slots[ci][s].slotType = 0xFD;
    g_slots[ci][s].priority  = 1;
  }
  ejit_activate("cell", ci);
  r = classify_slot(ci);
  uint32_t exp_a2 = 8 * 1 * 10;  // 8 slots × priority=1 × 10 = 80
  T(r == exp_a2, "classify_slot(%u) after change = %u (expected %u)", ci, r, exp_a2);

  //===-- 场景 B: 2D nested struct access -----------------------------------===//
  printf("\n--- B: 2D nested check_outer2d(ci=%u) ---\n", ci);

  ejit_deactivate("cell", ci);
  g_outer2d[ci][3].inner.mode = 0x55;
  ejit_activate("cell", ci);
  r = check_outer2d(ci);
  T(r == 777, "check_outer2d(%u) mode=0x55 = %u (expected 777)", ci, r);

  // 改值
  ejit_deactivate("cell", ci);
  g_outer2d[ci][3].inner.mode = 0xAA;
  ejit_activate("cell", ci);
  r = check_outer2d(ci);
  T(r == 888, "check_outer2d(%u) mode=0xAA = %u (expected 888)", ci, r);

  // 无匹配
  ejit_deactivate("cell", ci);
  g_outer2d[ci][3].inner.mode = 0x00;
  ejit_activate("cell", ci);
  r = check_outer2d(ci);
  T(r == 0, "check_outer2d(%u) mode=0x00 = %u (expected 0)", ci, r);

  ejit_shutdown();

  printf("\n=== Result: %d failures ===\n", g_fail);
  return g_fail ? 1 : 0;
}
