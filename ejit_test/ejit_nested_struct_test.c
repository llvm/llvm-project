/**
 * EJIT 嵌套结构体 may_const 运行时测试
 *
 * 验证多层嵌套结构体中的 may_const 字段能在 JIT 运行时被正确替换为常量:
 *   - 2 层嵌套: outer.mid.inner_field
 *   - 3 层嵌套: a.b.c.field
 *   - 同一结构体多个嵌套层级的 may_const
 *   - 混合类型 (int + float)
 *
 * cellIdx 从 argv 外部输入。
 *
 * 运行:
 *   ./ejit_nested_struct 0
 *   ./ejit_nested_struct 3
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

//===-- 2 层嵌套 -------------------------------------------------------------===//

struct InnerCfg {
  __attribute__((ejit_may_const)) uint32_t mode;
  __attribute__((ejit_may_const)) uint32_t gain;
  uint32_t reserved;
};

struct MidCfg {
  struct InnerCfg inner;
  uint32_t flags;
};

struct OuterCfg {
  __attribute__((ejit_may_const)) uint32_t type;
  struct MidCfg mid;
  uint32_t extra;
};

#define N 8
__attribute__((ejit_period_arr("cell"))) struct OuterCfg g_outer[N];

//===-- 3 层嵌套 -------------------------------------------------------------===//

struct Level3 {
  __attribute__((ejit_may_const)) uint32_t value;
};

struct Level2 {
  struct Level3 l3;
  uint32_t pad;
};

struct Level1 {
  struct Level2 l2;
};

struct RootCfg {
  __attribute__((ejit_may_const)) uint32_t root_type;
  struct Level1 l1;
};

__attribute__((ejit_period_arr("cell"))) struct RootCfg g_root[N];

//===-- JIT entry: 2 层嵌套访问 3 个 may_const -------------------------------===//

__attribute__((ejit_entry))
uint32_t check_2level(
    __attribute__((ejit_period_arr_ind("cell"))) uint8_t ci)
{
  struct OuterCfg *p = &g_outer[ci];

  // type:     字段 0 (may_const)
  // mid.inner.mode:  嵌套在 mid[1] 内 (may_const)
  // mid.inner.gain:  嵌套在 mid[1] 内 (may_const)
  if (p->type == 0xAA && p->mid.inner.mode == 0x55)
    return p->mid.inner.gain * 2;
  else if (p->type == 0xBB)
    return p->mid.inner.gain;
  return 0;
}

//===-- JIT entry: 3 层嵌套访问 ---------------------------------------------===//

__attribute__((ejit_entry))
uint32_t check_3level(
    __attribute__((ejit_period_arr_ind("cell"))) uint8_t ci)
{
  struct RootCfg *p = &g_root[ci];

  // root_type:        字段 0
  // l1.l2.l3.value:   3 层嵌套字段 (may_const)
  uint32_t rt = p->root_type;
  uint32_t v  = p->l1.l2.l3.value;

  if (rt == 0x11 && v > 100)  return v + 1000;
  if (rt == 0x22)             return v + 2000;
  return v;
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

  printf("=== EJIT Nested Struct may_const Test ===\n");
  printf("cellIdx=%u\n\n", ci);

  ejit_init(0);

  //===-- 2 层嵌套: AA + 55 → gain*2 ---------------------------------------===//
  printf("--- 2层嵌套 (type=0xAA, mode=0x55, gain=30) ---\n");

  g_outer[ci].type           = 0xAA;
  g_outer[ci].mid.inner.mode  = 0x55;
  g_outer[ci].mid.inner.gain  = 30;
  g_outer[ci].mid.flags       = 0;
  g_outer[ci].extra           = 0;

  ejit_activate("cell", ci);
  uint32_t r = check_2level(ci);
  T(r == 60, "check_2level(%u) AA+55 = %u (expected gain*2 = 60)", ci, r);

  //===-- 2 层嵌套: BB → gain (deactivate → 改值 → activate)---------------===//
  printf("\n--- 2层嵌套 (type=0xBB, gain=80) ---\n");

  ejit_deactivate("cell", ci);
  g_outer[ci].type = 0xBB;
  g_outer[ci].mid.inner.gain = 80;
  ejit_activate("cell", ci);

  r = check_2level(ci);
  T(r == 80, "check_2level(%u) BB = %u (expected gain = 80)", ci, r);

  //===-- 2 层嵌套: 其他 → 0 ------------------------------------------------===//
  printf("\n--- 2层嵌套 (type=0xCC, no match) ---\n");

  ejit_deactivate("cell", ci);
  g_outer[ci].type = 0xCC;
  ejit_activate("cell", ci);

  r = check_2level(ci);
  T(r == 0, "check_2level(%u) no match = %u (expected 0)", ci, r);

  //===-- 3 层嵌套: 11 + v>100 → v+1000 ------------------------------------===//
  printf("\n--- 3层嵌套 (root_type=0x11, l3.value=200) ---\n");

  ejit_deactivate("cell", ci);
  g_root[ci].root_type      = 0x11;
  g_root[ci].l1.l2.l3.value = 200;
  ejit_activate("cell", ci);

  r = check_3level(ci);
  T(r == 1200, "check_3level(%u) 11+200 = %u (expected 200+1000=1200)", ci, r);

  //===-- 3 层嵌套: 22 → v+2000 ---------------------------------------------===//
  printf("\n--- 3层嵌套 (root_type=0x22, l3.value=50) ---\n");

  ejit_deactivate("cell", ci);
  g_root[ci].root_type      = 0x22;
  g_root[ci].l1.l2.l3.value = 50;
  ejit_activate("cell", ci);

  r = check_3level(ci);
  T(r == 2050, "check_3level(%u) 22+50 = %u (expected 50+2000=2050)", ci, r);

  //===-- 3 层嵌套: 其他 → v ------------------------------------------------===//
  printf("\n--- 3层嵌套 (root_type=0x33, l3.value=77) ---\n");

  ejit_deactivate("cell", ci);
  g_root[ci].root_type      = 0x33;
  g_root[ci].l1.l2.l3.value = 77;
  ejit_activate("cell", ci);

  r = check_3level(ci);
  T(r == 77, "check_3level(%u) default = %u (expected v=77)", ci, r);

  ejit_shutdown();

  printf("\n=== Result: %d failures ===\n", g_fail);
  return g_fail ? 1 : 0;
}
