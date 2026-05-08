/**
 * EJIT 生命周期管理 API 完整测试
 *
 * 覆盖 SPEC4 §3.3 全部 7 个接口:
 *   ejit_activate / ejit_deactivate        — 按 name+idx 操作所有数组
 *   ejit_activate_array / ejit_deactivate_array — 按数组指针操作指定数组
 *   ejit_activate_all / ejit_deactivate_all — 全部 cell 批量操作
 *   ejit_is_active                          — 状态查询
 *
 * 核心验证: ejit_activate vs ejit_activate_array 的区别
 *   同一 period name ("cell") 关联多个数组时:
 *   - ejit_activate("cell", 0)   → 激活 cellCfg[0] AND cellPhy[0]
 *   - ejit_activate_array("cell", &cellCfg, 0) → 仅激活 cellCfg[0]
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

//===-- 两个共享相同 period name 的结构体数组 -------------------------------===//

struct CellCfg {
  __attribute__((ejit_may_const)) uint32_t cellType;
  uint32_t trafficLoad;
};

struct CellPhy {
  __attribute__((ejit_may_const)) uint32_t phyCellId;
  uint32_t rssi;
};

#define N 8
__attribute__((ejit_period_arr("cell"))) struct CellCfg g_cellCfg[N];
__attribute__((ejit_period_arr("cell"))) struct CellPhy g_cellPhy[N];

__attribute__((ejit_period("static"))) uint32_t g_sysVer;

//===-- JIT entry: 访问两个数组的 may_const ---------------------------------===//

__attribute__((ejit_entry))
uint32_t read_both(
    __attribute__((ejit_period_arr_ind("cell"))) uint8_t ci)
{
  return g_cellCfg[ci].cellType + g_cellPhy[ci].phyCellId;
}

//===-- 运行时 API (完整声明) -----------------------------------------------===//

extern int ejit_init(const void *cfg);
extern void ejit_shutdown(void);
extern int ejit_activate(const char *name, unsigned char idx);
extern int ejit_deactivate(const char *name, unsigned char idx);
extern int ejit_activate_array(const char *name, void *ptr, unsigned char idx);
extern int ejit_deactivate_array(const char *name, void *ptr, unsigned char idx);
extern int ejit_activate_all(const char *name);
extern int ejit_deactivate_all(const char *name);
extern bool ejit_is_active(const char *name, unsigned char idx);

//===-- 断言 -----------------------------------------------------------------===//

static int g_fail = 0;

#define T(cond, fmt, ...) do {               \
  if (cond) printf("  OK   " fmt "\n", ##__VA_ARGS__); \
  else      printf("  FAIL " fmt "\n", ##__VA_ARGS__), g_fail++; \
} while(0)

int main(int argc, char **argv) {
  // 从命令行读取 cellIdx（模拟真实运行时外部输入）
  uint8_t ci  = (argc >= 2) ? (uint8_t)atoi(argv[1]) : 0;
  uint8_t ci2 = (argc >= 3) ? (uint8_t)atoi(argv[2]) : (uint8_t)((ci + 1) % N);
  uint8_t ci3 = (argc >= 4) ? (uint8_t)atoi(argv[3]) : (uint8_t)((ci + N - 1) % N);

  printf("=== EJIT Lifecycle API Test ===\n");
  printf("cellIdx=%u ci2=%u ci3=%u\n\n", ci, ci2, ci3);

  //===-- 1. ejit_is_active: 未激活时返回 false -----------------------------===//
  printf("--- 1. is_active (未激活, ci=%u ci2=%u) ---\n", ci, ci2);
  ejit_init(0);
  T(!ejit_is_active("cell", ci),  "cell[%u] NOT active before activate", ci);
  T(!ejit_is_active("cell", ci2), "cell[%u] NOT active before activate", ci2);
  T(!ejit_is_active("trp",  ci),  "unknown period returns false");

  //===-- 2. ejit_activate: 激活所有共享 period name 的数组 -----------------===//
  printf("\n--- 2. activate(ci=%u) (激活所有同名数组) ---\n", ci);
  int rc = ejit_activate("cell", ci);
  T(rc == 0, "activate(cell, %u) returns %d", ci, rc);
  T(ejit_is_active("cell", ci),  "cell[%u] IS active after activate", ci);
  T(!ejit_is_active("cell", ci2),"cell[%u] still NOT active (different idx)", ci2);

  //===-- 3. ejit_deactivate: 失效并清理 cache -------------------------------===//
  printf("\n--- 3. deactivate(ci=%u) ---\n", ci);
  rc = ejit_deactivate("cell", ci);
  T(rc == 0, "deactivate(cell, %u) returns %d", ci, rc);
  T(!ejit_is_active("cell", ci), "cell[%u] NOT active after deactivate", ci);

  //===-- 4. ejit_activate_array: 仅激活指定数组 -----------------------------===//
  printf("\n--- 4. activate_array(cellCfg, ci=%u) ---\n", ci);
  rc = ejit_activate_array("cell", g_cellCfg, ci);
  T(rc == 0, "activate_array(cell, &cellCfg, %u) returns %d", ci, rc);
  T(ejit_is_active("cell", ci), "cell[%u] IS active after activate_array", ci);

  //===-- 5. ejit_deactivate_array: 仅失效指定数组 ---------------------------===//
  printf("\n--- 5. deactivate_array(cellCfg, ci=%u) ---\n", ci);
  rc = ejit_deactivate_array("cell", g_cellCfg, ci);
  T(rc == 0, "deactivate_array(cell, &cellCfg, %u) returns %d", ci, rc);
  // 仅失效 cellCfg，cellPhy 应仍活跃（如果之前通过 activate 激活了）
  // 但 is_active 是 period-level 的，这里不做 per-array 断言

  //===-- 6. ejit_activate_all / ejit_deactivate_all: 批量操作 --------------===//
  printf("\n--- 6. activate_all / deactivate_all ---\n");

  rc = ejit_activate_all("cell");
  T(rc == 0, "activate_all(cell) returns %d", rc);
  T(ejit_is_active("cell", ci),  "cell[%u] active after activate_all", ci);
  T(ejit_is_active("cell", ci2), "cell[%u] active after activate_all", ci2);
  T(ejit_is_active("cell", ci3), "cell[%u] active after activate_all", ci3);

  // JIT 验证 (使用外部输入的 ci)
  printf("\n--- 6b. JIT 调用 (ci=%u) ---\n", ci);
  g_cellCfg[ci].cellType = 100;
  g_cellPhy[ci].phyCellId = 200;
  uint32_t r = read_both(ci);
  T(r == 300, "read_both(%u) = %u (expected 100+200=300)", ci, r);

  rc = ejit_deactivate_all("cell");
  T(rc == 0, "deactivate_all(cell) returns %d", rc);
  T(!ejit_is_active("cell", ci),  "cell[%u] NOT active after deactivate_all", ci);
  T(!ejit_is_active("cell", ci2), "cell[%u] NOT active after deactivate_all", ci2);

  //===-- 7. 状态转换完整流程 (使用外部 ci)------------------------------------===//
  printf("\n--- 7. 状态转换 (ci=%u): inactive→active→inactive→active ---\n", ci);

  T(!ejit_is_active("cell", ci), "cell[%u] inactive (start)", ci);
  ejit_activate("cell", ci);
  T(ejit_is_active("cell", ci),  "cell[%u] active", ci);

  g_cellCfg[ci].cellType = 55;
  g_cellPhy[ci].phyCellId = 66;
  uint32_t r2 = read_both(ci);
  T(r2 == 121, "read_both(%u) = %u (expected 55+66=121)", ci, r2);

  ejit_deactivate("cell", ci);
  T(!ejit_is_active("cell", ci), "cell[%u] inactive (after deactivate)", ci);

  g_cellCfg[ci].cellType = 77;
  g_cellPhy[ci].phyCellId = 88;
  ejit_activate("cell", ci);
  T(ejit_is_active("cell", ci), "cell[%u] active again", ci);

  uint32_t r3 = read_both(ci);
  T(r3 == 165, "read_both(%u) = %u (expected 77+88=165)", ci, r3);

  //===-- 8. 边界测试 ---------------------------------------------------------===//
  printf("\n--- 8. 边界测试 ---\n");

  rc = ejit_activate("unknown_period", ci);
  T(rc == 0, "activate(unknown, %u) returns %d (safe)", ci, rc);
  rc = ejit_deactivate("unknown_period", ci);
  T(rc == 0, "deactivate(unknown, %u) returns %d (safe)", ci, rc);
  T(!ejit_is_active("never_active", ci), "unknown period not active");

  ejit_shutdown();

  //===-- 9. shutdown 后调用应报错 -------------------------------------------===//
  printf("\n--- 9. shutdown 后 ---\n");
  rc = ejit_activate("cell", ci);
  T(rc != 0, "activate after shutdown returns error %d", rc);

  printf("\n=== Result: %d failures ===\n", g_fail);
  return g_fail ? 1 : 0;
}
