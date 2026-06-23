/**
 * EJIT L3 嵌套循环折叠测试 — fold-to-constant
 *
 * 场景: 一个嵌套 for 循环，循环上界来自 may_const 字段 (outer_n / inner_n)。
 *       在 L3 下，IndVarSimplify 用 SCEV 求出累加器的闭式出口值，
 *       LoopDeletion 删除空循环，最终整个函数折叠成单个常量返回。
 *
 *   sum = Σ_{i=0..99} Σ_{j=0..99} (i*100 + j) = 49995000
 *
 * 仅 L3 能折叠: L1/L2 没有循环 pass，LoopFullUnroll 单独也会因 100 次
 * 迭代超出展开预算而放弃，函数仍是带 phi 的循环。
 *
 * 运行:
 *   ./ejit_fold_loop_test 0
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//===-- 周期数组配置 ---------------------------------------------------------===//

struct LoopFoldCfg {
  __attribute__((ejit_may_const)) uint32_t outer_n;
  __attribute__((ejit_may_const)) uint32_t inner_n;
};

#define N 16
__attribute__((ejit_period_arr("loopfold"))) struct LoopFoldCfg g_loopfold[N];

//===-- JIT entry: 嵌套循环，应折叠为常量 -----------------------------------===//

__attribute__((ejit_entry))
uint64_t fold_loop(
    __attribute__((ejit_period_arr_ind("loopfold"))) uint8_t ci)
{
  uint64_t sum = 0;
  for (uint32_t i = 0; i < g_loopfold[ci].outer_n; i++)
    for (uint32_t j = 0; j < g_loopfold[ci].inner_n; j++)
      sum += (uint64_t)i * g_loopfold[ci].inner_n + j;
  return sum;
}

//===-- 运行时 API -----------------------------------------------------------===//

typedef enum { EJIT_OK = 0 }            ejit_status_t;
typedef enum { EJIT_COMPILE_SYNC = 0 } ejit_compile_mode_t;
typedef enum { EJIT_OPT_L1 = 1, EJIT_OPT_L2 = 2, EJIT_OPT_L3 = 3 } ejit_opt_level_t;

typedef struct {
  ejit_compile_mode_t compileMode;
  ejit_opt_level_t    optLevel;
  size_t maxCodeMemory, maxDataMemory, maxCacheEntries, maxCacheSize;
  bool   enableLogger;
  const char *dumpJITDir;
} ejit_config_t;

typedef struct {
  size_t entryCount, totalCodeSize, maxSize;
  unsigned long long hits, misses, evictions;
} ejit_stats_t;

extern ejit_status_t ejit_init(const ejit_config_t *cfg);
extern void          ejit_shutdown(void);
extern ejit_status_t ejit_activate(const char *name, unsigned char idx);
extern ejit_status_t ejit_get_stats(ejit_stats_t *s);

//===-- 断言 -----------------------------------------------------------------===//

static int g_fail = 0;
#define T(cond, fmt, ...) do {                          \
  if (cond) printf("  OK   " fmt "\n", ##__VA_ARGS__);  \
  else      printf("  FAIL " fmt "\n", ##__VA_ARGS__), g_fail++; \
} while(0)

int main(int argc, char **argv) {
  uint8_t ci = (argc >= 2) ? (uint8_t)atoi(argv[1]) : 0;

  const uint32_t OUTER    = 100;
  const uint32_t INNER    = 100;
  const uint64_t EXPECTED = 49995000ULL;

  printf("=== EJIT L3 Loop-Fold Test ===\n");
  printf("cellIdx=%u  outer=%u inner=%u\n\n", ci, OUTER, INNER);

  g_loopfold[ci].outer_n = OUTER;
  g_loopfold[ci].inner_n = INNER;

  // 显式选择 L3 — 仅 L3 引入循环 pass (LoopSimplify/Unroll/IndVarSimplify/...)
  ejit_config_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.compileMode     = EJIT_COMPILE_SYNC;
  cfg.optLevel        = EJIT_OPT_L3;
  cfg.maxCodeMemory   = 512 * 1024;
  cfg.maxDataMemory   = 256 * 1024;
  cfg.maxCacheEntries = 64;
  cfg.maxCacheSize    = 1024 * 1024;

  T(ejit_init(&cfg) == EJIT_OK, "ejit_init(L3)");
  ejit_activate("loopfold", ci);

  uint64_t r = fold_loop(ci);
  T(r == EXPECTED, "fold_loop(%u) = %llu (expected %llu)",
    ci, (unsigned long long)r, (unsigned long long)EXPECTED);

  ejit_stats_t s;
  memset(&s, 0, sizeof(s));
  ejit_get_stats(&s);
  T(s.entryCount > 0, "JIT active (entries=%zu)", s.entryCount);

  ejit_shutdown();

  printf("\n=== Result: %d failures ===\n", g_fail);
  return g_fail ? 1 : 0;
}
