/**
 * EJIT 配置/统计/缓存 API 完整测试
 *
 * 覆盖 SPEC4 §3.2, §3.3, §3.5 中之前缺少测试的接口:
 *   ejit_config_t (自定义配置)
 *   ejit_set_compile_mode / ejit_get_compile_mode
 *   ejit_clear_cache
 *   ejit_invalidate
 *   ejit_get_last_error
 *
 * cellIdx 来自外部输入 (argv)
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//===-- 数据结构 -------------------------------------------------------------===//

struct CellCfg {
  __attribute__((ejit_may_const)) uint32_t cellType;
  uint32_t counter;
};
#define N 8
__attribute__((ejit_period_arr("cell"))) struct CellCfg g_cells[N];

__attribute__((ejit_entry))
uint32_t check_cell(
    __attribute__((ejit_period_arr_ind("cell"))) uint8_t ci)
{
  if (g_cells[ci].cellType == 0xFD) return 100;
  return 0;
}

//===-- 运行时 API -----------------------------------------------------------===//

typedef enum { EJIT_OK_C = 0 } ejit_status_t;
typedef enum { EJIT_COMPILE_SYNC = 0, EJIT_COMPILE_ASYNC = 1 } ejit_compile_mode_t;
typedef enum { EJIT_OPT_L1 = 1, EJIT_OPT_L2 = 2, EJIT_OPT_L3 = 3 } ejit_opt_level_t;

typedef struct {
  ejit_compile_mode_t compileMode;
  ejit_opt_level_t optLevel;
  size_t maxCodeMemory;
  size_t maxDataMemory;
  size_t maxCacheEntries;
  size_t maxCacheSize;
  bool enableLogger;
  const char *dumpJITDir;
} ejit_config_t;

typedef struct {
  size_t entryCount;
  size_t totalCodeSize;
  size_t maxSize;
  uint64_t hits;
  uint64_t misses;
  uint64_t evictions;
} ejit_stats_t;

typedef struct {
  int code;
  const char *message;
  const char *funcName;
} ejit_error_t;

extern ejit_status_t ejit_init(const ejit_config_t *cfg);
extern void ejit_shutdown(void);
extern ejit_status_t ejit_activate(const char *n, unsigned char i);
extern ejit_status_t ejit_deactivate(const char *n, unsigned char i);
extern bool ejit_is_active(const char *n, unsigned char i);
extern void ejit_clear_cache(void);
extern void ejit_invalidate(const char *n, unsigned char i);
extern ejit_status_t ejit_get_stats(ejit_stats_t *s);
extern const ejit_error_t *ejit_get_last_error(void);
extern void ejit_set_compile_mode(ejit_compile_mode_t m);
extern ejit_compile_mode_t ejit_get_compile_mode(void);

//===-- 断言 -----------------------------------------------------------------===//

static int g_fail = 0;
#define T(cond, fmt, ...) do {               \
  if (cond) printf("  OK   " fmt "\n", ##__VA_ARGS__); \
  else      printf("  FAIL " fmt "\n", ##__VA_ARGS__), g_fail++; \
} while(0)

int main(int argc, char **argv) {
  uint8_t ci = (argc >= 2) ? (uint8_t)atoi(argv[1]) : 0;

  printf("=== EJIT Config/Stats/Cache API Test ===\n");
  printf("cellIdx=%u\n\n", ci);

  //===-- 1. 自定义配置初始化 ------------------------------------------------===//
  printf("--- 1. ejit_init with custom config ---\n");

  ejit_config_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.compileMode    = EJIT_COMPILE_SYNC;
  cfg.optLevel       = EJIT_OPT_L2;
  cfg.maxCacheEntries = 32;
  cfg.maxCacheSize    = 256 * 1024;
  cfg.enableLogger    = true;

  int rc = ejit_init(&cfg);
  T(rc == EJIT_OK_C, "ejit_init(custom) returns %d (OK)", rc);

  //===-- 2. compile mode get/set -------------------------------------------===//
  printf("\n--- 2. compile mode get/set ---\n");

  ejit_compile_mode_t m = ejit_get_compile_mode();
  T(m == EJIT_COMPILE_SYNC, "get_compile_mode = %d (SYNC)", m);

  // Only sync mode is supported. Async is excluded in bare-metal builds.
  ejit_set_compile_mode(EJIT_COMPILE_SYNC);
  m = ejit_get_compile_mode();
  T(m == EJIT_COMPILE_SYNC, "get_compile_mode = %d (SYNC after set)", m);

  //===-- 3. get_stats (before any JIT) -------------------------------------===//
  printf("\n--- 3. get_stats (empty) ---\n");

  ejit_stats_t s;
  rc = ejit_get_stats(&s);
  T(rc == EJIT_OK_C, "get_stats returns %d", rc);
  T(s.entryCount == 0, "entries=0 before any JIT");
  T(s.hits == 0, "hits=0");
  T(s.misses == 0, "misses=0");
  T(s.evictions == 0, "evictions=0");

  //===-- 4. get_last_error (empty) -----------------------------------------===//
  printf("\n--- 4. get_last_error (empty) ---\n");

  const ejit_error_t *err = ejit_get_last_error();
  T(err == NULL, "get_last_error = NULL (no errors yet)");

  //===-- 5. JIT 编译 + cache 验证 -------------------------------------------===//
  printf("\n--- 5. JIT compile + cache ---\n");

  g_cells[ci].cellType = 0xFD;
  ejit_activate("cell", ci);

  uint32_t r = check_cell(ci);
  T(r == 100, "check_cell(%u) = %u (expected 100)", ci, r);

  ejit_get_stats(&s);
  T(s.entryCount >= 1, "entries >= 1 after JIT compile (actual %zu)", s.entryCount);
  T(s.misses >= 1, "misses >= 1 (actual %llu)", (unsigned long long)s.misses);

  // 第二次调用: cache hit
  r = check_cell(ci);
  T(r == 100, "check_cell(%u) 2nd call = %u", ci, r);
  ejit_get_stats(&s);
  T(s.hits >= 1, "hits >= 1 (actual %llu)", (unsigned long long)s.hits);

  //===-- 6. ejit_clear_cache -----------------------------------------------===//
  printf("\n--- 6. clear_cache ---\n");

  size_t before = s.entryCount;
  ejit_clear_cache();
  ejit_get_stats(&s);
  T(s.entryCount == 0, "entries=0 after clear_cache (was %zu)", before);

  // 重新 JIT
  r = check_cell(ci);
  T(r == 100, "check_cell(%u) after clear+recompile = %u", ci, r);
  ejit_get_stats(&s);
  T(s.entryCount >= 1, "entries >= 1 after recompile");
  T(s.misses >= 2, "misses increased after clear+recompile");

  //===-- 7. ejit_invalidate (独立失效，不改变状态) ---------------------------===//
  printf("\n--- 7. invalidate (独立失效，状态不变) ---\n");

  bool active = ejit_is_active("cell", ci);
  T(active, "cell[%u] IS active before invalidate", ci);

  before = s.entryCount;
  ejit_invalidate("cell", ci);

  active = ejit_is_active("cell", ci);
  T(active, "cell[%u] still active after invalidate (state unchanged)", ci);

  ejit_get_stats(&s);
  T(s.entryCount == 0, "entries=0 after invalidate (was %zu)", before);

  // 重新 JIT
  r = check_cell(ci);
  T(r == 100, "check_cell(%u) after invalidate+recompile = %u", ci, r);

  //===-- 8. get_last_error after scenario ----------------------------------===//
  printf("\n--- 8. get_last_error (after operations) ---\n");
  // 调用一个不存在的函数来触发错误
  // (直接调用 get_last_error 看看有没有残留错误)
  err = ejit_get_last_error();
  // 可能有日志记录的错误，也可能为 NULL
  printf("  last_error: %s\n", err ? err->message : "(null)");

  //===-- 9. get_stats with NULL pointer ------------------------------------===//
  printf("\n--- 9. get_stats(NULL) ---\n");
  rc = ejit_get_stats(NULL);
  T(rc != EJIT_OK_C, "get_stats(NULL) returns error %d (expected != 0)", rc);

  ejit_shutdown();

  //===-- 10. 反初始化后调用应报错 -------------------------------------------===//
  printf("\n--- 10. after shutdown ---\n");

  m = ejit_get_compile_mode();
  T(m == EJIT_COMPILE_SYNC, "get_compile_mode after shutdown = %d (safe default SYNC)", m);

  rc = ejit_get_stats(&s);
  T(rc != EJIT_OK_C, "get_stats after shutdown returns error %d", rc);

  err = ejit_get_last_error();
  T(err == NULL, "get_last_error after shutdown = NULL");

  ejit_clear_cache();  // should not crash
  printf("  clear_cache after shutdown: no crash\n");

  printf("\n=== Result: %d failures ===\n", g_fail);
  return g_fail ? 1 : 0;
}
