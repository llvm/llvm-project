/**
 * EJIT 耗时统计测试 — 测量 EJIT 各阶段的时间开销
 *
 * 测量项:
 *   1. Wrapper overhead:   wrapper 入口到 cache 查找完成的时间
 *   2. JIT compile time:   首次调用触发 JIT 编译的耗时
 *   3. Cache hit latency:  Cache 命中时的调用延迟
 *   4. Fallback latency:   时间窗未激活时的 fallback 延迟
 *   5. Activate/Deactivate: 生命周期 API 耗时
 *
 * 运行:
 *   ./ejit_timing_test <cellIdx>
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

//===-- EJIT 属性 -----------------------------------------------------------===//

struct CellCfg {
  __attribute__((ejit_may_const)) uint32_t cellType;
  __attribute__((ejit_may_const)) uint32_t priority;
  __attribute__((ejit_may_const)) uint32_t maxPower;
  uint32_t result;
};

#define N 16
__attribute__((ejit_period_arr("cell"))) struct CellCfg g_cfg[N];

//===-- JIT entry (简单函数，减少自身计算对测量的干扰) ----------------------===//

// 场景 A: 无维度参数 (仅依赖 static)
__attribute__((ejit_entry))
uint32_t simple_jit(void) {
  return 42;
}

// 场景 B: 单维度参数
__attribute__((ejit_entry))
uint32_t cell_jit(
    __attribute__((ejit_period_arr_ind("cell"))) uint8_t ci)
{
  if (g_cfg[ci].cellType == 0xFD) {
    return g_cfg[ci].maxPower * 2;
  }
  return g_cfg[ci].maxPower;
}

// 场景 C: 双维度参数
struct TrpCfg {
  __attribute__((ejit_may_const)) uint32_t trpType;
  uint32_t status;
};
#define M 8
__attribute__((ejit_period_arr("trp"))) struct TrpCfg g_trp[M];

__attribute__((ejit_entry))
uint32_t multi_jit(
    __attribute__((ejit_period_arr_ind("cell"))) uint8_t ci,
    __attribute__((ejit_period_arr_ind("trp")))  uint8_t ti)
{
  if (g_cfg[ci].cellType == 0xFD && g_trp[ti].trpType == 1) {
    return g_cfg[ci].maxPower * g_trp[ti].trpType;
  }
  return g_cfg[ci].maxPower;
}

// 对照组: 与 cell_jit 相同逻辑，无 EJIT attribute
uint32_t cell_nojit(uint8_t ci) {
  if (g_cfg[ci].cellType == 0xFD) {
    return g_cfg[ci].maxPower * 2;
  }
  return g_cfg[ci].maxPower;
}

//===-- 运行时 API -----------------------------------------------------------===//

extern int ejit_init(const void *cfg);
extern int ejit_activate(const char *name, unsigned char idx);
extern int ejit_deactivate(const char *name, unsigned char idx);
extern int ejit_is_active(const char *name, unsigned char idx);
extern void ejit_shutdown(void);

typedef struct {
  uint32_t entries;
  uint64_t codeSize;
  uint64_t maxSize;
  uint64_t hits;
  uint64_t misses;
  uint64_t evictions;
  uint32_t active;
} ejit_stats_t;

extern int ejit_get_stats(ejit_stats_t *s);

//===-- 高精度计时 (batch timing, 摊销 clock_gettime 开销) -----------------===//

static double now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e9 + ts.tv_nsec;
}

#define WARMUP 3

// Batch timing: measure N iterations as one block to amortize the
// ~20-40ns clock_gettime overhead.  For functions in the 20-80ns range,
// per-call timing would make the timer the dominant noise source.
static void batch_call(const char *label, uint32_t (*fn)(uint8_t),
                       uint8_t arg, uint64_t iters) {
  double min_batch = 1e18;
  double sum = 0;
  int batches = 5;
  volatile uint64_t sink = 0;  // prevent loop-invariant hoisting
  for (int w = 0; w < WARMUP; w++) fn(arg);
  for (int b = 0; b < batches; b++) {
    double t0 = now_ns();
    for (uint64_t i = 0; i < iters; i++) {
      sink += fn(arg);
    }
    double t1 = now_ns();
    double batch_ns = t1 - t0;
    if (batch_ns < min_batch) min_batch = batch_ns;
    sum += batch_ns;
  }
  uint64_t avg_per_call = (uint64_t)((sum / batches) / iters);
  uint64_t best_per_call = (uint64_t)(min_batch / iters);
  printf("  %-30s  avg=%6llu ns/call  (best batch %llu)\n", label,
         (unsigned long long)avg_per_call, (unsigned long long)best_per_call);
}

static void batch_fn0(const char *label, uint32_t (*fn)(void),
                      uint64_t iters) {
  double min_batch = 1e18;
  double sum = 0;
  int batches = 5;
  volatile uint64_t sink = 0;
  for (int w = 0; w < WARMUP; w++) fn();
  for (int b = 0; b < batches; b++) {
    double t0 = now_ns();
    for (uint64_t i = 0; i < iters; i++) {
      sink += fn();
    }
    double t1 = now_ns();
    double batch_ns = t1 - t0;
    if (batch_ns < min_batch) min_batch = batch_ns;
    sum += batch_ns;
  }
  uint64_t avg_per_call = (uint64_t)((sum / batches) / iters);
  uint64_t best_per_call = (uint64_t)(min_batch / iters);
  printf("  %-30s  avg=%6llu ns/call  (best batch %llu)\n", label,
         (unsigned long long)avg_per_call, (unsigned long long)best_per_call);
}

static void batch_fn2(const char *label, uint32_t (*fn)(uint8_t, uint8_t),
                      uint8_t a, uint8_t b, uint64_t iters) {
  double min_batch = 1e18;
  double sum = 0;
  int batches = 5;
  volatile uint64_t sink = 0;
  for (int w = 0; w < WARMUP; w++) fn(a, b);
  for (int v = 0; v < batches; v++) {
    double t0 = now_ns();
    for (uint64_t i = 0; i < iters; i++) {
      sink += fn(a, b);
    }
    double t1 = now_ns();
    double batch_ns = t1 - t0;
    if (batch_ns < min_batch) min_batch = batch_ns;
    sum += batch_ns;
  }
  uint64_t avg_per_call = (uint64_t)((sum / batches) / iters);
  uint64_t best_per_call = (uint64_t)(min_batch / iters);
  printf("  %-30s  avg=%6llu ns/call  (best batch %llu)\n", label,
         (unsigned long long)avg_per_call, (unsigned long long)best_per_call);
}

#define ITERS_LARGE  1000000ULL
#define ITERS_SMALL    10000ULL

int main(int argc, char **argv) {
  uint8_t ci = (argc >= 2) ? (uint8_t)atoi(argv[1]) : 0;
  uint8_t ti = (argc >= 3) ? (uint8_t)atoi(argv[2]) : 0;

  printf("=== EJIT Timing Test (batch amortized) ===\n");
  printf("cellIdx=%u trpIdx=%u\n\n", ci, ti);

  // Initialize data
  for (int i = 0; i < N; i++) {
    g_cfg[i].cellType = 0xFD;
    g_cfg[i].priority = 20;
    g_cfg[i].maxPower = 100;
    g_cfg[i].result = 0;
  }
  for (int i = 0; i < M; i++) {
    g_trp[i].trpType = 1;
    g_trp[i].status = 0;
  }

  ejit_init(0);

  //=== 1. Baseline: plain function call (no EJIT) ===//
  printf("--- [1] Baseline: plain function (no EJIT, %llu iters/batch) ---\n",
         (unsigned long long)ITERS_LARGE);
  batch_call("cell_nojit", cell_nojit, ci, ITERS_LARGE);

  //=== 2. Fallback: wrapper + not active ===//
  // Explicit precondition: no time window has been activated yet,
  // so ejit_compile_or_get returns NULL → wrapper falls through to AOT.
  printf("\n--- [2] Fallback: wrapper + not active (%llu iters/batch) ---\n",
         (unsigned long long)ITERS_LARGE);
  if (!ejit_is_active("cell", ci))
    printf("  (precondition OK: cell not yet active)\n");
  batch_call("cell_jit (fallback)", cell_jit, ci, ITERS_LARGE);

  //=== 3. First-call JIT compile (sync) ===//
  // Activate before timing so JIT compile time is measured cleanly.
  ejit_activate("cell", ci);
  printf("\n--- [3] First-call JIT compile (cold, activate done beforehand) ---\n");
  {
    double t0 = now_ns();
    uint32_t r = cell_jit(ci);
    double t1 = now_ns();
    printf("  first call:  %llu ns  (JIT compile + first call)\n",
           (unsigned long long)(uint64_t)(t1 - t0));
    printf("  result=%u\n", r);
  }

  ejit_stats_t s;
  ejit_get_stats(&s);
  printf("  stats: entries=%u hits=%llu misses=%llu\n",
         s.entries, (unsigned long long)s.hits, (unsigned long long)s.misses);

  //=== 4. Cache hit latency ===//
  printf("\n--- [4] Cache hit path (%llu iters/batch) ---\n",
         (unsigned long long)ITERS_LARGE);
  batch_call("cell_jit (cache hit)", cell_jit, ci, ITERS_LARGE);

  ejit_get_stats(&s);
  printf("  stats: entries=%u hits=%llu misses=%llu\n\n",
         s.entries, (unsigned long long)s.hits, (unsigned long long)s.misses);

  //=== 5. Multi-dim ===//
  printf("--- [5] Multi-dim (cell+trp, %llu iters/batch) ---\n",
         (unsigned long long)ITERS_SMALL);
  ejit_activate("trp", ti);
  {
    double t0 = now_ns();
    uint32_t r = multi_jit(ci, ti);
    double t1 = now_ns();
    printf("  first call (cold): %llu ns\n",
           (unsigned long long)(uint64_t)(t1 - t0));
    (void)r;
  }
  batch_fn2("multi_jit (cache hit)", multi_jit, ci, ti, ITERS_SMALL);

  //=== 6. No-dim function ===//
  printf("\n--- [6] No-dim function (static only, %llu iters/batch) ---\n",
         (unsigned long long)ITERS_SMALL);
  {
    double t0 = now_ns();
    uint32_t r = simple_jit();
    double t1 = now_ns();
    printf("  first call (cold): %llu ns\n",
           (unsigned long long)(uint64_t)(t1 - t0));
    (void)r;
  }
  batch_fn0("simple_jit (cache hit)", simple_jit, ITERS_SMALL);

  //=== 7. Activate/Deactivate overhead ===//
  printf("\n--- [7] Activate/Deactivate API ---\n");

  // 7a. Single-shot deactivate with cache entry present.
  // This triggers the real invalidateByPeriod path (periodIndex_ scan + cache erase).
  // Cannot batch because the first deactivate removes the cache entry.
  {
    double t0 = now_ns();
    ejit_deactivate("cell", ci);
    double t1 = now_ns();
    printf("  deactivate (w/ cache):    %llu ns  (single-shot)\n",
           (unsigned long long)(uint64_t)(t1 - t0));
  }

  // 7b. Re-activate, re-JIT to rebuild cache, then measure activate on
  //     already-active window (no-op path).
  ejit_activate("cell", ci);
  cell_jit(ci);  // re-compile to rebuild cache entry
  {
    double t0 = now_ns();
    for (uint64_t i = 0; i < ITERS_SMALL; i++)
      ejit_activate("cell", ci);   // already active → no-op path
    double t1 = now_ns();
    printf("  activate (already active): avg=%llu ns/call\n",
           (unsigned long long)((uint64_t)(t1 - t0) / ITERS_SMALL));
  }

  //=== Summary ===//
  ejit_get_stats(&s);
  printf("\n=== Summary ===\n");
  printf("  Cache entries:  %u\n", s.entries);
  printf("  Cache hits:     %llu\n", (unsigned long long)s.hits);
  printf("  Cache misses:   %llu\n", (unsigned long long)s.misses);
  printf("  Evictions:      %llu\n", (unsigned long long)s.evictions);

  ejit_shutdown();

  printf("\n=== Timing Test Complete ===\n");
  return 0;
}
