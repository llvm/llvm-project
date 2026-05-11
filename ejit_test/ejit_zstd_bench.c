//===-- ejit_zstd_bench.c - EJIT + zstd compression benchmark --------------===//
//
// Validates EJIT with a real-world C library (zstd).
//
// Design:
//   - Compression parameters (level, windowLog, etc.) are ejit_may_const
//     fields in a global struct marked ejit_period(static).
//   - The compress() function is ejit_entry — JIT specializes it for the
//     current parameter values, replacing loads with constants.
//   - After InstCombine + Inline, the constant values propagate into
//     zstd's internal code, eliminating dead branches.
//
// Usage:
//   ./ejit_zstd_bench [rounds]  — runs compression with JIT specialization
//   rounds=3 (default): tests levels 1, 6, 12 with different data sizes
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <string.h>
#include <time.h>

#define ejit_may_const          __attribute__((ejit_may_const))
#define ejit_period(x)          __attribute__((ejit_period(#x)))
#define ejit_entry              __attribute__((ejit_entry))

#include "llvm/ExecutionEngine/EJIT/EJitRuntime.h"
#include "zstd.h"

//===--- EJIT-annotated compression parameters ----------------------------===//

typedef struct {
  int ejit_may_const compressionLevel;   // 1..22
  int ejit_may_const windowLog;          // 10..31
  int ejit_may_const chainLog;           // 6..30
  int ejit_may_const hashLog;            // 6..30
  int ejit_may_const searchLog;          // 1..10
  int ejit_may_const minMatch;           // 3..7
  int ejit_may_const targetLength;       // 0..large
  int ejit_may_const strategy;           // ZSTD_fast..ZSTD_btultra2
} CompressCfg;

ejit_period(static) CompressCfg g_cfg;

// Pre-allocated compression context (reused across calls).
// Not annotated — this is runtime mutable state.
static ZSTD_CCtx *g_cctx = NULL;

//===--- EJIT entry: compress with specialized parameters -----------------===//

ejit_entry size_t compress_specialized(
    const void *src, size_t srcSize,
    void *dst, size_t dstCapacity)
{
  ZSTD_CCtx_reset(g_cctx, ZSTD_reset_session_and_parameters);

  // These reads will be replaced with constants by JIT PASS6.
  // After InstCombine+Inline, the constants propagate into zstd internals.
  ZSTD_CCtx_setParameter(g_cctx, ZSTD_c_compressionLevel, g_cfg.compressionLevel);
  ZSTD_CCtx_setParameter(g_cctx, ZSTD_c_windowLog,       g_cfg.windowLog);
  ZSTD_CCtx_setParameter(g_cctx, ZSTD_c_chainLog,        g_cfg.chainLog);
  ZSTD_CCtx_setParameter(g_cctx, ZSTD_c_hashLog,         g_cfg.hashLog);
  ZSTD_CCtx_setParameter(g_cctx, ZSTD_c_searchLog,       g_cfg.searchLog);
  ZSTD_CCtx_setParameter(g_cctx, ZSTD_c_minMatch,        g_cfg.minMatch);
  ZSTD_CCtx_setParameter(g_cctx, ZSTD_c_targetLength,    g_cfg.targetLength);
  ZSTD_CCtx_setParameter(g_cctx, ZSTD_c_strategy,        g_cfg.strategy);

  return ZSTD_compress2(g_cctx, dst, dstCapacity, src, srcSize);
}

//===--- Non-EJIT baseline (same logic, no JIT) ---------------------------===//

size_t compress_baseline(
    const void *src, size_t srcSize,
    void *dst, size_t dstCapacity)
{
  ZSTD_CCtx_reset(g_cctx, ZSTD_reset_session_and_parameters);
  ZSTD_CCtx_setParameter(g_cctx, ZSTD_c_compressionLevel, g_cfg.compressionLevel);
  ZSTD_CCtx_setParameter(g_cctx, ZSTD_c_windowLog,       g_cfg.windowLog);
  ZSTD_CCtx_setParameter(g_cctx, ZSTD_c_chainLog,        g_cfg.chainLog);
  ZSTD_CCtx_setParameter(g_cctx, ZSTD_c_hashLog,         g_cfg.hashLog);
  ZSTD_CCtx_setParameter(g_cctx, ZSTD_c_searchLog,       g_cfg.searchLog);
  ZSTD_CCtx_setParameter(g_cctx, ZSTD_c_minMatch,        g_cfg.minMatch);
  ZSTD_CCtx_setParameter(g_cctx, ZSTD_c_targetLength,    g_cfg.targetLength);
  ZSTD_CCtx_setParameter(g_cctx, ZSTD_c_strategy,        g_cfg.strategy);
  return ZSTD_compress2(g_cctx, dst, dstCapacity, src, srcSize);
}

//===--- Test helpers -----------------------------------------------------===//

static int g_failures = 0;
#define CHECK(cond, msg) do { \
  if (!(cond)) { g_failures++; printf("  FAIL: %s\n", msg); } \
  else printf("  OK  : %s\n", msg); \
} while(0)

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// Generate compressible test data with repeating patterns
static void gen_data(void *buf, size_t size, int seed) {
  unsigned char *p = (unsigned char *)buf;
  const char *pattern = "The quick brown fox jumps over the lazy dog. "
                         "Lorem ipsum dolor sit amet, consectetur. "
                         "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnop. ";
  size_t plen = strlen(pattern);
  for (size_t i = 0; i < size; i++) {
    // 70% repeating pattern, 30% variation to keep things interesting
    if ((i + seed) % 10 < 7)
      p[i] = pattern[(i + seed * 31) % plen];
    else
      p[i] = (unsigned char)(((i * 7 + seed * 13) & 0x3f) + 0x20);
  }
}

// Configure compression parameters for a given "preset"
static void set_preset(int level) {
  g_cfg.compressionLevel = level;
  g_cfg.windowLog     = (level >= 10) ? 24 : 20;
  g_cfg.chainLog      = (level >= 10) ? 28 : 22;
  g_cfg.hashLog       = (level >= 10) ? 26 : 20;
  g_cfg.searchLog     = (level >= 10) ? 6  : 4;
  g_cfg.minMatch      = (level >= 10) ? 4  : 3;
  g_cfg.targetLength  = (level <= 3)  ? 16 : 0;
  g_cfg.strategy      = (level <= 3)  ? ZSTD_fast :
                        (level <= 7)  ? ZSTD_lazy :
                        (level <= 12) ? ZSTD_btlazy2 :
                                        ZSTD_btopt;
}

//===--- Main --------------------------------------------------------------===//

int main(int argc, char **argv) {
  int rounds = (argc > 1) ? atoi(argv[1]) : 3;
  if (rounds < 1) rounds = 1;
  if (rounds > 5) rounds = 5;

  printf("=== EJIT + zstd Compression Benchmark ===\n");
  printf("Rounds: %d\n\n", rounds);

  // Init EJIT
  ejit_config_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.compileMode = EJIT_COMPILE_SYNC;
  cfg.optLevel = EJIT_OPT_L2;

  ejit_status_t st = ejit_init(&cfg);
  CHECK(st == EJIT_OK, "ejit_init");

  // Create zstd compression context
  g_cctx = ZSTD_createCCtx();
  CHECK(g_cctx != NULL, "ZSTD_createCCtx");

  // Test data sizes
  size_t sizes[] = { 1024, 16384, 65536 };
  const char *size_names[] = { "1KB", "16KB", "64KB" };
  int levels[] = { 1, 6, 12 };

  for (int r = 0; r < rounds; r++) {
    int level = levels[r % 3];

    printf("--- Round %d: compression level %d ---\n", r + 1, level);
    set_preset(level);
    ejit_clear_cache();

    for (int s = 0; s < 3; s++) {
      size_t srcSize = sizes[s];
      size_t dstCap = ZSTD_compressBound(srcSize);

      void *src = malloc(srcSize);
      void *dst_jit = malloc(dstCap);
      void *dst_base = malloc(dstCap);

      gen_data(src, srcSize, level * 100 + s);
      memset(dst_jit, 0, dstCap);
      memset(dst_base, 0, dstCap);

      // --- EJIT path: first call triggers JIT compilation ---
      double t1 = now_ms();
      size_t r1 = compress_specialized(src, srcSize, dst_jit, dstCap);
      double t2 = now_ms();
      double jit_time = t2 - t1;

      // --- EJIT path: second call should hit the JIT cache ---
      double t3 = now_ms();
      size_t r2 = compress_specialized(src, srcSize, dst_jit, dstCap);
      double t4 = now_ms();
      double cache_time = t4 - t3;

      // --- Baseline: same logic without JIT (for correctness check) ---
      size_t r_base = compress_baseline(src, srcSize, dst_base, dstCap);

      // Correctness: JIT result must match baseline
      int ok = (r1 == r_base) && (r2 == r_base) &&
               !ZSTD_isError(r1) && !ZSTD_isError(r_base);
      const char *label = size_names[s];
      printf("  %4s: JIT=%zu baseline=%zu  (1st:%.2fms  2nd:%.2fms)",
             label, r1, r_base, jit_time, cache_time);

      if (ok) {
        printf("  OK\n");
        // For small data, zstd frame header can cause expansion.
        // Only check compression ratio for larger inputs.
        if (srcSize >= 16384)
          CHECK(r1 < srcSize, "compressed smaller than source (>=16KB)");
      } else {
        printf("  FAIL\n");
        g_failures++;
        fprintf(stderr, "  JIT error: %s\n", ZSTD_getErrorName(r1));
      }

      free(src); free(dst_jit); free(dst_base);
    }
  }

  // Stats
  printf("\n--- EJIT Stats ---\n");
  ejit_stats_t stats;
  memset(&stats, 0, sizeof(stats));
  if (ejit_get_stats(&stats) == EJIT_OK) {
    printf("  cache entries  : %zu\n", stats.entryCount);
    printf("  cache hits     : %" PRIu64 "\n", stats.hits);
    printf("  cache misses   : %" PRIu64 "\n", stats.misses);
    printf("  total code size: %zu bytes\n", stats.totalCodeSize);
  }

  ZSTD_freeCCtx(g_cctx);
  ejit_shutdown();

  printf("\n=== %s: %d failures ===\n",
         g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures > 0 ? 1 : 0;
}
