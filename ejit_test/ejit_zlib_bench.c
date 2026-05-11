//===-- ejit_zlib_bench.c - EJIT + zlib compression benchmark -------------===//
//
// Validates EJIT with zlib (the most widely deployed C compression library).
//
// Unlike the zstd test which needed a separate build, zlib is available as
// a system library on virtually every platform — no extra dependencies.
//
// Design:
//   - Compression level (1-9) and strategy are ejit_may_const fields in
//     a global struct marked ejit_period(static).
//   - The compress() function is ejit_entry — JIT specializes it for the
//     current parameters.
//   - After JIT compilation, constant-parameter branches are eliminated.
//
//===----------------------------------------------------------------------===//

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <zlib.h>

#include "llvm/ExecutionEngine/EJIT/EJitRuntime.h"

//===--- EJIT-annotated compression parameters ----------------------------===//

typedef struct {
  int ejit_may_const level;         // 1..9
  int ejit_may_const strategy;      // Z_DEFAULT_STRATEGY, Z_FILTERED, etc.
  int ejit_may_const memLevel;      // 1..9
  int ejit_may_const windowBits;    // 9..15
} CompressCfg;

ejit_period(static) CompressCfg g_cfg;

//===--- EJIT entry: compress with specialized parameters -----------------===//

ejit_entry int compress_specialized(
    const unsigned char *src, size_t srcLen,
    unsigned char *dst, size_t *dstLen)
{
  z_stream strm;
  memset(&strm, 0, sizeof(strm));
  strm.next_in = (unsigned char *)src;
  strm.avail_in = (uInt)srcLen;
  strm.next_out = dst;
  strm.avail_out = (uInt)*dstLen;

  // These reads become constants after JIT specialization.
  // zlib's deflateInit2 branches on level/strategy/memLevel/windowBits
  // internally — the constants propagate and dead branches are eliminated.
  int rc = deflateInit2(&strm, g_cfg.level, Z_DEFLATED,
                        g_cfg.windowBits, g_cfg.memLevel, g_cfg.strategy);
  if (rc != Z_OK) return rc;

  rc = deflate(&strm, Z_FINISH);
  if (rc != Z_STREAM_END) { deflateEnd(&strm); return (rc == Z_OK) ? Z_DATA_ERROR : rc; }

  *dstLen = strm.total_out;
  deflateEnd(&strm);
  return Z_OK;
}

//===--- Non-EJIT baseline -------------------------------------------------===//

int compress_baseline(
    const unsigned char *src, size_t srcLen,
    unsigned char *dst, size_t *dstLen)
{
  z_stream strm;
  memset(&strm, 0, sizeof(strm));
  strm.next_in = (unsigned char *)src;
  strm.avail_in = (uInt)srcLen;
  strm.next_out = dst;
  strm.avail_out = (uInt)*dstLen;

  int rc = deflateInit2(&strm, g_cfg.level, Z_DEFLATED,
                        g_cfg.windowBits, g_cfg.memLevel, g_cfg.strategy);
  if (rc != Z_OK) return rc;

  rc = deflate(&strm, Z_FINISH);
  if (rc != Z_STREAM_END) { deflateEnd(&strm); return (rc == Z_OK) ? Z_DATA_ERROR : rc; }

  *dstLen = strm.total_out;
  deflateEnd(&strm);
  return Z_OK;
}

//===--- Helpers -----------------------------------------------------------===//

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

static void gen_data(void *buf, size_t size, int seed) {
  unsigned char *p = (unsigned char *)buf;
  const char *pat = "The quick brown fox jumps over the lazy dog. "
                    "Lorem ipsum dolor sit amet, consectetur adipiscing. "
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ";
  size_t plen = strlen(pat);
  for (size_t i = 0; i < size; i++)
    p[i] = (i + seed) % 10 < 7 ? pat[(i + seed * 31) % plen]
                               : (unsigned char)((i * 7 + seed * 13) & 0x1f) + 0x40;
}

static void set_preset(int level) {
  g_cfg.level = level;
  g_cfg.strategy = (level <= 3)  ? Z_DEFAULT_STRATEGY :
                   (level <= 6)  ? Z_FILTERED :
                                   Z_HUFFMAN_ONLY;
  g_cfg.memLevel = 8;
  g_cfg.windowBits = 15;
}

//===--- Main --------------------------------------------------------------===//

int main(int argc, char **argv) {
  int rounds = (argc > 1) ? atoi(argv[1]) : 3;
  if (rounds < 1) rounds = 1;
  if (rounds > 5) rounds = 5;

  printf("=== EJIT + zlib Compression Benchmark ===\nRounds: %d\n\n", rounds);

  ejit_config_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.compileMode = EJIT_COMPILE_SYNC;
  cfg.optLevel = EJIT_OPT_L2;
  CHECK(ejit_init(&cfg) == EJIT_OK, "ejit_init");

  size_t sizes[] = { 1024, 16384, 65536 };
  const char *labels[] = { "1KB", "16KB", "64KB" };
  int levels[] = { 1, 6, 9 };

  for (int r = 0; r < rounds; r++) {
    int level = levels[r % 3];
    printf("--- Round %d: level %d, strategy=%d ---\n", r + 1, level, g_cfg.strategy);
    set_preset(level);
    ejit_clear_cache();

    for (int s = 0; s < 3; s++) {
      size_t srcSize = sizes[s];
      size_t dstCap = compressBound((uLong)srcSize);

      void *src = malloc(srcSize);
      void *dst_jit = malloc(dstCap);
      void *dst_base = malloc(dstCap);

      gen_data(src, srcSize, level * 100 + s);
      size_t jitLen = dstCap, baseLen = dstCap;

      // First call: triggers JIT compilation
      double t1 = now_ms();
      int r1 = compress_specialized(src, srcSize, dst_jit, &jitLen);
      double t2 = now_ms();

      // Second call: should hit JIT cache
      double t3 = now_ms();
      int r2 = compress_specialized(src, srcSize, dst_jit, &jitLen);
      double t4 = now_ms();

      // Baseline
      int rb = compress_baseline(src, srcSize, dst_base, &baseLen);

      int ok = (r1 == Z_OK) && (r2 == Z_OK) && (rb == Z_OK) &&
               (jitLen == baseLen) && !memcmp(dst_jit, dst_base, jitLen);

      printf("  %4s: JIT=%zuB base=%zuB  (1st:%.2fms 2nd:%.2fms)",
             labels[s], jitLen, baseLen, t2 - t1, t4 - t3);

      if (ok) {
        printf("  OK\n");
        CHECK(jitLen < srcSize, "compressed smaller than source");
      } else {
        printf("  FAIL\n");
        g_failures++;
      }

      free(src); free(dst_jit); free(dst_base);
    }
  }

  // Stats
  ejit_stats_t stats;
  memset(&stats, 0, sizeof(stats));
  if (ejit_get_stats(&stats) == EJIT_OK) {
    printf("\n--- EJIT Stats ---\n");
    printf("  cache entries : %zu\n", stats.entryCount);
    printf("  cache hits    : %" PRIu64 "\n", stats.hits);
    printf("  cache misses  : %" PRIu64 "\n", stats.misses);
    printf("  code size     : %zu bytes\n", stats.totalCodeSize);
  }

  ejit_shutdown();

  printf("\n=== %s: %d failures ===\n",
         g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures > 0 ? 1 : 0;
}
