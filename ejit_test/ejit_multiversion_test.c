/**
 * EJIT 多版本 Cache 测试 — 同一函数在不同 cellIdx 下产生独立特化版本
 *
 * 验证:
 *   1. f(cellIdx=A) 首次调用 → JIT miss, entries+1
 *   2. f(cellIdx=B) 首次调用 → JIT miss, entries+1 (新特化版本)
 *   3. f(cellIdx=A) 再次调用 → cache HIT (命中之前编译的版本)
 *   4. 每个版本产生正确结果 (取决于各自 cellIdx 对应的 may_const 值)
 *   5. 所有 cellIdx 均来自外部输入 (argv)
 *
 * 运行:
 *   ./ejit_multiversion 0 3 7     # 3 个不同 cellIdx
 *   ./ejit_multiversion 1 5 9 13  # 4 个不同 cellIdx
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

//===-- 数据结构 -------------------------------------------------------------===//

struct CellCfg {
  __attribute__((ejit_may_const)) uint32_t cellType;
  __attribute__((ejit_may_const)) uint32_t cellId;
  uint32_t counter;
};

#define N 16
__attribute__((ejit_period_arr("cell"))) struct CellCfg g_cells[N];

//===-- JIT entry: 分支依赖 may_const 字段 -----------------------------------===//

__attribute__((ejit_entry))
uint32_t classify_cell(
    __attribute__((ejit_period_arr_ind("cell"))) uint8_t ci)
{
  uint32_t t = g_cells[ci].cellType;

  // 根据 cellType 返回不同值，JIT 应折叠此分支
  if (t == 0xFD)       return 100;
  else if (t == 0xEC)  return 200;
  else if (t == 0xAA)  return 300;
  else                 return 0;
}

//===-- 运行时 API -----------------------------------------------------------===//

extern int ejit_init(const void *cfg);
extern int ejit_activate(const char *name, unsigned char idx);
extern int ejit_deactivate(const char *name, unsigned char idx);
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

//===-- 测试 -----------------------------------------------------------------===//

static int g_fail = 0;

static void init_cell(uint8_t ci, uint32_t cellType) {
  g_cells[ci].cellType = cellType;
  g_cells[ci].cellId   = ci * 100;
  g_cells[ci].counter  = 0;
  ejit_activate("cell", ci);
}

static void check(const char *label, int cond,
                  const char *fmt, ...) {
  if (cond) printf("  OK   %s\n", label);
  else      printf("  FAIL %s\n", label), g_fail++;
  (void)fmt;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    printf("Usage: %s <cellIdx1> <cellIdx2> [cellIdx3 ...]\n", argv[0]);
    return 2;
  }

  printf("=== EJIT Multi-Version Cache Test ===\n");
  printf("cellIdx args:");
  for (int i = 1; i < argc; i++)
    printf(" %s", argv[i]);
  printf("\n\n");

  ejit_init(0);

  // 为每个外部输入的 cellIdx 设置数据并激活
  for (int i = 1; i < argc; i++) {
    uint8_t ci = (uint8_t)atoi(argv[i]);
    init_cell(ci, 0xFD + i - 1); // 第1个=0xFD, 第2个=0xFE, 第3个=0xFF
  }

  // --- 第一轮: 每个 cellIdx 首次调用 (全部应该 miss) ---
  printf("--- 第一轮: 首次调用 (全部 miss) ---\n");
  ejit_stats_t s0;
  ejit_get_stats(&s0);

  for (int i = 1; i < argc; i++) {
    uint8_t ci = (uint8_t)atoi(argv[i]);
    uint32_t r = classify_cell(ci);
    uint32_t expected = (g_cells[ci].cellType == 0xFD) ? 100 :
                        (g_cells[ci].cellType == 0xEC) ? 200 :
                        (g_cells[ci].cellType == 0xAA) ? 300 : 0;
    check("", r == expected, "classify(%u)=%u expected=%u", ci, r, expected);
  }

  ejit_stats_t s1;
  ejit_get_stats(&s1);

  // 每个不同的 cellIdx 应该产生一个独立的 cache entry
  int n_unique = argc - 1;
  check("entries == n_unique", (int)s1.entries == n_unique,
        "entries=%u expected=%d", s1.entries, n_unique);
  check("misses >= n_unique", s1.misses >= (uint64_t)n_unique,
        "misses=%llu expected>=%d",
        (unsigned long long)s1.misses, n_unique);

  printf("  stats: entries=%u hits=%llu misses=%llu\n",
         s1.entries, (unsigned long long)s1.hits, (unsigned long long)s1.misses);

  // --- 第二轮: 再次调用相同 cellIdx (全部应该 hit) ---
  printf("\n--- 第二轮: 再次调用相同 cellIdx (全部 hit) ---\n");

  for (int i = 1; i < argc; i++) {
    uint8_t ci = (uint8_t)atoi(argv[i]);
    uint32_t r = classify_cell(ci);
    uint32_t expected = (g_cells[ci].cellType == 0xFD) ? 100 :
                        (g_cells[ci].cellType == 0xEC) ? 200 :
                        (g_cells[ci].cellType == 0xAA) ? 300 : 0;
    check("", r == expected, "classify(%u)=%u  (cache hit)", ci, r);
  }

  ejit_stats_t s2;
  ejit_get_stats(&s2);

  // entries 不变，hits 增加
  check("entries unchanged", (int)s2.entries == n_unique,
        "entries=%u (was %u)", s2.entries, s1.entries);
  check("hits increased", s2.hits >= s1.hits + (uint64_t)n_unique,
        "hits=%llu (was %llu)",
        (unsigned long long)s2.hits, (unsigned long long)s1.hits);
  check("misses unchanged", (int)s2.misses == (int)s1.misses,
        "misses=%llu (was %llu)",
        (unsigned long long)s2.misses, (unsigned long long)s1.misses);

  printf("  stats: entries=%u hits=%llu misses=%llu\n",
         s2.entries, (unsigned long long)s2.hits, (unsigned long long)s2.misses);

  // --- 第三轮: 修改某个 cellIdx 的数据，deactivate → 重新 JIT ---
  if (argc >= 3) {
    uint8_t ci0 = (uint8_t)atoi(argv[1]);
    printf("\n--- 第三轮: deactivate(%u) → 改值 → activate → 重新 JIT ---\n", ci0);

    ejit_deactivate("cell", ci0);
    uint64_t misses_before = s2.misses;
    uint64_t entries_before = s2.entries;

    // 把 cellType 改成 0xEC (原来可能是 0xFD)
    g_cells[ci0].cellType = 0xEC;
    ejit_activate("cell", ci0);

    uint32_t r = classify_cell(ci0);
    check("new value after recompile", r == 200,
          "classify(%u)=%u (expected 200, now cellType=0xEC)", ci0, r);

    ejit_stats_t s3;
    ejit_get_stats(&s3);
    check("new miss triggered", s3.misses > misses_before,
          "misses %llu -> %llu",
          (unsigned long long)misses_before, (unsigned long long)s3.misses);
    check("entries stable after recompile", s3.entries == entries_before,
          "entries=%u (was %u: old evicted, new added)",
          s3.entries, entries_before);
  }

  ejit_shutdown();

  printf("\n=== Result: %d failures ===\n", g_fail);
  return g_fail ? 1 : 0;
}
