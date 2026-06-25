/**
 * EJIT 多翻译单元 (multi-TU) 集成测试 — 主 TU
 *
 * 目的: 复现并验证最初报的链接错误已修复:
 *   "ld.lld: error: duplicate symbol __ejit_registry_bitcode / __ejit_registry_period"
 * 当多个 .c 文件各自包含 ejit_entry 函数 + period 全局变量时,旧的 AOT pass 会在每个
 * TU 里用固定外部名发射强符号 __ejit_registry_{bitcode,period},多 TU 链接即冲突。
 *
 * 本测试由两个 TU 组成:
 *   - ejit_multi_tu_test.c   (本文件): ejit_entry jit_a + period 数组 g_aCfg("acell")
 *   - ejit_multi_tu_test_b.c (第二 TU): ejit_entry jit_b + period 数组 g_bCfg("bcell")
 * 两者一起编译链接。
 *
 * 验证点:
 *   1. 能成功链接 (修复前这一步就会 duplicate symbol 失败)
 *   2. 两个 TU 的 ejit_entry 都被 JIT 编译 (entries >= 2),证明两个 TU 各自的
 *      bitcode/period 注册表在链接后都被运行时正确解析 (跨 TU 合并)
 *   3. 两个函数的特化结果正确
 *
 * 运行:  ./ejit_multi_tu_test <idxA> <idxB>     (默认 0 0)
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

//===-- TU-A 的 EJIT 数据与入口 --------------------------------------------===//

struct ACfg {
  __attribute__((ejit_may_const)) uint32_t kind;
  uint32_t x;
};

#define N_A 8
__attribute__((ejit_period_arr("acell"))) struct ACfg g_aCfg[N_A];

__attribute__((ejit_entry))
uint32_t jit_a(__attribute__((ejit_period_arr_ind("acell"))) uint8_t i)
{
  // JIT 应把 g_aCfg[i].kind 替换为常量并折叠此分支
  if (g_aCfg[i].kind == 0xAA)
    return 100;
  return 1;
}

//===-- 第二 TU 的接口 -----------------------------------------------------===//

extern uint32_t jit_b(uint8_t i);       // ejit_entry in the other TU
extern void b_init(uint8_t idx, uint32_t kind);  // sets g_bCfg[idx].kind

//===-- 运行时 API ---------------------------------------------------------===//

extern int ejit_init(const void *cfg);
extern int ejit_activate(const char *name, unsigned char idx);
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
extern int ejit_get_stats(ejit_stats_t *stats);

//===-- 断言 ---------------------------------------------------------------===//

static int g_failures = 0;

#define VERIFY(cond, fmt, ...) do {                  \
  if (!(cond)) {                                     \
    printf("  FAIL: " fmt "\n", ##__VA_ARGS__);      \
    g_failures++;                                    \
  } else {                                           \
    printf("  OK:   " fmt "\n", ##__VA_ARGS__);      \
  }                                                  \
} while (0)

//===-- main ---------------------------------------------------------------===//

int main(int argc, char **argv)
{
  uint8_t ia = (argc >= 2) ? (uint8_t)atoi(argv[1]) : 0;
  uint8_t ib = (argc >= 3) ? (uint8_t)atoi(argv[2]) : 0;
  if (ia >= N_A) ia = 0;

  printf("=== EJIT multi-TU registry test ===\n");
  printf("idxA=%u  idxB=%u\n\n", ia, ib);

  // Seed data in both TUs.
  g_aCfg[ia].kind = 0xAA;     // TU-A global (this file)
  b_init(ib, 0xBB);           // TU-B global (other file)

  int rc = ejit_init(0);
  VERIFY(rc == 0, "ejit_init returned %d", rc);

  ejit_activate("acell", ia);
  ejit_activate("bcell", ib);

  //=== Both entry functions execute and specialize correctly ===
  printf("\n--- 两个 TU 的 ejit_entry 都应被 JIT 编译 ---\n");

  uint32_t ra1 = jit_a(ia);
  VERIFY(ra1 == 100, "jit_a(%u) = %u (expected 100, kind=0xAA)", ia, ra1);

  uint32_t rb1 = jit_b(ib);
  VERIFY(rb1 == 200, "jit_b(%u) = %u (expected 200, kind=0xBB)", ib, rb1);

  ejit_stats_t s1;
  ejit_get_stats(&s1);
  printf("  stats: entries=%u hits=%llu misses=%llu\n",
         s1.entries, (unsigned long long)s1.hits, (unsigned long long)s1.misses);

  // The key multi-TU assertion: BOTH TUs' bitcode got registered and JIT'd.
  // If the two TUs' registries had collided / dropped one, only one function
  // would produce a JIT entry.
  VERIFY(s1.entries >= 2, "JIT entries >= 2 (both TUs compiled, actual %u)",
         s1.entries);
  VERIFY(s1.misses >= 2, "JIT misses >= 2 (both first calls, actual %llu)",
         (unsigned long long)s1.misses);

  //=== Second calls hit the cache (specialized code was stored) ===
  printf("\n--- 二次调用应命中缓存 ---\n");

  uint32_t ra2 = jit_a(ia);
  uint32_t rb2 = jit_b(ib);
  VERIFY(ra2 == 100, "jit_a(%u) 2nd = %u", ia, ra2);
  VERIFY(rb2 == 200, "jit_b(%u) 2nd = %u", ib, rb2);

  ejit_stats_t s2;
  ejit_get_stats(&s2);
  printf("  stats: entries=%u hits=%llu misses=%llu\n",
         s2.entries, (unsigned long long)s2.hits, (unsigned long long)s2.misses);
  VERIFY(s2.hits >= 2, "Cache hits >= 2 (both 2nd calls, actual %llu)",
         (unsigned long long)s2.hits);

  ejit_shutdown();

  printf("\n=== %s (%d failures) ===\n",
         g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}
