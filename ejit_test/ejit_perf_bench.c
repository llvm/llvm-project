/**
 * EJIT 性能基准测试 — 验证 JIT 优化后有可量化的性能收益
 *
 * 场景: 结构体中有多个 may_const 字段，循环中大量访问。
 *       JIT 将这些字段替换为常量并折叠分支，消除循环内的
 *       内存读取和条件跳转。
 *
 * 运行:
 *   ./ejit_perf <cellIdx>
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//===-- 数据结构: 6 个 may_const 字段 -----------------------------------------===//

struct CellCfg {
  __attribute__((ejit_may_const)) uint32_t cellType;   // 分支条件
  __attribute__((ejit_may_const)) uint32_t priority;    // 分支条件
  __attribute__((ejit_may_const)) uint32_t maxPower;    // 阈值
  __attribute__((ejit_may_const)) uint32_t minPower;    // 阈值
  __attribute__((ejit_may_const)) uint32_t bandWidth;   // 乘数
  __attribute__((ejit_may_const)) uint32_t timeSlot;    // 除数
  uint32_t result;  // 非 may_const: 写入目标
};

#define N 16
__attribute__((ejit_period_arr("cell"))) struct CellCfg g_cfg[N];

//===-- JIT entry: 循环中大量访问 may_const 字段 -----------------------------===//

#define LOOP_ITERS 20000000ULL  // 20M 次迭代

__attribute__((ejit_entry))
uint64_t compute_cell(
    __attribute__((ejit_period_arr_ind("cell"))) uint8_t ci)
{
  uint64_t sum = 0;
  struct CellCfg *p = &g_cfg[ci];

  for (uint64_t i = 0; i < LOOP_ITERS; i++) {
    // AOT: 每次迭代从内存加载 6 个 may_const 字段 + 多次分支判断
    // JIT: 所有 may_const 替换为常量，分支被折叠，只剩简单算术
    if (p->cellType == 0xFD) {
      if (p->priority > 10) {
        sum += p->maxPower * p->bandWidth / (p->timeSlot + 1);
      } else {
        sum += p->minPower * p->bandWidth / (p->timeSlot + 1);
      }
    } else if (p->cellType == 0xEC) {
      sum += (p->maxPower + p->minPower) * p->bandWidth / (p->timeSlot + 1);
    }
  }
  return sum;
}

// 与 compute_cell 完全相同的逻辑，但不加 EJIT attribute，
// 用于对比纯 AOT（无 wrapper 开销、无 JIT 编译）
uint64_t compute_cell_nojit(uint8_t ci)
{
  uint64_t sum = 0;
  struct CellCfg *p = &g_cfg[ci];

  for (uint64_t i = 0; i < LOOP_ITERS; i++) {
    if (p->cellType == 0xFD) {
      if (p->priority > 10) {
        sum += p->maxPower * p->bandWidth / (p->timeSlot + 1);
      } else {
        sum += p->minPower * p->bandWidth / (p->timeSlot + 1);
      }
    } else if (p->cellType == 0xEC) {
      sum += (p->maxPower + p->minPower) * p->bandWidth / (p->timeSlot + 1);
    }
  }
  return sum;
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

//===-- 计时 -----------------------------------------------------------------===//

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

//===-- main -----------------------------------------------------------------===//

int main(int argc, char **argv) {
  uint8_t ci = (argc >= 2) ? (uint8_t)atoi(argv[1]) : 0;
  int n_warmup = (argc >= 3) ? atoi(argv[2]) : 2;

  printf("=== EJIT Performance Benchmark ===\n");
  printf("cellIdx=%u  iter=%llu  warmup=%d\n\n", ci,
         (unsigned long long)LOOP_ITERS, n_warmup);

  // 初始化: cellType=0xFD, priority=20, 走 then 分支
  g_cfg[ci].cellType  = 0xFD;
  g_cfg[ci].priority  = 20;
  g_cfg[ci].maxPower  = 100;
  g_cfg[ci].minPower  = 50;
  g_cfg[ci].bandWidth = 4;
  g_cfg[ci].timeSlot  = 3;
  g_cfg[ci].result    = 0;

  ejit_init(0);

  printf("--- Pure AOT (无 attribute, 无 wrapper 开销) ---\n");

  uint64_t pure_sum = 0;
  double pure_time = 0;
  for (int w = 0; w < n_warmup; w++) {
    double t0 = now_ms();
    uint64_t r = compute_cell_nojit(ci);
    double t1 = now_ms();
    pure_time = t1 - t0;
    pure_sum = r;
    if (w == n_warmup - 1) {
      printf("  result=%llu  time=%.1f ms\n",
             (unsigned long long)r, pure_time);
    }
  }

  printf("\n--- AOT Fallback (不激活时间窗) ---\n");

  // 不激活 → ejit_compile_or_get 发现 time window not active → fallback
  uint64_t aot_sum = 0;
  double aot_time = 0;
  for (int w = 0; w < n_warmup; w++) {
    double t0 = now_ms();
    uint64_t r = compute_cell(ci);
    double t1 = now_ms();
    aot_time = t1 - t0;
    aot_sum = r;
    if (w == n_warmup - 1) {
      printf("  result=%llu  time=%.1f ms\n",
             (unsigned long long)r, aot_time);
    }
  }

  printf("\n--- JIT Optimized (激活时间窗) ---\n");
  ejit_activate("cell", ci);

  uint64_t jit_sum = 0;
  double jit_time = 0;
  for (int w = 0; w < n_warmup; w++) {
    double t0 = now_ms();
    uint64_t r = compute_cell(ci);
    double t1 = now_ms();
    jit_time = t1 - t0;
    jit_sum = r;
    if (w == n_warmup - 1) {
      printf("  result=%llu  time=%.1f ms\n",
             (unsigned long long)r, jit_time);
    }
  }

  ejit_stats_t s;
  ejit_get_stats(&s);
  printf("\n  JIT stats: entries=%u hits=%llu misses=%llu\n",
         s.entries, (unsigned long long)s.hits, (unsigned long long)s.misses);

  ejit_shutdown();

  // 验证
  if (jit_sum == aot_sum && aot_sum == pure_sum) {
    printf("  Result MATCH: Pure=%llu AOT=%llu JIT=%llu\n",
           (unsigned long long)pure_sum, (unsigned long long)aot_sum,
           (unsigned long long)jit_sum);
  } else {
    printf("  Result MISMATCH: Pure=%llu AOT=%llu JIT=%llu\n",
           (unsigned long long)pure_sum, (unsigned long long)aot_sum,
           (unsigned long long)jit_sum);
  }

  printf("\n  --- 性能对比 ---\n");
  printf("  Pure AOT    (无 wrapper):  %.1f ms\n", pure_time);
  printf("  AOT Fallback (有 wrapper):  %.1f ms\n", aot_time);
  printf("  JIT 优化    (特化编译):    %.1f ms\n", jit_time);

  if (pure_time > 0) {
    printf("  JIT vs Pure AOT 加速: %.1fx\n", pure_time / jit_time);
    printf("  AOT Fallback vs Pure overhead: %.1fx\n", aot_time / pure_time);
  }

  printf("\n=== Benchmark Complete ===\n");
  return 0;
}
