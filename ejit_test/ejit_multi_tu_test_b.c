/**
 * EJIT 多翻译单元 (multi-TU) 集成测试 — 第二 TU
 *
 * 与 ejit_multi_tu_test.c 配对。本 TU 独立包含一个 ejit_entry 函数 (jit_b) 和它自己的
 * period 数组 g_bCfg("bcell")。两个 TU 各自的 AOT pass 都会发射 bitcode/period 注册项;
 * 修复后它们以 private + 专用 section 形式输出,链接时按 section 合并、不再重复定义。
 *
 * 没有 main —— 由主 TU 调用 jit_b / b_init。
 */

#include <stdint.h>

struct BCfg {
  __attribute__((ejit_may_const)) uint32_t kind;
  uint32_t y;
};

#define N_B 8
__attribute__((ejit_period_arr("bcell"))) struct BCfg g_bCfg[N_B];

__attribute__((ejit_entry))
uint32_t jit_b(__attribute__((ejit_period_arr_ind("bcell"))) uint8_t i)
{
  // JIT 应把 g_bCfg[i].kind 替换为常量并折叠此分支
  if (g_bCfg[i].kind == 0xBB)
    return 200;
  return 2;
}

// Lets the main TU seed this TU's file-local global.
void b_init(uint8_t idx, uint32_t kind)
{
  if (idx < N_B)
    g_bCfg[idx].kind = kind;
}
