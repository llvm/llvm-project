/**
 * EJIT 运行时 trace 测试 — 验证 JIT 优化流程是否按设计执行
 *
 * 编译:
 *   build/bin/clang -c -O2 ejit_trace_test.c -o ejit_trace_test.o
 * 链接:
 *   LIBS=$(ls build_x86/lib/x.a | sort)
 *   clang++ -Os -Wl,--gc-sections -Wl,--strip-all ejit_trace_test.o \
 *     -Wl,--whole-archive $LIBS -Wl,--no-whole-archive \
 *     -lpthread -ldl -o ejit_trace_test
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

//===-- EJIT 属性 -----------------------------------------------------------===//

struct BoardConfig {
    __attribute__((ejit_may_const)) uint32_t boardType;
    uint32_t xx;
};

struct CellConfig {
    __attribute__((ejit_may_const)) uint32_t cellType;
    __attribute__((ejit_may_const)) uint32_t cellId;
    uint32_t trafficLoad;
};

struct TrpConfig {
    __attribute__((ejit_may_const)) uint32_t trpType;
    uint32_t status;
};

//===-- 全局变量 -------------------------------------------------------------===//

__attribute__((ejit_period("static"))) struct BoardConfig g_boardCfg;

#define N_CELL 16
__attribute__((ejit_period_arr("cell"))) struct CellConfig g_cellCfg[N_CELL];

#define M_TRP 8
__attribute__((ejit_period_arr("trp"))) struct TrpConfig g_trpCfg[M_TRP];

//===-- 场景 1: 仅 static 时间窗 ----------------------------------------------===//

__attribute__((ejit_entry))
void process_board(void)
{
    if (g_boardCfg.boardType == 0x01) {
        g_boardCfg.xx = 100;
    } else {
        g_boardCfg.xx = 200;
    }
}

//===-- 场景 2: static + cell -----------------------------------------------===//

__attribute__((ejit_entry))
void process_cell(
    __attribute__((ejit_period_arr_ind("cell"))) uint8_t cellIdx)
{
    if (g_cellCfg[cellIdx].cellType == 0xFD) {
        g_cellCfg[cellIdx].trafficLoad += 5;
    } else {
        g_cellCfg[cellIdx].trafficLoad += 15;
    }
}

__attribute__((ejit_period_lc("cell")))
void update_cell_config(
    __attribute__((ejit_period_arr_ind("cell"))) uint8_t cellIdx)
{
    g_cellCfg[cellIdx].cellType = 0xFD;
    g_cellCfg[cellIdx].cellId   = 42;
}

//===-- 场景 3: static + cell + trp -----------------------------------------===//

__attribute__((ejit_entry))
void process_trp_task(
    __attribute__((ejit_period_arr_ind("cell"))) uint8_t cellIdx,
    __attribute__((ejit_period_arr_ind("trp")))  uint8_t trpIdx)
{
    uint32_t cell_t = g_cellCfg[cellIdx].cellType;
    uint32_t trp_t  = g_trpCfg[trpIdx].trpType;
    if (cell_t == 0xFD && trp_t == 1) {
        g_cellCfg[cellIdx].trafficLoad += 5;
    } else if (cell_t == 0xEC && trp_t == 2) {
        g_cellCfg[cellIdx].trafficLoad += 15;
    }
}

__attribute__((ejit_period_lc("cell")))
__attribute__((ejit_period_lc("trp")))
void update_all_config(
    __attribute__((ejit_period_arr_ind("cell"))) uint8_t cellIdx,
    __attribute__((ejit_period_arr_ind("trp")))  uint8_t trpIdx)
{
    g_cellCfg[cellIdx].cellType = 0xEC;
    g_trpCfg[trpIdx].trpType   = 2;
}

//===-- 运行时 API (from EJitRuntime.h) ------------------------------------===//

extern int ejit_init(const void *cfg);
extern int ejit_activate(const char *name, unsigned char idx);
extern int ejit_deactivate(const char *name, unsigned char idx);
extern int ejit_is_active(const char *name, unsigned char idx);
extern void ejit_shutdown(void);
extern int ejit_get_stats(void *stats);

//===-- main: 运行时 trace --------------------------------------------------===//

int main(void)
{
    printf("=== EJIT Runtime Trace Test ===\n\n");

    // --- 1. 初始化 ---
    printf("[1] ejit_init...\n");
    int rc = ejit_init(0);
    printf("    ejit_init => %d %s\n\n", rc, rc == 0 ? "(OK)" : "(FAIL)");

    // --- 2. 设置全局变量初值 ---
    g_boardCfg.boardType = 0x01;
    g_boardCfg.xx = 0;

    // cell[0] 初始化为常量值
    g_cellCfg[0].cellType = 0xFD;
    g_cellCfg[0].cellId   = 1;
    g_cellCfg[0].trafficLoad = 0;

    // trp[0] 初始化
    g_trpCfg[0].trpType = 1;
    g_trpCfg[0].status  = 0;

    printf("[2] 初始化全局变量\n");
    printf("    g_boardCfg.boardType = 0x%x\n", g_boardCfg.boardType);
    printf("    g_cellCfg[0].cellType = 0x%x\n", g_cellCfg[0].cellType);
    printf("    g_trpCfg[0].trpType = %d\n\n", g_trpCfg[0].trpType);

    // --- 3. 激活时间窗 ---
    printf("[3] 激活时间窗...\n");
    rc = ejit_activate("cell", 0);
    printf("    ejit_activate(\"cell\", 0) => %d, is_active=%d\n",
           rc, ejit_is_active("cell", 0));
    rc = ejit_activate("trp", 0);
    printf("    ejit_activate(\"trp\", 0)  => %d, is_active=%d\n\n",
           rc, ejit_is_active("trp", 0));

    // --- 4. 调用 ejit_entry 函数 (JIT dispatch) ---
    printf("[4] 调用 ejit_entry 函数 (首次触发 JIT dispatch)...\n");

    printf("    process_board()...\n");
    process_board();
    printf("    => g_boardCfg.xx = %d (期望 100, boardType=0x01 走 fast path)\n",
           g_boardCfg.xx);

    printf("    process_cell(0)...\n");
    process_cell(0);
    printf("    => g_cellCfg[0].trafficLoad = %d (期望 5, cellType=0xFD 走 +5 分支)\n",
           g_cellCfg[0].trafficLoad);

    printf("    process_trp_task(0, 0)...\n");
    process_trp_task(0, 0);
    printf("    => g_cellCfg[0].trafficLoad = %d (期望 10, cell=0xFD && trp=1 走 +5)\n\n",
           g_cellCfg[0].trafficLoad);

    // --- 5. 生命周期管理 ---
    printf("[5] 生命周期管理: update_cell_config(1)...\n");
    printf("    修改前: g_cellCfg[1].cellType = 0x%x\n", g_cellCfg[1].cellType);
    update_cell_config(1);
    printf("    修改后: g_cellCfg[1].cellType = 0x%x, cellId = %d\n",
           g_cellCfg[1].cellType, g_cellCfg[1].cellId);
    printf("    (PASS4 应在函数入口插入 deactivate_array, 出口插入 activate_array)\n\n");

    printf("[6] 多时间窗生命周期: update_all_config(0, 0)...\n");
    printf("    修改前: g_cellCfg[0].cellType=0x%x, g_trpCfg[0].trpType=%d\n",
           g_cellCfg[0].cellType, g_trpCfg[0].trpType);
    update_all_config(0, 0);
    printf("    修改后: g_cellCfg[0].cellType=0x%x, g_trpCfg[0].trpType=%d\n\n",
           g_cellCfg[0].cellType, g_trpCfg[0].trpType);

    // --- 6. 再次调用 (走 cache hit 或重新 JIT) ---
    printf("[7] 再次调用 (时间窗已更新)...\n");
    g_cellCfg[0].trafficLoad = 0; // reset
    ejit_activate("cell", 0);
    printf("    ejit_activate(\"cell\", 0) => is_active=%d\n",
           ejit_is_active("cell", 0));
    process_cell(0);
    printf("    => g_cellCfg[0].trafficLoad = %d\n\n",
           g_cellCfg[0].trafficLoad);

    // --- 7. 关闭 ---
    printf("[8] ejit_shutdown...\n");
    ejit_shutdown();
    printf("    Done.\n\n");

    printf("=== Trace Test Complete ===\n");
    return 0;
}
