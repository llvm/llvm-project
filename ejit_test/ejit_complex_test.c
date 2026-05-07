/**
 * EJIT 复杂测试用例 — cellIdx 从外部输入
 *
 * 编译:
 *   build/bin/clang -c -O2 ejit_complex_test.c -o ejit_complex_test.o
 * 链接:
 *   LIBS=$(echo build_x86/lib/x.a); clang++ ... -Wl,--whole-archive $LIBS -Wl,--no-whole-archive
 *   clang++ -Os -Wl,--gc-sections -Wl,--strip-all ejit_complex_test.o \
 *     -Wl,--whole-archive $ALL_LIBS -Wl,--no-whole-archive \
 *     -lpthread -ldl -o ejit_complex_test
 * 运行:
 *   ./ejit_complex_test 0 1 2    # 测试 cell=0, trp=1, slice=2
 *   ./ejit_complex_test 3 0       # 不提供 slice 则默认 0
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

//===-- 外部 API (声明提前，供辅助函数使用) ----------------------------------===//

extern int  ejit_init(const void *cfg);
extern int  ejit_activate(const char *name, unsigned char idx);
extern int  ejit_deactivate(const char *name, unsigned char idx);
extern int  ejit_is_active(const char *name, unsigned char idx);
extern void ejit_shutdown(void);

typedef struct { size_t entryCount; size_t totalCodeSize; size_t maxSize; unsigned long long hits, misses, evictions; } ejit_stats_t;
extern int ejit_get_stats(ejit_stats_t *stats);

//===-- 结构体定义 (多类型 may_const) -----------------------------------------===//

struct PhyConfig {
    __attribute__((ejit_may_const)) uint32_t phyMode;   // TDD/FDD
    __attribute__((ejit_may_const)) uint32_t carrierFreq;
    __attribute__((ejit_may_const)) uint32_t bandwidth;
    uint32_t rxGain;                                     // 可变
    uint32_t txPower;
};

struct CellConfig {
    __attribute__((ejit_may_const)) uint32_t cellType;
    __attribute__((ejit_may_const)) uint32_t cellId;
    __attribute__((ejit_may_const)) uint32_t pci;        // Physical Cell ID
    uint32_t trafficLoad;
    struct {
        __attribute__((ejit_may_const)) uint32_t freqBand;
        __attribute__((ejit_may_const)) uint32_t antennaPorts;
        uint32_t currentPower;
    } rf;
};

struct TrpConfig {
    __attribute__((ejit_may_const)) uint32_t trpType;
    __attribute__((ejit_may_const)) uint32_t beamCount;
    uint32_t activeBeams;
};

struct SliceConfig {
    __attribute__((ejit_may_const)) uint32_t sliceType;  // eMBB/URLLC/mMTC
    __attribute__((ejit_may_const)) uint32_t priority;
    __attribute__((ejit_may_const)) float    latencyTarget;
    uint32_t throughput;
};

//===-- 全局变量: 多数组共享同一时间窗名 -----------------------------------------===//

__attribute__((ejit_period("static"))) struct PhyConfig g_phyCfg;

#define N_CELL  16
#define M_TRP   8
#define K_SLICE 4
#define P_CARRIER 4

// cell 时间窗: 两个数组共享 "cell" 名称
__attribute__((ejit_period_arr("cell"))) struct CellConfig  g_cellCfg[N_CELL];
__attribute__((ejit_period_arr("cell"))) struct PhyConfig   g_cellPhy[N_CELL];

__attribute__((ejit_period_arr("trp")))  struct TrpConfig   g_trpCfg[M_TRP];
__attribute__((ejit_period_arr("slice"))) struct SliceConfig g_sliceCfg[K_SLICE];
__attribute__((ejit_period_arr("carrier"))) struct PhyConfig  g_carrierCfg[P_CARRIER];

//===-- 场景 A: 4 维度 max (cell + trp + slice + carrier) ---------------------===//

/**
 * 极限测试: 同时依赖 4 个时间窗数组 (SPEC4 规定的最大值)。
 * JIT 时 4 个维度参数的运行值确定，对应的 may_const 字段全部特化。
 */
__attribute__((ejit_entry))
int process_multi_dim(
    __attribute__((ejit_period_arr_ind("cell")))    uint8_t cellIdx,
    __attribute__((ejit_period_arr_ind("trp")))     uint8_t trpIdx,
    __attribute__((ejit_period_arr_ind("slice")))   uint8_t sliceIdx,
    __attribute__((ejit_period_arr_ind("carrier"))) uint8_t carrierIdx)
{
    int result = 0;

    // PHY 层决策 (依赖 static)
    if (g_phyCfg.phyMode == 0x01) {          // TDD
        result += 100;
    } else {                                  // FDD
        result += 200;
    }

    // Cell 层 (依赖 cell[cellIdx])
    if (g_cellCfg[cellIdx].cellType == 0xFD) {
        result += 10;
        // 嵌套 rf 字段
        if (g_cellCfg[cellIdx].rf.freqBand == 78) {
            result += 1;
        }
    } else if (g_cellCfg[cellIdx].cellType == 0xEC) {
        result += 20;
    }

    // Cell PHY (共享 "cell" 时间窗 — cellPhy 也随 cellIdx 特化)
    if (g_cellPhy[cellIdx].carrierFreq == 3500) {
        result += 5;
    }

    // TRP 层 (依赖 trp[trpIdx])
    if (g_trpCfg[trpIdx].trpType == 1) {
        result += 50;
        if (g_trpCfg[trpIdx].beamCount == 64) {
            result += 2;
        }
    }

    // Slice 层 (依赖 slice[sliceIdx])
    switch (g_sliceCfg[sliceIdx].sliceType) {
    case 1: // eMBB
        result += 1000;
        if (g_sliceCfg[sliceIdx].latencyTarget > 10.0f) {
            result += 1;
        }
        break;
    case 2: // URLLC
        result += 2000;
        break;
    case 3: // mMTC
        result += 3000;
        break;
    default:
        result -= 1;
    }

    // Carrier 层 (依赖 carrier[carrierIdx])
    if (g_carrierCfg[carrierIdx].bandwidth == 100) {
        result += 500;
    }

    return result;
}

//===-- 场景 B: 循环 + switch + early return ---------------------------------===//

__attribute__((ejit_entry))
uint32_t process_all_trps(
    __attribute__((ejit_period_arr_ind("cell"))) uint8_t cellIdx)
{
    uint32_t total = 0;
    uint32_t cellType = g_cellCfg[cellIdx].cellType;

    // 遍历所有 TRP (不使用 trp ind — 遍历全部)
    for (uint8_t t = 0; t < M_TRP; t++) {
        // trp 不是 ejit_period_arr_ind，所以 trpType 不是 JIT 常量
        // 但 cellType 是 JIT 常量 → switch 可被部分折叠
        switch (cellType) {
        case 0xFD:
            total += g_trpCfg[t].activeBeams * 2;
            break;
        case 0xEC:
            total += g_trpCfg[t].activeBeams * 3;
            break;
        default:
            total += g_trpCfg[t].activeBeams;
        }
    }

    // early return 测试 — 生命周期 hook 是否在所有 return 点插入
    if (total == 0)
        return 0;

    return total + g_cellCfg[cellIdx].pci;
}

//===-- 场景 C: 循环展开边界 (L3 Optimize) ------------------------------------===//

/**
 * 小循环 + may_const 界限: JIT 在 L3 优化等级可完全展开。
 */
__attribute__((ejit_entry))
uint32_t process_slice_loop(
    __attribute__((ejit_period_arr_ind("slice"))) uint8_t sliceIdx)
{
    uint32_t sum = 0;

    // sliceType 是 JIT 常量 → switch 可完全折叠
    switch (g_sliceCfg[sliceIdx].sliceType) {
    case 1: // eMBB: 处理 4 个 carrier
        for (uint8_t i = 0; i < 4; i++) {
            sum += g_carrierCfg[i].bandwidth;
        }
        break;
    case 2: // URLLC: low latency path
        sum += g_carrierCfg[0].bandwidth;
        break;
    case 3: // mMTC: 处理所有 carrier
        for (uint8_t i = 0; i < P_CARRIER; i++) {
            sum += g_carrierCfg[i].bandwidth * 2;
        }
        break;
    }

    return sum;
}

//===-- 场景 D: 复杂生命周期 — 多 return + 多时间窗 ---------------------------===//

/**
 * 生命周期函数有多个 return 点。
 * PASS4 必须在每个 return 前插入 activate (逆序)。
 */
__attribute__((ejit_period_lc("cell")))
__attribute__((ejit_period_lc("trp")))
int reconfig_cell_trp(
    __attribute__((ejit_period_arr_ind("cell"))) uint8_t cellIdx,
    __attribute__((ejit_period_arr_ind("trp")))  uint8_t trpIdx,
    uint32_t newCellType,
    uint32_t newTrpType)
{
    // 参数校验 — 可能会 early return
    if (newCellType > 0xFF) {
        return -1;  // PASS4 必须在此处也插入 activate
    }

    if (newTrpType > 10) {
        return -2;  // PASS4 必须在此处也插入 activate
    }

    // 实际修改
    g_cellCfg[cellIdx].cellType = newCellType;
    g_cellCfg[cellIdx].rf.currentPower = 30;
    g_trpCfg[trpIdx].trpType = newTrpType;
    g_trpCfg[trpIdx].activeBeams = 0;

    return 0;
}

//===-- 场景 E: 生命周期 — 仅修改非 may_const 字段 ----------------------------===//

/**
 * 即使只修改非 may_const 字段，如果标记了 ejit_period_lc，
 * 仍然触发 deactivate/activate。
 */
__attribute__((ejit_period_lc("trp")))
void update_trp_power(
    __attribute__((ejit_period_arr_ind("trp"))) uint8_t trpIdx,
    uint32_t power)
{
    g_trpCfg[trpIdx].activeBeams = power;  // 非 may_const 字段
}

//===-- 辅助函数 ------------------------------------------------------------===//

static void init_globals(void)
{
    g_phyCfg.phyMode     = 0x01;  // TDD
    g_phyCfg.carrierFreq = 3500;
    g_phyCfg.bandwidth   = 100;

    for (int i = 0; i < N_CELL; i++) {
        g_cellCfg[i].cellType     = (i % 2) ? 0xFD : 0xEC;
        g_cellCfg[i].cellId       = i;
        g_cellCfg[i].pci          = i * 10;
        g_cellCfg[i].trafficLoad  = 0;
        g_cellCfg[i].rf.freqBand      = 78;
        g_cellCfg[i].rf.antennaPorts   = 4;
        g_cellCfg[i].rf.currentPower   = 23;

        g_cellPhy[i].carrierFreq = 3500;
        g_cellPhy[i].bandwidth   = 100;
    }

    for (int i = 0; i < M_TRP; i++) {
        g_trpCfg[i].trpType     = (i < 4) ? 1 : 2;
        g_trpCfg[i].beamCount   = 64;
        g_trpCfg[i].activeBeams = 16;
    }

    for (int i = 0; i < K_SLICE; i++) {
        g_sliceCfg[i].sliceType     = i + 1;       // 1=eMBB, 2=URLLC, 3=mMTC, 4=other
        g_sliceCfg[i].priority      = i;
        g_sliceCfg[i].latencyTarget = (i == 2) ? 1.0f : 20.0f;
        g_sliceCfg[i].throughput    = 0;
    }

    for (int i = 0; i < P_CARRIER; i++) {
        g_carrierCfg[i].bandwidth = 100;
    }
}

static void activate_all(void)
{
    for (int i = 0; i < N_CELL; i++)
        ejit_activate("cell", i);
    for (int i = 0; i < M_TRP; i++)
        ejit_activate("trp", i);
    for (int i = 0; i < K_SLICE; i++)
        ejit_activate("slice", i);
    for (int i = 0; i < P_CARRIER; i++)
        ejit_activate("carrier", i);
}

//===-- main --------------------------------------------------------------===//

int main(int argc, char **argv)
{
    // 从命令行读取 cellIdx / trpIdx / sliceIdx / carrierIdx
    uint8_t cellIdx    = (argc > 1) ? (uint8_t)atoi(argv[1]) : 0;
    uint8_t trpIdx     = (argc > 2) ? (uint8_t)atoi(argv[2]) : 0;
    uint8_t sliceIdx   = (argc > 3) ? (uint8_t)atoi(argv[3]) : 0;
    uint8_t carrierIdx = (argc > 4) ? (uint8_t)atoi(argv[4]) : 0;

    printf("=== EJIT Complex Test ===\n");
    printf("indices: cell=%u trp=%u slice=%u carrier=%u\n\n",
           cellIdx, trpIdx, sliceIdx, carrierIdx);

    init_globals();
    ejit_init(0);
    activate_all();

    // --- 场景 A: 4 维度 ---
    printf("--- 场景A: 4-dim process_multi_dim ---\n");
    int r = process_multi_dim(cellIdx, trpIdx, sliceIdx, carrierIdx);
    printf("  result = %d\n", r);

    // 手动验证预期值
    {
        int exp = 0;
        exp += (g_phyCfg.phyMode == 0x01) ? 100 : 200;
        if (g_cellCfg[cellIdx].cellType == 0xFD) {
            exp += 10;
            if (g_cellCfg[cellIdx].rf.freqBand == 78) exp += 1;
        } else if (g_cellCfg[cellIdx].cellType == 0xEC) {
            exp += 20;
        }
        if (g_cellPhy[cellIdx].carrierFreq == 3500) exp += 5;
        if (g_trpCfg[trpIdx].trpType == 1) {
            exp += 50;
            if (g_trpCfg[trpIdx].beamCount == 64) exp += 2;
        }
        switch (g_sliceCfg[sliceIdx].sliceType) {
        case 1: exp += 1000;
            if (g_sliceCfg[sliceIdx].latencyTarget > 10.0f) exp += 1;
            break;
        case 2: exp += 2000; break;
        case 3: exp += 3000; break;
        default: exp -= 1;
        }
        if (g_carrierCfg[carrierIdx].bandwidth == 100) exp += 500;
        printf("  expected = %d  [%s]\n\n", exp, r == exp ? "MATCH" : "MISMATCH");
    }

    // --- 场景 B: 循环 + switch ---
    printf("--- 场景B: process_all_trps(cell=%u) ---\n", cellIdx);
    uint32_t trpSum = process_all_trps(cellIdx);
    printf("  total = %u\n", trpSum);

    {
        uint32_t exp = 0;
        uint32_t ct = g_cellCfg[cellIdx].cellType;
        for (uint8_t t = 0; t < M_TRP; t++) {
            if (ct == 0xFD)       exp += g_trpCfg[t].activeBeams * 2;
            else if (ct == 0xEC)  exp += g_trpCfg[t].activeBeams * 3;
            else                  exp += g_trpCfg[t].activeBeams;
        }
        if (exp > 0) exp += g_cellCfg[cellIdx].pci;
        printf("  expected = %u  [%s]\n\n", exp, trpSum == exp ? "MATCH" : "MISMATCH");
    }

    // --- 场景 C: slice loop ---
    printf("--- 场景C: process_slice_loop(slice=%u) ---\n", sliceIdx);
    uint32_t slSum = process_slice_loop(sliceIdx);
    printf("  sum = %u\n", slSum);

    {
        uint32_t exp = 0;
        switch (g_sliceCfg[sliceIdx].sliceType) {
        case 1:
            for (int i = 0; i < 4; i++) exp += g_carrierCfg[i].bandwidth;
            break;
        case 2:
            exp += g_carrierCfg[0].bandwidth;
            break;
        case 3:
            for (int i = 0; i < P_CARRIER; i++) exp += g_carrierCfg[i].bandwidth * 2;
            break;
        }
        printf("  expected = %u  [%s]\n\n", exp, slSum == exp ? "MATCH" : "MISMATCH");
    }

    // --- 场景 D: 生命周期 multi-return ---
    printf("--- 场景D: reconfig_cell_trp(cell=%u, trp=%u) ---\n", cellIdx, trpIdx);
    printf("  before: cellType=0x%x trpType=%u activeBeams=%u\n",
           g_cellCfg[cellIdx].cellType, g_trpCfg[trpIdx].trpType,
           g_trpCfg[trpIdx].activeBeams);

    int rc = reconfig_cell_trp(cellIdx, trpIdx, 0xAA, 5);
    printf("  rc = %d\n", rc);
    printf("  after:  cellType=0x%x trpType=%u activeBeams=%u\n",
           g_cellCfg[cellIdx].cellType, g_trpCfg[trpIdx].trpType,
           g_trpCfg[trpIdx].activeBeams);

    // 测试 early return 路径
    printf("  testing early return (invalid newCellType)...\n");
    rc = reconfig_cell_trp(cellIdx, trpIdx, 0xFFFF, 5);
    printf("  rc = %d (expected -1)\n\n", rc);

    // --- 场景 E: 生命周期 仅修改非 may_const ---
    printf("--- 场景E: update_trp_power(trp=%u, power=32) ---\n", trpIdx);
    printf("  before: activeBeams=%u\n", g_trpCfg[trpIdx].activeBeams);
    update_trp_power(trpIdx, 32);
    printf("  after:  activeBeams=%u\n\n", g_trpCfg[trpIdx].activeBeams);

    // JIT Statistics
    {
        ejit_stats_t s;
        ejit_get_stats(&s);
        printf("--- JIT Stats ---\n");
        printf("  entries: %zu  codeSize: %zu\n", s.entryCount, s.totalCodeSize);
        printf("  hits: %llu  misses: %llu  evictions: %llu\n",
               s.hits, s.misses, s.evictions);
        printf("  JIT: %s\n", s.entryCount > 0 ? "ACTIVE" : "AOT-FALLBACK-ONLY");
    }

    ejit_shutdown();
    printf("=== All Tests Complete ===\n");
    return 0;
}
