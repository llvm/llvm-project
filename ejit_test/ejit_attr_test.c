/**
 * EmbeddedJIT Attribute 综合测试用例
 *
 * 基于 SPEC4.md §2 (用户标注接口), §5 (应用场景案例)
 * 以及 clang/test/CodeGen/ejit_metadata.c 的正确属性写法
 *
 * === 使用 debug clang 编译 + 链接 release libs ===
 *
 *   # Sema 检查:
 *   build/bin/clang -fsyntax-only ejit_attr_test.c
 *
 *   # 生成 IR (观察 metadata):
 *   build/bin/clang -S -emit-llvm -O0 ejit_attr_test.c -o - | grep 'ejit'
 *
 *   # 编译为 .o (debug clang):
 *   build/bin/clang -c -O2 ejit_attr_test.c -o ejit_attr_test.o
 *
 *   # 链接 (release 静态库):
 *   clang++ -Os -Wl,--gc-sections -Wl,--strip-all \
 *     ejit_attr_test.o \
 *     ALL_LIBS=$(ls build_x86/lib/x.a); clang++ ... -Wl,--whole-archive $ALL_LIBS -Wl,--no-whole-archive
 *     -lpthread -ldl -o ejit_attr_test
 */

#include <stdint.h>
#include <stdbool.h>

//===-- 结构体定义 -----------------------------------------------------------===//

// 属性直接使用 __attribute__((...)) 写法，参数为字符串字面量

struct BoardConfig {
    __attribute__((ejit_may_const)) uint32_t boardType;
    uint32_t xx;
    struct {
        __attribute__((ejit_may_const)) uint32_t chipId;
        uint32_t reserved;
    } inner;
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

struct SensorConfig {
    __attribute__((ejit_may_const)) bool     enabled;
    __attribute__((ejit_may_const)) float    threshold;
    __attribute__((ejit_may_const)) uint64_t sensorId;
    uint32_t readings;
};

//===-- 全局变量声明 ----------------------------------------------------------===//

// static 时间窗: JIT 运行期不变 (SPEC4 §2.2.1)
__attribute__((ejit_period("static"))) struct BoardConfig g_boardCfg;

// 时间窗数组: cell (SPEC4 §2.2.2, §5 场景 2)
__attribute__((ejit_period_arr("cell"))) struct CellConfig g_cellCfg[16];

// 时间窗数组: trp (SPEC4 §2.2.2, §5 场景 3)
__attribute__((ejit_period_arr("trp"))) struct TrpConfig g_trpCfg[8];

// static 时间窗: sensor
__attribute__((ejit_period("static"))) struct SensorConfig g_sensorCfg;

//===-- 场景 1: 仅依赖 static 时间窗 (SPEC4 §5) -------------------------------===//

__attribute__((ejit_entry))
void process_board(void)
{
    // JIT 时 g_boardCfg.boardType 被视为常量 → 分支折叠
    if (g_boardCfg.boardType == 0x01) {
        g_boardCfg.xx = 100;
    } else {
        g_boardCfg.xx = 200;
    }
}

__attribute__((ejit_entry))
void process_sensor(void)
{
    if (g_sensorCfg.enabled) {
        if (g_sensorCfg.threshold > 0.5f) {
            g_sensorCfg.readings += 1;
        }
    }
}

//===-- 场景 2: 单时间窗依赖 (static + cell) (SPEC4 §5) -----------------------===//

__attribute__((ejit_entry))
void process_cell(
    __attribute__((ejit_period_arr_ind("cell"))) uint8_t cellIdx)
{
    // JIT 时 g_cellCfg[cellIdx].cellType 被视为常量
    if (g_boardCfg.boardType == 0x01) {
        if (g_cellCfg[cellIdx].cellType == 0xFD) {
            g_cellCfg[cellIdx].trafficLoad += 10;
        } else {
            g_cellCfg[cellIdx].trafficLoad += 20;
        }
    }
}

// 生命周期管理: 修改 cell 时间窗数据 (SPEC4 §2.3.3)
__attribute__((ejit_period_lc("cell")))
void update_cell_config(
    __attribute__((ejit_period_arr_ind("cell"))) uint8_t cellIdx)
{
    g_cellCfg[cellIdx].cellType = 0xFD;
    g_cellCfg[cellIdx].cellId   = 42;
}

//===-- 场景 3: 多时间窗依赖 (static + cell + trp) (SPEC4 §5) ------------------===//

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

// 同时管理多个时间窗生命周期 (CLANG_ATTR_DESIGN §4.4.6 Model 1: 允许多实例)
__attribute__((ejit_period_lc("cell")))
__attribute__((ejit_period_lc("trp")))
void update_all_config(
    __attribute__((ejit_period_arr_ind("cell"))) uint8_t cellIdx,
    __attribute__((ejit_period_arr_ind("trp")))  uint8_t trpIdx)
{
    g_cellCfg[cellIdx].cellType = 0xEC;
    g_trpCfg[trpIdx].trpType = 2;
}

// 分别管理 (SPEC4 §5 场景 3 方式二)
__attribute__((ejit_period_lc("cell")))
void change_cell_only(
    __attribute__((ejit_period_arr_ind("cell"))) uint8_t cellIdx)
{
    g_cellCfg[cellIdx].cellType = 0xEC;
}

__attribute__((ejit_period_lc("trp")))
void change_trp_only(
    __attribute__((ejit_period_arr_ind("trp"))) uint8_t trpIdx)
{
    g_trpCfg[trpIdx].trpType = 1;
}

//===-- 嵌套数组 + 嵌套结构体 may_const --------------------------------------===//

struct NestedCell {
    __attribute__((ejit_may_const)) uint32_t type;
    struct {
        __attribute__((ejit_may_const)) uint32_t subType;
        uint32_t data;
    } sub;
};

__attribute__((ejit_period_arr("nested"))) struct NestedCell g_nestedCells[32];

__attribute__((ejit_entry))
void process_nested(
    __attribute__((ejit_period_arr_ind("nested"))) uint8_t idx)
{
    // JIT 时 g_nestedCells[idx].type 和 .sub.subType 均被视为常量
    if (g_nestedCells[idx].type == 1) {
        if (g_nestedCells[idx].sub.subType == 100) {
            g_nestedCells[idx].sub.data += 1;
        }
    }
}

//===-- Main: 运行时演示 ----------------------------------------------------===//

// 声明运行时 C API (来自 EJitRuntime.h)
extern int ejit_init(const void *config);
extern int ejit_activate(const char *name, unsigned char idx);
extern int ejit_deactivate(const char *name, unsigned char idx);
extern void ejit_shutdown(void);

int main(void)
{
    ejit_init(0);

    // 激活时间窗实例
    ejit_activate("cell", 0);
    ejit_activate("trp", 0);
    ejit_activate("nested", 0);

    // 调用 ejit_entry 函数 (首次触发 JIT 编译)
    process_board();
    process_sensor();
    process_cell(0);
    process_trp_task(0, 0);
    process_nested(0);

    // 生命周期管理
    update_cell_config(1);
    change_cell_only(0);

    ejit_shutdown();
    return 0;
}
