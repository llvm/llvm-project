/**
 * EJIT 静态注册表路径测试 (forceStaticRegistry)
 *
 * 验证 PASS1/PASS2 生成的 __ejit_registry_bitcode[] 和
 * __ejit_registry_period[] 能被 ejit_init() 正确消费。
 *
 * 构造器路径使用 llvm.global_ctors，裸核不可用。
 * 静态注册表路径将数据放在全局常量数组中，ejit_init() 直接遍历。
 * 通过 forceStaticRegistry=true 强制走静态路径（测试/裸核用）。
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

//===-- 结构体 --------------------------------------------------------------===//

struct CellCfg {
  __attribute__((ejit_may_const)) uint32_t cellType;
  uint32_t trafficLoad;
};

//===-- 全局变量 ------------------------------------------------------------===//

__attribute__((ejit_period_arr("cell")))
struct CellCfg g_cellCfg[4] = {
  {1, 100}, {2, 200}, {3, 300}, {4, 400}
};

//===-- EJIT entry ---------------------------------------------------------===//

__attribute__((ejit_entry))
uint32_t jit_check_cell_type(__attribute__((ejit_period_arr_ind("cell"))) uint8_t cellIndex) {
  if (g_cellCfg[cellIndex].cellType == 1)
    return 10;
  else if (g_cellCfg[cellIndex].cellType == 2)
    return 20;
  else if (g_cellCfg[cellIndex].cellType == 3)
    return 30;
  return 40;
}

//===-- 测试主函数 ----------------------------------------------------------===//

typedef enum {
  EJIT_OK = 0,
  EJIT_ERR_NOT_ACTIVE = -2,
} ejit_status_t;

typedef enum { EJIT_COMPILE_SYNC = 0 } ejit_compile_mode_t;
typedef enum { EJIT_OPT_L1 = 1, EJIT_OPT_L2 = 2 } ejit_opt_level_t;

typedef struct {
  ejit_compile_mode_t compileMode;
  ejit_opt_level_t optLevel;
  size_t maxCodeMemory;
  size_t maxDataMemory;
  size_t maxCacheEntries;
  size_t maxCacheSize;
  int enableLogger;
  int forceStaticRegistry;
  const char *dumpJITDir;
} ejit_config_t;

typedef struct {
  const char *periodName;
  uint32_t index;
} ejit_dim_t;

extern ejit_status_t ejit_init(const ejit_config_t *config);
extern void ejit_shutdown(void);
extern ejit_status_t ejit_activate(const char *periodName, uint8_t cellIdx);
extern ejit_status_t ejit_deactivate(const char *periodName, uint8_t cellIdx);
extern void *ejit_compile_or_get(const char *funcName,
                                  const ejit_dim_t *dims, uint32_t count,
                                  void **out_pfn);

typedef uint32_t (*jit_fn_t)(uint8_t);

int main(void) {
  int failures = 0;

  ejit_config_t cfg = {0};
  cfg.optLevel = EJIT_OPT_L1;
  cfg.forceStaticRegistry = 1;  // force static table path

  if (ejit_init(&cfg) != EJIT_OK) {
    printf("FAIL: ejit_init\n");
    return 1;
  }
  printf("OK: ejit_init (forceStaticRegistry)\n");

  // Activate cell 0: cellType=1 → expect return 10
  if (ejit_activate("cell", 0) != EJIT_OK) {
    printf("FAIL: ejit_activate cell[0]\n");
    failures++;
  } else {
    printf("OK: ejit_activate cell[0]\n");
  }

  // Compile via JIT — bitcode comes from static registry table
  ejit_dim_t dims[] = {{.periodName = "cell", .index = 0}};
  void *pfn = NULL;
  void *result = ejit_compile_or_get("jit_check_cell_type", dims, 1, &pfn);
  if (!result || !pfn) {
    printf("FAIL: JIT compile (static registry path)\n");
    failures++;
  } else {
    jit_fn_t fn = (jit_fn_t)pfn;
    uint32_t val = fn(0);
    if (val == 10) {
      printf("OK: JIT compile + execute: cell[0]=%u (expected 10)\n", val);
    } else {
      printf("FAIL: JIT result %u, expected 10\n", val);
      failures++;
    }
  }

  // Test cell 2: cellType=3 → expect 30
  if (ejit_activate("cell", 2) != EJIT_OK) {
    printf("FAIL: ejit_activate cell[2]\n");
    failures++;
  } else {
    ejit_dim_t dims2[] = {{.periodName = "cell", .index = 2}};
    void *pfn2 = NULL;
    void *result2 = ejit_compile_or_get("jit_check_cell_type", dims2, 1, &pfn2);
    if (!result2 || !pfn2) {
      printf("FAIL: JIT compile cell[2]\n");
      failures++;
    } else {
      jit_fn_t fn2 = (jit_fn_t)pfn2;
      uint32_t val2 = fn2(2);
      if (val2 == 30) {
        printf("OK: JIT compile + execute: cell[2]=%u (expected 30)\n", val2);
      } else {
        printf("FAIL: JIT result cell[2]=%u, expected 30\n", val2);
        failures++;
      }
    }
  }

  ejit_shutdown();
  printf("OK: ejit_shutdown\n");

  printf("\n=== %s: %d failures ===\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
