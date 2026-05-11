//===-- ejit_ptr_period_test.c - Pointer-type period/period_arr tests -----===//
//
// Test pointer-type ejit_period (static time-window) and
// pointer-type ejit_period_arr (period array) with ejit_may_const fields.
//
// Verifies that PASS6 correctly handles indirect access through pointer
// global variables — reading *(void**)&GV to resolve the actual data
// base address before computing field offsets.
//
//===----------------------------------------------------------------------===//

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llvm/ExecutionEngine/EJIT/EJitRuntime.h"

//===--- Data types --------------------------------------------------------===//

typedef enum { MODE_A = 0, MODE_B = 1, MODE_C = 2 } Mode_t;

typedef struct {
  Mode_t ejit_may_const mode;
  int ejit_may_const threshold;
  int gain;
} CellCfg;

typedef struct {
  int ejit_may_const version;
  int ejit_may_const max_cells;
  char name[32];
} BoardCfg;

//===--- Pointer-type ejit_period_arr (dynamic array) ---------------------===//

// Single pointer to struct array — period_arr on a pointer variable.
// The actual array is malloc'd at runtime. User guarantees the pointer
// is assigned before ejit_activate and that indices are within bounds.
ejit_period_arr(cell) CellCfg *g_cell_ptr;

//===--- Pointer-type ejit_period (static time-window) --------------------===//

// Pointer to a single struct — period (static time-window).
// Changed infrequently; JIT specializes based on its fields.
ejit_period(static) BoardCfg *g_board_ptr;

//===--- Static period array for comparison testing -----------------------===//

ejit_period_arr(cell) CellCfg g_cell_arr[16];

//===--- Pointer-array type period_arr ------------------------------------===//

// Array of pointers — each element points to a CellCfg.
ejit_period_arr(cell) CellCfg *g_cell_ptrs[8];

//===--- Helper ------------------------------------------------------------===//

static int g_failures = 0;
#define CHECK(cond, msg) do { \
  if (!(cond)) { g_failures++; printf("  FAIL: %s\n", msg); } \
  else printf("  OK  : %s\n", msg); \
} while(0)

//===--- Test 1: Pointer period_arr JIT specialization --------------------===//

ejit_entry int test_ptr_period_arr(ejit_period_arr_ind(cell) uint8_t idx) {
  if (g_cell_ptr[idx].mode == MODE_A)
    return g_cell_ptr[idx].threshold * 10 + g_cell_ptr[idx].gain;
  else if (g_cell_ptr[idx].mode == MODE_B)
    return g_cell_ptr[idx].threshold * 20 + g_cell_ptr[idx].gain;
  else
    return g_cell_ptr[idx].threshold * 5 + g_cell_ptr[idx].gain;
}

//===--- Test 2: Static pointer period JIT specialization -----------------===//

ejit_entry int test_ptr_period(void) {
  if (g_board_ptr->version >= 2)
    return g_board_ptr->max_cells * 100;
  else
    return g_board_ptr->max_cells * 10;
}

//===--- Test 3: Static array period_arr (existing behavior) ---------------===//

ejit_entry int test_arr_period_arr(ejit_period_arr_ind(cell) uint8_t idx) {
  if (g_cell_arr[idx].mode == MODE_A)
    return g_cell_arr[idx].threshold * 10 + g_cell_arr[idx].gain;
  else
    return g_cell_arr[idx].threshold * 5 + g_cell_arr[idx].gain;
}

//===--- Test 4: Pointer-array period_arr JIT specialization --------------===//

ejit_entry int test_ptr_arr_period_arr(ejit_period_arr_ind(cell) uint8_t idx) {
  if (g_cell_ptrs[idx]->mode == MODE_A)
    return g_cell_ptrs[idx]->threshold * 10 + g_cell_ptrs[idx]->gain;
  else
    return g_cell_ptrs[idx]->threshold * 5 + g_cell_ptrs[idx]->gain;
}

//===--- Test 5: Multiple calls with different cell indices ----------------===//

ejit_entry int test_multi_index(ejit_period_arr_ind(cell) uint8_t idx) {
  return g_cell_ptr[idx].threshold * 2 + g_cell_ptr[idx].gain;
}

//===--- Test 6: Nested struct through pointer ----------------------------===//

typedef struct {
  int ejit_may_const id;
  CellCfg ejit_may_const cfg;
} CompoundCfg;

ejit_period_arr(cell) CompoundCfg *g_compound_ptr;

ejit_entry int test_nested_ptr(ejit_period_arr_ind(cell) uint8_t idx) {
  if (g_compound_ptr[idx].cfg.mode == MODE_A)
    return g_compound_ptr[idx].cfg.threshold + g_compound_ptr[idx].id;
  else
    return g_compound_ptr[idx].cfg.threshold;
}

//===--- Lifecycle test ----------------------------------------------------===//

ejit_period_lc(cell) void test_lifecycle(ejit_period_arr_ind(cell) uint8_t idx) {
  g_cell_ptr[idx].gain = 42;
}

//===--- Main --------------------------------------------------------------===//

int main(int argc, char **argv) {
  int cellIdx = 0;
  if (argc > 1) cellIdx = atoi(argv[1]);

  printf("=== EJIT Pointer Period Test ===\n");
  printf("cellIdx=%d\n\n", cellIdx);

  // Initialize EJIT
  ejit_config_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.compileMode = EJIT_COMPILE_SYNC;

  ejit_status_t st = ejit_init(&cfg);
  CHECK(st == EJIT_OK, "ejit_init");

  // Allocate and initialize dynamic arrays for pointer-type period_arr
  int num_cells = 16;
  g_cell_ptr = calloc(num_cells, sizeof(CellCfg));
  CHECK(g_cell_ptr != NULL, "malloc g_cell_ptr");

  for (int i = 0; i < num_cells; i++) {
    g_cell_ptr[i].mode = (i % 3 == 1) ? MODE_B : MODE_A;
    g_cell_ptr[i].threshold = 10 + i;
    g_cell_ptr[i].gain = i;
  }

  // Initialize static pointer period
  g_board_ptr = calloc(1, sizeof(BoardCfg));
  CHECK(g_board_ptr != NULL, "malloc g_board_ptr");
  g_board_ptr->version = 3;
  g_board_ptr->max_cells = 64;

  // Initialize static array
  for (int i = 0; i < 16; i++) {
    g_cell_arr[i].mode = MODE_A;
    g_cell_arr[i].threshold = 20 + i;
    g_cell_arr[i].gain = i * 2;
  }

  // Initialize pointer array
  for (int i = 0; i < 8; i++) {
    g_cell_ptrs[i] = calloc(1, sizeof(CellCfg));
    g_cell_ptrs[i]->mode = (i % 2) ? MODE_C : MODE_A;
    g_cell_ptrs[i]->threshold = 5 + i;
    g_cell_ptrs[i]->gain = i * 3;
  }

  // Initialize compound pointer
  g_compound_ptr = calloc(num_cells, sizeof(CompoundCfg));
  CHECK(g_compound_ptr != NULL, "malloc g_compound_ptr");
  for (int i = 0; i < num_cells; i++) {
    g_compound_ptr[i].id = 100 + i;
    g_compound_ptr[i].cfg.mode = MODE_A;
    g_compound_ptr[i].cfg.threshold = 30 + i;
    g_compound_ptr[i].cfg.gain = i;
  }

  // Activate cells
  st = ejit_activate("cell", (uint8_t)cellIdx);
  CHECK(st == EJIT_OK, "ejit_activate cell");

  // --- Test 1: Pointer period_arr ---
  printf("\n--- Test 1: Pointer period_arr ---\n");
  ejit_clear_cache();
  int r1 = test_ptr_period_arr((uint8_t)cellIdx);
  printf("  result=%d\n", r1);
  // cellIdx=0: MODE_A(10*10+0=100); cellIdx=1: MODE_B(11*20+1=221)
  if (cellIdx == 0) CHECK(r1 == 100, "ptr_period_arr cell 0");
  if (cellIdx == 1) CHECK(r1 == 221, "ptr_period_arr cell 1");

  // --- Test 2: Static pointer period ---
  printf("\n--- Test 2: Static pointer period ---\n");
  ejit_clear_cache();
  int r2 = test_ptr_period();
  printf("  result=%d\n", r2);
  // version=3 >= 2: max_cells(64) * 100 = 6400
  CHECK(r2 == 6400, "ptr_period static");

  // --- Test 3: Static array period_arr (existing functionality) ---
  printf("\n--- Test 3: Static array period_arr ---\n");
  ejit_clear_cache();
  int r3 = test_arr_period_arr((uint8_t)cellIdx);
  printf("  result=%d (static array)\n", r3);
  if (cellIdx == 0) CHECK(r3 == 200, "arr_period_arr cell 0");  // 20*10+0=200

  // --- Test 4: Pointer-array period_arr ---
  printf("\n--- Test 4: Pointer-array period_arr ---\n");
  ejit_clear_cache();
  int r4 = test_ptr_arr_period_arr((uint8_t)cellIdx);
  printf("  result=%d (pointer array)\n", r4);
  if (cellIdx == 0) CHECK(r4 == 50, "ptr_arr_period_arr cell 0");  // 5*10+0=50

  // --- Test 5: Multiple indices ---
  printf("\n--- Test 5: Multiple cell indices ---\n");
  if (cellIdx == 0) {
    ejit_clear_cache();
    int r5a = test_multi_index(0);
    // Reactivate for different index
    ejit_activate("cell", 5);
    ejit_clear_cache();
    int r5b = test_multi_index(5);
    printf("  idx=0: %d, idx=5: %d\n", r5a, r5b);
    CHECK(r5a == 20, "multi_index cell 0");   // 10*2+0=20
    CHECK(r5b == 35, "multi_index cell 5");   // 15*2+5=35... wait let me recalculate
    // cell 5: threshold=15, gain=5 → 15*2+5=35
    // Hmm actually cell 5 was not explicitly initialized. Let me check.
    // The g_cell_ptr[5] was initialized in the loop: mode=MODE_A, threshold=10+5=15, gain=5
    // So: 15*2+5=35
    // Actually wait, the JIT compiles for cell 0 first, then we activate cell 5 and clear cache.
    // The JIT should recompile for cell 5 with threshold=15, gain=5.
    // Result = 15*2+5 = 35
  }

  // --- Test 6: Nested struct through pointer ---
  printf("\n--- Test 6: Nested struct through pointer ---\n");
  ejit_clear_cache();
  int r6 = test_nested_ptr((uint8_t)cellIdx);
  printf("  result=%d\n", r6);
  // cellIdx=0: MODE_A → cfg.threshold(30) + id(100) = 130
  if (cellIdx == 0) CHECK(r6 == 130, "nested_ptr cell 0");

  // --- Test 7: Lifecycle ---
  printf("\n--- Test 7: Lifecycle (ejit_period_lc with pointer period_arr) ---\n");
  test_lifecycle((uint8_t)cellIdx);
  CHECK(g_cell_ptr[cellIdx].gain == 42, "lifecycle gain updated");

  // --- Test 8: JIT recompilation after clear_cache ---
  // Test 7 (lifecycle) modified g_cell_ptr[].gain to 42. Restore for clean test.
  g_cell_ptr[cellIdx].gain = cellIdx;
  printf("\n--- Test 8: JIT recompilation after clear_cache ---\n");
  ejit_clear_cache();
  int r8a = test_ptr_period_arr((uint8_t)cellIdx);
  printf("  result after recache=%d\n", r8a);
  // cell 0: MODE_A → 10*10+0=100; cell 1: MODE_B → 11*20+1=221; cell 3: MODE_A → 13*10+3=133
  int expect8 = (cellIdx == 1) ? (11*20 + cellIdx) : ((10 + cellIdx)*10 + cellIdx);
  CHECK(r8a == expect8, "recompile ptr_period_arr");

  // --- Test 9: ejit_period on pointer (static, always active) ---
  // ejit_period(static) is implicitly active; no explicit ejit_activate needed.
  printf("\n--- Test 9: Static pointer period (always active) ---\n");
  ejit_clear_cache();
  int r9 = test_ptr_period();
  printf("  result=%d\n", r9);
  CHECK(r9 == 6400, "ptr_period static implicit");

  // Shutdown
  ejit_shutdown();
  printf("  OK: ejit_shutdown\n");

  // Cleanup
  free(g_cell_ptr);
  free(g_board_ptr);
  for (int i = 0; i < 8; i++) free(g_cell_ptrs[i]);
  free(g_compound_ptr);

  printf("\n=== %s: %d failures ===\n",
         g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures > 0 ? 1 : 0;
}
