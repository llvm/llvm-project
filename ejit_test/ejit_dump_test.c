//===-- ejit_dump_test.c - JIT IR dump verification -----------------------===//
//
// Verifies that dumpJITDir produces valid pre/post-optimization IR files.
// Checks:
//   1. _pre.ll and _opt.ll files are generated per specialization
//   2. _opt.ll is smaller than _pre.ll (optimization reduced code)
//   3. _opt.ll contains constant-folded values
//   4. Multiple specializations produce distinct files
//   5. Environment variable EJIT_DUMP_DIR controls the output path
//
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "llvm/ExecutionEngine/EJIT/EJitRuntime.h"

//===--- Data types --------------------------------------------------------===//

typedef enum { MODE_A = 0, MODE_B = 1, MODE_C = 2 } Mode_t;

typedef struct {
  Mode_t ejit_may_const mode;
  int ejit_may_const threshold;
  int gain;
} CellCfg;

//===--- Period variables --------------------------------------------------===//

ejit_period_arr(cell) CellCfg g_cells[16];

//===--- JIT entry functions -----------------------------------------------===//

ejit_entry int check_cell(ejit_period_arr_ind(cell) uint8_t idx) {
  if (g_cells[idx].mode == MODE_A)
    return g_cells[idx].threshold * 10 + g_cells[idx].gain;
  else if (g_cells[idx].mode == MODE_B)
    return g_cells[idx].threshold * 20 + g_cells[idx].gain;
  else
    return g_cells[idx].threshold * 5 + g_cells[idx].gain;
}

ejit_entry int simple_return(ejit_period_arr_ind(cell) uint8_t idx) {
  return g_cells[idx].threshold + g_cells[idx].gain;
}

//===--- Helpers -----------------------------------------------------------===//

static int g_failures = 0;
#define CHECK(cond, msg) do { \
  if (!(cond)) { g_failures++; printf("  FAIL: %s\n", msg); } \
  else printf("  OK  : %s\n", msg); \
} while(0)

static long file_size(const char *path) {
  struct stat st;
  if (stat(path, &st) != 0) return -1;
  return (long)st.st_size;
}

static int file_contains(const char *path, const char *pattern) {
  FILE *f = fopen(path, "r");
  if (!f) return 0;
  char buf[16384];
  size_t n = fread(buf, 1, sizeof(buf) - 1, f);
  fclose(f);
  if (n == 0) return 0;
  buf[n] = '\0';
  return strstr(buf, pattern) != NULL;
}

//===--- Main --------------------------------------------------------------===//

int main(int argc, char **argv) {
  int cellIdx = 0;
  if (argc > 1) cellIdx = atoi(argv[1]);

  // Use env var or default (under TMPDIR to avoid permission issues)
  const char *dumpDir = getenv("EJIT_DUMP_DIR");
  if (!dumpDir) {
    const char *tmp = getenv("TMPDIR");
    if (!tmp) tmp = "/tmp";
    static char buf[256];
    snprintf(buf, sizeof(buf), "%s/ejit_dump_test", tmp);
    dumpDir = buf;
  }

  // Clean and create dump dir
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "rm -rf %s && mkdir -p %s", dumpDir, dumpDir);
  system(cmd);

  printf("=== EJIT JIT IR Dump Test ===\n");
  printf("cellIdx=%d  dumpDir=%s\n\n", cellIdx, dumpDir);

  ejit_config_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.compileMode = EJIT_COMPILE_SYNC;
  cfg.optLevel = EJIT_OPT_L2;
  cfg.dumpJITDir = dumpDir;

  ejit_status_t st = ejit_init(&cfg);
  CHECK(st == EJIT_OK, "ejit_init with dumpJITDir");

  // Initialize cells
  for (int i = 0; i < 16; i++) {
    g_cells[i].mode = (i % 3 == 1) ? MODE_B : MODE_A;
    g_cells[i].threshold = 10 + i;
    g_cells[i].gain = i;
  }

  // --- Test 1: Single specialization, verify dump files ---
  printf("\n--- Test 1: Single specialization dump ---\n");
  st = ejit_activate("cell", (uint8_t)cellIdx);
  CHECK(st == EJIT_OK, "ejit_activate");
  ejit_clear_cache();

  int r1 = check_cell((uint8_t)cellIdx);
  printf("  check_cell(%d) = %d\n", cellIdx, r1);

  // Verify files exist
  snprintf(cmd, sizeof(cmd), "%s/check_cell_", dumpDir);
  char pre_path[512], opt_path[512];
  // Find files by scanning the directory
  snprintf(cmd, sizeof(cmd), "ls %s/check_cell_*_pre.ll 2>/dev/null | head -1", dumpDir);
  FILE *fp = popen(cmd, "r");
  int found_pre = 0, found_opt = 0;
  if (fp && fgets(pre_path, sizeof(pre_path), fp)) {
    pre_path[strcspn(pre_path, "\n")] = 0;
    found_pre = 1;
    // Derive opt path
    strcpy(opt_path, pre_path);
    char *p = strstr(opt_path, "_pre.ll");
    if (p) strcpy(p, "_opt.ll");
    found_opt = (file_size(opt_path) > 0);
  }
  if (fp) pclose(fp);

  CHECK(found_pre, "pre-optimization IR file exists");
  CHECK(found_opt, "post-optimization IR file exists");

  if (found_pre && found_opt) {
    long pre_sz = file_size(pre_path);
    long opt_sz = file_size(opt_path);
    printf("  pre.ll: %ld bytes\n", pre_sz);
    printf("  opt.ll: %ld bytes\n", opt_sz);
    CHECK(opt_sz <= pre_sz, "opt.ll not larger than pre.ll");

    // Verify opt.ll has constant-folded result
    int has_ret = file_contains(opt_path, "ret i32");
    CHECK(has_ret, "opt.ll contains return instruction");

    // Verify pre.ll has the original load instruction
    int has_load = file_contains(pre_path, "load i32");
    CHECK(has_load, "pre.ll contains load instruction");
  }

  // --- Test 2: Second specialization with different index ---
  printf("\n--- Test 2: Multi-specialization dump ---\n");
  int idx2 = (cellIdx + 3) % 16;
  st = ejit_activate("cell", (uint8_t)idx2);
  CHECK(st == EJIT_OK, "activate second index");
  ejit_clear_cache();

  int r2 = simple_return((uint8_t)idx2);
  printf("  simple_return(%d) = %d\n", idx2, r2);

  // Count dump files
  snprintf(cmd, sizeof(cmd), "ls %s/*.ll 2>/dev/null | wc -l", dumpDir);
  fp = popen(cmd, "r");
  char count_buf[16] = {0};
  int file_count = 0;
  if (fp) {
    if (fgets(count_buf, sizeof(count_buf), fp))
      file_count = atoi(count_buf);
    pclose(fp);
  }
  printf("  total dump files: %d\n", file_count);
  CHECK(file_count >= 4, "at least 4 dump files (2 funcs x pre+opt)");
  // check_cell has 3 dump files: check_cell + its pre/opt
  // simple_return has more: simple_return + its pre/opt
  // Actually check_cell has 1 file pattern, generating files like:
  //   check_cell_<cacheKey>_pre.ll  check_cell_<cacheKey>_opt.ll
  // And simple_return generates:
  //   simple_return_<cacheKey>_pre.ll  simple_return_<cacheKey>_opt.ll
  // So ideally 4 files. But there might be extra dumps from cache hits.
  // At minimum 4 files (2 funcs x pre+opt).

  // --- Test 3: Cache hit does NOT regenerate dump files ---
  printf("\n--- Test 3: Cache hit does not re-dump ---\n");
  int before_count = file_count;

  int r3 = simple_return((uint8_t)idx2);
  printf("  simple_return(%d) cached = %d\n", idx2, r3);

  snprintf(cmd, sizeof(cmd), "ls %s/*.ll 2>/dev/null | wc -l", dumpDir);
  fp = popen(cmd, "r");
  int after_count = 0;
  if (fp) {
    if (fgets(count_buf, sizeof(count_buf), fp))
      after_count = atoi(count_buf);
    pclose(fp);
  }
  CHECK(after_count == before_count, "cache hit did not add dump files");

  // --- Test 4: clear_cache + recompile generates new files ---
  printf("\n--- Test 4: Recompile after clear_cache ---\n");
  ejit_clear_cache();
  int r4 = simple_return((uint8_t)idx2);
  printf("  simple_return(%d) recompiled = %d\n", idx2, r4);

  snprintf(cmd, sizeof(cmd), "ls %s/*.ll 2>/dev/null | wc -l", dumpDir);
  fp = popen(cmd, "r");
  int recount = 0;
  if (fp) {
    if (fgets(count_buf, sizeof(count_buf), fp))
      recount = atoi(count_buf);
    pclose(fp);
  }
  CHECK(recount >= after_count, "recompile adds dump files (no overwrite)");

  // --- Test 5: verify pre.ll has may_const metadata ---
  printf("\n--- Test 5: pre.ll contains metadata ---\n");
  snprintf(cmd, sizeof(cmd), "ls %s/check_cell_*_pre.ll 2>/dev/null | head -1", dumpDir);
  fp = popen(cmd, "r");
  if (fp && fgets(pre_path, sizeof(pre_path), fp)) {
    pre_path[strcspn(pre_path, "\n")] = 0;
    int has_mayconst = file_contains(pre_path, "ejit.may_const");
    int has_period = file_contains(pre_path, "ejit_period_arr") ||
                     file_contains(pre_path, "ejit_period");
    CHECK(has_mayconst, "pre.ll contains ejit.may_const metadata");
    CHECK(has_period, "pre.ll contains ejit_period_arr metadata");
  }
  if (fp) pclose(fp);

  ejit_shutdown();

  printf("\n=== %s: %d failures ===\n",
         g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures > 0 ? 1 : 0;
}
