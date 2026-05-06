#include "llvm/ExecutionEngine/EJIT/EJitRuntime.h"
#include <stdio.h>
#include <string.h>

int main(void) {
  printf("=== EmbeddedJIT Minimal Demo ===\n\n");

  // 1. Initialize EJIT with config
  ejit_config_t cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.compileMode = EJIT_COMPILE_SYNC;
  cfg.optLevel = EJIT_OPT_L3;
  cfg.maxCodeMemory = 512 * 1024;   // 512KB
  cfg.maxDataMemory = 256 * 1024;   // 256KB
  cfg.maxCacheEntries = 64;
  cfg.maxCacheSize = 1024 * 1024;   // 1MB
  cfg.enableLogger = false;

  printf("[1] Initializing EJIT runtime...\n");
  ejit_status_t rc = ejit_init(&cfg);
  printf("    ejit_init => %d (%s)\n", rc, rc == EJIT_OK ? "OK" : "FAIL");

  // 2. Check compile mode
  ejit_compile_mode_t mode = ejit_get_compile_mode();
  printf("\n[2] Compile mode: %s\n",
         mode == EJIT_COMPILE_SYNC ? "SYNC" : "ASYNC");

  // 3. Activate a period
  printf("\n[3] Activating period 'sensor'...\n");
  rc = ejit_activate("sensor", 0);
  printf("    ejit_activate(sensor, 0) => %d\n", rc);
  printf("    ejit_is_active(sensor, 0) => %s\n",
         ejit_is_active("sensor", 0) ? "true" : "false");

  // 4. Activate all cells for a period
  printf("\n[4] Activating all cells for 'control'...\n");
  rc = ejit_activate_all("control");
  printf("    ejit_activate_all(control) => %d\n", rc);

  // 5. Get stats
  printf("\n[5] Runtime statistics:\n");
  ejit_stats_t stats;
  memset(&stats, 0, sizeof(stats));
  rc = ejit_get_stats(&stats);
  if (rc == EJIT_OK) {
    printf("    entries: %zu\n", stats.entryCount);
    printf("    codeSize: %zu bytes\n", stats.totalCodeSize);
    printf("    maxSize: %zu bytes\n", stats.maxSize);
    printf("    hits: %lu  misses: %lu  evictions: %lu\n",
           stats.hits, stats.misses, stats.evictions);
  }

  // 6. Deactivate and invalidate
  printf("\n[6] Deactivating and invalidating...\n");
  rc = ejit_deactivate("sensor", 0);
  printf("    ejit_deactivate(sensor, 0) => %d\n", rc);
  ejit_invalidate("sensor", 0);
  printf("    ejit_invalidate(sensor, 0) done\n");

  // 7. Shutdown
  printf("\n[7] Shutting down EJIT runtime...\n");
  ejit_shutdown();
  printf("    Done.\n");

  printf("\n=== Demo complete ===\n");
  return 0;
}
