// RUN: %clangxx_tsan -O1 %s -o %t
// `handle_sigbus=0` is required because when the rdar://problem/58789439 bug was
// present TSan's runtime could derefence bad memory leading to SIGBUS being raised.
// If the signal was caught TSan would deadlock because it would try to run the
// symbolizer again.
// RUN: %env_tsan_opts=handle_sigbus=0,symbolize=1 %run %t 2>&1 | FileCheck %s
// RUN: %env_tsan_opts=handle_sigbus=0,symbolize=1 __check_mach_ports_lookup=some_value %run %t 2>&1 | FileCheck %s
#include <sanitizer/common_interface_defs.h>
#include <stdio.h>
#include <stdlib.h>

const char *kEnvName = "__UNLIKELY_ENV_VAR_NAME__";

int main() {
  if (getenv(kEnvName)) {
    fprintf(stderr, "Env var %s should not be set\n", kEnvName);
    abort();
  }

  // This will set an environment variable that isn't already in
  // the environment array. This will cause Darwin's Libc to
  // malloc() a new array.
  if (setenv(kEnvName, "some_value", /*overwrite=*/1)) {
    fprintf(stderr, "Failed to set %s \n", kEnvName);
    abort();
  }

  // rdar://problem/58789439
  // Now trigger symbolization. If symbolization tries to call
  // to `setenv` that adds a new environment variable, then Darwin
  // Libc will call `realloc()` and TSan's runtime will hit
  // an assertion failure because TSan's runtime uses a different
  // allocator during symbolization which leads to `realloc()` being
  // called on a pointer that the allocator didn't allocate.
  //
  // CHECK: #{{[0-9]}} main {{.*}}no_call_setenv_in_symbolize.cpp:[[@LINE+1]]
  __sanitizer_print_stack_trace();

  // CHECK: DONE
  fprintf(stderr, "DONE\n");

  return 0;
}
