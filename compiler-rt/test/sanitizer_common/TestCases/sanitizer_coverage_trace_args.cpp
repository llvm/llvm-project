// Test the dataflow-coverage callbacks (-fsanitize-coverage=trace-args,trace-ret):
// the instrumentation invokes __sanitizer_cov_trace_args / __sanitizer_cov_trace_ret
// with the observed argument and return values, and a user definition overrides
// compiler-rt's weak defaults.

// trace-args reads arguments at the function entry insertion point, so it needs
// -O1+ (at -O0 the trace call precedes the prologue stores to the argument slots).

// REQUIRES: has_sancovcc
// UNSUPPORTED: i386-darwin
// RUN: %clangxx -O2 -g -fsanitize-coverage=trace-pc-guard,trace-args,trace-ret %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

#include <cstdint>
#include <cstdio>

extern "C" {
// Standalone stubs so the program links without a sanitizer runtime (these are
// weak in compiler-rt; our strong definitions win).
void __sanitizer_cov_trace_pc_guard(uint32_t *) {}
void __sanitizer_cov_trace_pc_guard_init(uint32_t *, uint32_t *) {}

// The dataflow consumers under test. arg/ret may be null (skipped/optimized-away)
// -- do not dereference a null pointer.
void __sanitizer_cov_trace_args(uint64_t pc, uint32_t arg_idx, uint32_t arg_size,
                                void *arg, uint64_t *offsets, uint32_t nfields) {
  if (arg && arg_size == sizeof(int))
    fprintf(stderr, "ARG idx=%u val=%d\n", arg_idx, *(int *)arg);
}
void __sanitizer_cov_trace_ret(uint64_t pc, uint32_t ret_size, void *ret,
                               uint64_t *offsets, uint32_t nfields) {
  if (ret && ret_size == sizeof(int))
    fprintf(stderr, "RET val=%d\n", *(int *)ret);
}
}

__attribute__((noinline)) int add(int a, int b) { return a + b; }

int main() {
  volatile int r = add(41, 2);
  fprintf(stderr, "r=%d\n", (int)r);
  return 0;
}

// CHECK-DAG: ARG idx=0 val=41
// CHECK-DAG: ARG idx=1 val=2
// CHECK: RET val=43
// CHECK: r=43
