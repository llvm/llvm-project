// Check that -fsanitize-coverage=trace-args,trace-ret calls the two callbacks
// with the observed argument and return values, and that a user definition
// overrides compiler-rt's weak defaults.
//
// The arguments are the IR arguments, reported on entry, so the values are
// right at any optimization level, with or without debug information.
//
// REQUIRES: has_sancovcc
// UNSUPPORTED: i386-darwin
// RUN: %clangxx -O2 -g -fsanitize-coverage=trace-pc-guard,trace-args,trace-ret %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s
// RUN: %clangxx -O0 -fsanitize-coverage=trace-pc-guard,trace-args,trace-ret %s -o %t.O0
// RUN: %run %t.O0 2>&1 | FileCheck %s

#include <cstdint>
#include <cstdio>

extern "C" {
// Stubs so the program links without a sanitizer runtime; these are weak in
// compiler-rt, so these definitions win.
void __sanitizer_cov_trace_pc_guard(uint32_t *) {}
void __sanitizer_cov_trace_pc_guard_init(uint32_t *, uint32_t *) {}

// The consumers under test. A size of zero means the instrumentation had
// nothing to report, and a non-zero num_fields would mean `val` is an address
// rather than a value. Neither callback is passed the function it speaks for;
// a consumer that wants it takes its own return address.
void __sanitizer_cov_trace_args(uint32_t arg_idx, uint32_t size, uint64_t val,
                                uint64_t *offsets, uint32_t num_fields) {
  if (size == sizeof(int) && !num_fields)
    fprintf(stderr, "ARG idx=%u val=%d\n", arg_idx, (int)val);
}

void __sanitizer_cov_trace_ret(uint32_t size, uint64_t val, uint64_t *offsets,
                               uint32_t num_fields) {
  if (size == sizeof(int) && !num_fields)
    fprintf(stderr, "RET val=%d\n", (int)val);
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
