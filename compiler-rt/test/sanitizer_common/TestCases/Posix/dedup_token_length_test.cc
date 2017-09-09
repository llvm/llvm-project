// Test dedup_token_length
// RUN: %clangxx -O0 %s -o %t
// RUN: env %tool_options='abort_on_error=0'                       not %run %t 2>&1   | FileCheck %s --check-prefix=CHECK0 --match-full-lines
// RUN: env %tool_options='abort_on_error=0, dedup_token_length=0' not %run %t 2>&1   | FileCheck %s --check-prefix=CHECK0 --match-full-lines
// RUN: env %tool_options='abort_on_error=0, dedup_token_length=1' not %run %t 2>&1   | FileCheck %s --check-prefix=CHECK1 --match-full-lines
// RUN: env %tool_options='abort_on_error=0, dedup_token_length=2' not %run %t 2>&1   | FileCheck %s --check-prefix=CHECK2 --match-full-lines
// RUN: env %tool_options='abort_on_error=0, dedup_token_length=3' not %run %t 2>&1   | FileCheck %s --check-prefix=CHECK3 --match-full-lines

// REQUIRES: stable-runtime
// FIXME: implement SEGV handler in other sanitizers, not just asan.
// XFAIL: msan
// XFAIL: lsan
// XFAIL: tsan
// XFAIL: ubsan

volatile int *null = 0;

namespace Xyz {
  template<class A, class B> void Abc() {
    *null = 0;
  }
}

extern "C" void bar() {
  Xyz::Abc<int, int>();
}

void FOO() {
  bar();
}

int main(int argc, char **argv) {
  FOO();
}

// CHECK0-NOT: DEDUP_TOKEN:
// CHECK1: DEDUP_TOKEN: void Xyz::Abc<int, int>()
// CHECK2: DEDUP_TOKEN: void Xyz::Abc<int, int>()--bar
// CHECK3: DEDUP_TOKEN: void Xyz::Abc<int, int>()--bar--FOO()
