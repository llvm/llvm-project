// Test dedup_token_length
// Fails with debug checks: https://bugs.llvm.org/show_bug.cgi?id=46860
// XFAIL: !compiler-rt-optimized && tsan
// RUN: %clangxx -O0 %s -o %t
// RUN: env %tool_options='abort_on_error=0'                       not %run %t 2>&1   | FileCheck %s --check-prefix=CHECK0 --match-full-lines
// RUN: env %tool_options='abort_on_error=0, dedup_token_length=0' not %run %t 2>&1   | FileCheck %s --check-prefix=CHECK0 --match-full-lines
// RUN: env %tool_options='abort_on_error=0, dedup_token_length=1' not %run %t 2>&1   | FileCheck %s --check-prefix=CHECK1 --match-full-lines
// RUN: env %tool_options='abort_on_error=0, dedup_token_length=2' not %run %t 2>&1   | FileCheck %s --check-prefix=CHECK2 --match-full-lines
// RUN: env %tool_options='abort_on_error=0, dedup_token_length=3' not %run %t 2>&1   | FileCheck %s --check-prefix=CHECK3 --match-full-lines

// REQUIRES: stable-runtime

// rdar://158303080 top few frames are at times inaccurate in ubsan fast stack
// unwind on darwin
// XFAIL: (darwin && ubsan && (arm64-target-arch || arm64e-target-arch))

// XFAIL: target={{.*netbsd.*}} && !asan

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
