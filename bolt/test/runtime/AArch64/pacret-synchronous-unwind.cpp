// Test to demonstrate that functions compiled with synchronous unwind tables
// are ignored by the PointerAuthCFIAnalyzer.
// Exception handling is needed to have _any_ unwind tables, otherwise the
// PointerAuthCFIAnalyzer does not run on these functions, so it does not ignore
// any function.
//
// REQUIRES: system-linux,bolt-runtime
//
// RUN: %clangxx --target=aarch64-unknown-linux-gnu \
// RUN: -mbranch-protection=pac-ret \
// RUN: -fno-asynchronous-unwind-tables \
// RUN: %s -o %t.exe -Wl,-q
// RUN: llvm-bolt %t.exe -o %t.bolt | FileCheck %s --check-prefix=CHECK
//
// CHECK: PointerAuthCFIAnalyzer ran on 3 functions. Ignored
// CHECK-NOT: 0 functions (0.00%) because of CFI inconsistencies
// CHECK-SAME: 1 functions (33.33%) because of CFI inconsistencies
// CHECK-NEXT: BOLT-WARNING: PointerAuthCFIAnalyzer only supports asynchronous
// CHECK-SAME: unwind tables. For C compilers, see -fasynchronous-unwind-tables.

#include <cstdio>
#include <stdexcept>

void foo() { throw std::runtime_error("Exception from foo()."); }

int main() {
  try {
    foo();
  } catch (const std::exception &e) {
    printf("Exception caught: %s\n", e.what());
  }
  return 0;
}
