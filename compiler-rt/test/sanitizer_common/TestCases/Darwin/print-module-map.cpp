// Checks that module map does not print at 0, prints once after aborting with 1,
// and prints once before and after aborting with 2

// RUN: %clangxx -DUSING_%tool_name %s -o %t -w

// RUN: %env_tool_opts="print_module_map=0" not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-MM0
// RUN: %env_tool_opts="print_module_map=1" not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-MM1
// RUN: %env_tool_opts="print_module_map=2" not %run %t 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-MM2

// tsan support pending rdar://67747473
// XFAIL: tsan

// FIXME: Add linux support.
// XFAIL: msan && linux

// FIXME: Add lsan support.
// XFAIL: lsan

int global;

int main() {
#if defined(USING_ubsan)
  int value = 5;
  int computation = value / 0; // Division by zero.
#else
  volatile int *a = new int[100];
  delete[] a;
  global = a[0]; // use-after-free: triggers ASan/TSan report.
#endif
  return 0;
}

// CHECK: SUMMARY:
// CHECK-MM0-NOT: Process module map:
// CHECK-MM1-NOT: Process module map:
// CHECK-MM2: Process module map:
// CHECK: ABORTING
// CHECK-MM0-NOT: Process module map:
// CHECK-MM1-NEXT: Process module map:
// CHECK-MM2-NEXT: Process module map:
