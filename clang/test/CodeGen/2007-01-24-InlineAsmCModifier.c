// Verify that the %c modifier works and strips off any prefixes from
// immediates.
// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

void foo(void) {
  // CHECK: i32 789514
  __asm__         volatile("/* " "pickANumber" ": %c0 */"::"i"(0xC0C0A));

  // Check that non-c modifiers work also
  // CHECK: i32 123
   __asm__         volatile("/* " "pickANumber2 " ": %0 */"::"i"(123));
}
