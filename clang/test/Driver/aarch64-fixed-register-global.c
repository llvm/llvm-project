// Check that -ffixed register handled for globals.
// Regression test for #76426
// RUN: %clang --target=aarch64-none-gnu -ffixed-x15 -### %s 2>&1 | FileCheck %s
// CHECK-NOT: fatal error: error in backend: Invalid register name "x15".
register int i1 __asm__("x15");

int foo() {
  return i1;
}
int main() {
  return foo();
}
