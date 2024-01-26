// RUN: %clang_hwasan -g %s -o %t && not %run %t 2>&1 | FileCheck %s

// Dynamic allocation of stack objects does not affect FP, so the backend should
// still be using FP-relative debug info locations that we can use to find stack
// objects.

// Stack histories are currently not recorded on x86.
// XFAIL: target=x86_64{{.*}}

__attribute((noinline))
char *buggy(int b) {
  char c[64];
  char *volatile p = c;
  if (b) {
    p = __builtin_alloca(64);
    p = c;
  }
  return p;
}

int main() {
  char *p = buggy(1);
  // CHECK: Potentially referenced stack objects:
  // CHECK-NEXT: use-after-scope
  // CHECK-NEXT: 0x{{.*}} is located 0 bytes inside a 64-byte local variable c [0x{{.*}},0x{{.*}}) in buggy
  p[0] = 0;
}
