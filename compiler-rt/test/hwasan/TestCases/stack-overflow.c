// RUN: %clang_hwasan -O0 -g %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clang_hwasan -O3 -g %s -o %t && not %run %t 2>&1 | FileCheck %s

// Stack histories currently are not recorded on x86.
// XFAIL: target=x86_64{{.*}}

__attribute((noinline)) void buggy() {
  char c[64];
  char *volatile p = c;
  p[65] = 0;
}

int main() {
  buggy();
  // CHECK: WRITE of size 1 at
  // CHECK: #0 {{.*}} in buggy{{.*}}stack-overflow.c:[[@LINE-6]]
  // CHECK: Cause: stack tag-mismatch
  // CHECK: is located in stack of thread
  // CHECK: Potentially referenced stack objects:
  // CHECK: Cause: stack-buffer-overflow
  // CHECK-NEXT: 0x{{.*}} is located 1 bytes after a 64-byte local variable c [0x{{.*}},0x{{.*}}) in buggy {{.*}}stack-overflow.c:
  // CHECK: Memory tags around the buggy address

  // CHECK: SUMMARY: HWAddressSanitizer: tag-mismatch {{.*}} in buggy
}
