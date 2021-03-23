// RUN: %clang_hwasan -g %s -o %t && not %run %t 2>&1 | FileCheck %s

// Dynamic stack realignment causes debug info locations to use non-FP-relative
// offsets because stack frames are realigned below FP, which means that we
// can't associate addresses with stack objects in this case. Ideally we should
// be able to handle this case somehow (e.g. by using a different register for
// DW_AT_frame_base) but at least we shouldn't get confused by it.

// Stack aliasing is not implemented on x86.
// XFAIL: x86_64

__attribute((noinline))
char *buggy() {
  _Alignas(64) char c[64];
  char *volatile p = c;
  return p;
}

int main() {
  char *p = buggy();
  // CHECK-NOT: Potentially referenced stack objects:
  p[0] = 0;
}
