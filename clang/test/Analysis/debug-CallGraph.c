// RUN: %clang_cc1 -analyze -analyzer-checker=debug.DumpCallGraph %s 2>&1 | FileCheck %s

static void mmm(int y) {
  if (y != 0)
      y++;
  y = y/0;
}

static int foo(int x, int y) {
    mmm(y);
    if (x != 0)
      x++;
    return 5/x;
}

void aaa() {
  foo(1,2);
}

// CHECK:--- Call graph Dump ---
// CHECK: Function: < root > calls: aaa
