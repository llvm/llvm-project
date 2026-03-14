// RUN: %clang_cc1 -Wno-error=return-type -Wno-error=int-conversion %s -emit-llvm-only
// REQUIRES: LP64

// Mismatched type between return and function result.
int test2() { return; }
void test3() { return 4; }


void test4() {
bar:
baz:
blong:
bing:
 ;

// PR5131
static long x = &&bar - &&baz;
static long y = &&baz;
  &&bing;
  &&blong;
  if (y)
    goto *y;

  goto *x;
}

// PR3869
int test5(long long b) {
  static void *lbls[] = { &&lbl };
  goto *b;
 lbl:
  return 0;
}

