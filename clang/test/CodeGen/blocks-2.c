// RUN: %clang_cc1 -g %s -emit-llvm -o %t -fblocks
// RUN: grep "func.start" %t | count 4
// RUN: %clang_cc1 -g %s -triple i386-unknown-unknown -emit-llvm -o %t -fblocks -fblock-introspection
// RUN: grep "v8@?0i4" %t | count 1
// RUN: %clang_cc1 -g %s -triple i386-unknown-unknown -emit-llvm -o %t -fblocks
// RUN: grep "v8@?0i4" %t | count 0
// 1 declaration, 1 bar, 1 test_block_dbg and 1 for the block.
// XFAIL: *

static __inline__ __attribute__((always_inline)) int bar(int va, int vb) { return (va == vb); }

int test_block_dbg() {
  extern int g;
  static int i = 1;
  ^(int j){ i = bar(3,4); }(0);
  return i + g;
}

