// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.DumpTraversal %s | FileCheck %s
// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.DumpTraversal -DUSE_EXPR %s | FileCheck %s

int a();
int b();
int c();

#ifdef USE_EXPR
#define CHECK(x) ((x) & 1)
#else
#define CHECK(x) (x)
#endif

void testRemoveDeadBindings() {
  int i = a();
  if (CHECK(i))
    a();
  else
    b();

  // At this point the symbol bound to 'i' is dead.
  // The effects of a() and b() are identical (they both invalidate globals).
  // We should unify the two paths here and only get one end-of-path node.
  c();
}

// CHECK: --END FUNCTION--
// CHECK-NOT: --END FUNCTION--
