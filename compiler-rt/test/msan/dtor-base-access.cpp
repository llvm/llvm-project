// RUN: %clangxx_msan %s -O0 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && MSAN_OPTIONS=poison_in_dtor=1 %run %t >%t.out 2>&1

// RUN: %clangxx_msan %s -O1 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && MSAN_OPTIONS=poison_in_dtor=1 %run %t >%t.out 2>&1

// RUN: %clangxx_msan %s -O2 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && MSAN_OPTIONS=poison_in_dtor=1 %run %t >%t.out 2>&1

#include <sanitizer/msan_interface.h>
#include <assert.h>

class Base {
 public:
   int b;
   Base() { b = 1; }
   ~Base();
};

class TrivialBaseBefore {
public:
  int tb0;
  TrivialBaseBefore() { tb0 = 1; }
};

class TrivialBaseAfter {
public:
  int tb1;
  TrivialBaseAfter() { tb1 = 1; }
};

class Derived : public TrivialBaseBefore, public Base, public TrivialBaseAfter {
public:
  int d;
  Derived() { d = 1; }
  ~Derived();
};

Derived *g;

Base::~Base() {
  // ok to access its own members and earlier bases
  assert(__msan_test_shadow(&g->tb0, sizeof(g->tb0)) == -1);
  assert(__msan_test_shadow(&g->b, sizeof(g->b)) == -1);
  // not ok to access others
  assert(__msan_test_shadow(&g->tb1, sizeof(g->tb1)) == 0);
  assert(__msan_test_shadow(&g->d, sizeof(g->d)) == 0);
}

Derived::~Derived() {
  // ok to access everything
  assert(__msan_test_shadow(&g->tb0, sizeof(g->tb0)) == -1);
  assert(__msan_test_shadow(&g->b, sizeof(g->b)) == -1);
  assert(__msan_test_shadow(&g->tb1, sizeof(g->tb1)) == -1);
  assert(__msan_test_shadow(&g->d, sizeof(g->d)) == -1);
}

int main() {
  g = new Derived();
  // ok to access everything
  assert(__msan_test_shadow(&g->tb0, sizeof(g->tb0)) == -1);
  assert(__msan_test_shadow(&g->b, sizeof(g->b)) == -1);
  assert(__msan_test_shadow(&g->tb1, sizeof(g->tb1)) == -1);
  assert(__msan_test_shadow(&g->d, sizeof(g->d)) == -1);

  g->~Derived();
  // not ok to access everything
  assert(__msan_test_shadow(&g->tb0, sizeof(g->tb0)) == 0);
  assert(__msan_test_shadow(&g->b, sizeof(g->b)) == 0);
  assert(__msan_test_shadow(&g->tb1, sizeof(g->tb1)) == 0);
  assert(__msan_test_shadow(&g->d, sizeof(g->d)) == 0);
  return 0;
}
