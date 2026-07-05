// RUN: %clangxx_msan %s -O0 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && %run %t >%t.out 2>&1

// RUN: %clangxx_msan %s -O1 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && %run %t >%t.out 2>&1

// RUN: %clangxx_msan %s -O2 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && %run %t >%t.out 2>&1

// TODO Success pending on resolution of
// https://github.com/google/sanitizers/issues/596

// XFAIL: *

#include <assert.h>
#include <sanitizer/msan_interface.h>

template <class T> class Vector {
 public:
  int size;
  ~Vector() {}
};

struct NonTrivial {
  int a;
  Vector<int> v;
};

struct Trivial {
  int a;
  int b;
};

int main() {
  NonTrivial *nt = new NonTrivial();
  nt->~NonTrivial();
  assert(__msan_test_shadow(nt, sizeof(*nt)) != -1);

  Trivial *t = new Trivial();
  t->~Trivial();
  assert(__msan_test_shadow(t, sizeof(*t)) != -1);

  return 0;
}
