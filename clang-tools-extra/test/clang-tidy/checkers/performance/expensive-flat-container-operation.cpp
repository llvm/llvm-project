// RUN: %check_clang_tidy %s performance-expensive-flat-container-operation %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:             [{key: performance-expensive-flat-container-operation.WarnOutsideLoops, \
// RUN:               value: false}, \
// RUN:              {key: performance-expensive-flat-container-operation.FlatContainers, \
// RUN:               value: "::MyFlatSet"}] \
// RUN:             }"

#include <stddef.h>

template <class Key> struct MyFlatSet {
  using key_type = Key;
  void erase(const key_type &x);
};

void testWhileLoop() {
  MyFlatSet<int> set;
  while (true) {
    set.erase(0);
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Single element
    // operations are expensive for flat containers.
  }
}

void testOutsideLoop(MyFlatSet<int> &set) { set.erase(0); }

void testForLoop() {
  MyFlatSet<int> set;
  for (;;) {
    set.erase(0);
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Single element
    // operations are expensive for flat containers.
  }
}

template <class Iterable> void testRangeForLoop(const Iterable &v) {
  MyFlatSet<int> set;
  for (const auto &x : v) {
    set.erase(0);
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Single element
    // operations are expensive for flat containers.
  }
}

void testDoWhileLoop() {
  MyFlatSet<int> set;
  do {
    set.erase(0);
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Single element
    // operations are expensive for flat containers.
  } while (true);
}

void testMultipleCasesInLoop() {
  MyFlatSet<int> set;
  for (;;) {
    set.erase(0);
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Single element
    // operations are expensive for flat containers.
    set.erase(1);
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Single element
    // operations are expensive for flat containers.
  }

  MyFlatSet<int> set2;
  MyFlatSet<int> set3;
  for (;;) {
    set2.erase(0);
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Single element
    // operations are expensive for flat containers.
    set3.erase(1);
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Single element
    // operations are expensive for flat containers.
  }

  MyFlatSet<int> set4;
  for (;;) {
    set4.erase(0);
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: Single element
    // operations are expensive for flat containers.
    MyFlatSet<int> set5;
    set5.erase(1);
  }
}

void testOperationAndDeclarationInLoop() {
  for (;;) {
    MyFlatSet<int> set;
    set.erase(0);
  }
}
