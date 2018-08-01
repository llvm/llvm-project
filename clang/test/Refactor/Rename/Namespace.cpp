namespace gcc /* Test 1 */ {  // CHECK: rename [[@LINE]]:11 -> [[@LINE]]:14
  int x;
}

void boo() {
  gcc::x = 42;                // CHECK: rename [[@LINE]]:3 -> [[@LINE]]:6
}

// Test 1.
// RUN: clang-refactor-test rename-initiate -at=%s:1:11 -new-name=clang %s | FileCheck %s

namespace ns1 {
namespace ns2 { // CHECK2: rename [[@LINE]]:11 -> [[@LINE]]:14
void f();
}
}

void testVisitTwice() {
  ns1::ns2::f(); // CHECK2: rename [[@LINE]]:8 -> [[@LINE]]:11
}

// RUN: clang-refactor-test rename-initiate -at=%s:19:8 -new-name=clang %s | FileCheck --check-prefix=CHECK2 %s
