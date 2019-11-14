// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -o - -emit-interface-stubs %s | \
// RUN: FileCheck %s

// RUN: %clang -target x86_64-unknown-linux-gnu -o - -c %s | llvm-nm - 2>&1 | \
// RUN: FileCheck -check-prefix=CHECK-SYMBOLS %s

// CHECK: Symbols:
// CHECK-DAG:  "_ZN3qux3barEii" : { Type: Func }
// CHECK-DAG:  "_ZN3baz3addIiEET_S1_S1_" : { Type: Func }
// CHECK-DAG:  "_Z4fbarff" : { Type: Func }
// CHECK-DAG:  "_ZN3baz3addIfEET_S1_S1_" : { Type: Func }

// Same symbols just different order.
// CHECK-SYMBOLS-DAG:  _Z4fbarff
// CHECK-SYMBOLS-DAG:  _ZN3baz3addIfEET_S1_S1_
// CHECK-SYMBOLS-DAG:  _ZN3baz3addIiEET_S1_S1_
// CHECK-SYMBOLS-DAG:  _ZN3qux3barEii

namespace baz {
template <typename T>
T add(T a, T b) {
  return a + b;
}
} // namespace baz

namespace qux {
int bar(int a, int b) { return baz::add<int>(a, b); }
} // namespace qux

float fbar(float a, float b) { return baz::add<float>(a, b); }
