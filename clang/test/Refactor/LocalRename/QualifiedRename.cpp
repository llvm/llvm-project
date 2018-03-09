// RUN: clang-refactor local-rename -old-qualified-name="foo::A" -new-qualified-name="bar::B" %s -- -std=c++11 2>&1 | grep -v CHECK | FileCheck %s

namespace foo {
class A {};
}
// CHECK: namespace foo {
// CHECK-NEXT: class B {};
// CHECK-NEXT: }

namespace bar {
void f(foo::A* a) {
  foo::A b;
}
// CHECK: void f(B* a) {
// CHECK-NEXT:   B b;
// CHECK-NEXT: }
}

void f(foo::A* a) {
  foo::A b;
}
// CHECK: void f(bar::B* a) {
// CHECK-NEXT:   bar::B b;
// CHECK-NEXT: }
