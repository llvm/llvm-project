template <typename T>
struct Foo {};
template <typename T>
struct Foo<T *> { Foo(T); };

void foo() {
  Foo<int>();
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:7:12 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
  // CHECK-CC1: OVERLOAD: Foo()
  // CHECK-CC1: OVERLOAD: Foo(<#const Foo<int> &#>)
  // CHECK-CC1: OVERLOAD: Foo(<#Foo<int> &&#>
  Foo<int *>(3);
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:12:14 %s -o - | FileCheck -check-prefix=CHECK-CC2 %s
  // CHECK-CC2: OVERLOAD: Foo(<#int#>)
  // CHECK-CC2: OVERLOAD: Foo(<#const Foo<int *> &#>)
  // CHECK-CC2: OVERLOAD: Foo(<#Foo<int *> &&#>
}
