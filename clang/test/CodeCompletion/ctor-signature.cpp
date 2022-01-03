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

namespace std {
template <typename> struct initializer_list {};
} // namespace std

struct Bar {
  // CHECK-BRACED: OVERLOAD: Bar{<#int#>}
  Bar(int);
  // CHECK-BRACED: OVERLOAD: Bar{<#double#>, double}
  Bar(double, double);
  // FIXME: no support for init-list constructors yet.
  // CHECK-BRACED-NOT: OVERLOAD: {{.*}}char
  Bar(std::initializer_list<char> C);
  // CHECK-BRACED: OVERLOAD: Bar{<#const Bar &#>}
  // CHECK-BRACED: OVERLOAD: Bar{<#T *Pointer#>}
  template <typename T> Bar(T *Pointer);
};

auto b1 = Bar{};
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:36:15 %s | FileCheck -check-prefix=CHECK-BRACED %s
Bar b2{};
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:38:8 %s | FileCheck -check-prefix=CHECK-BRACED %s
static int consumeBar(Bar) { return 0; }
int b3 = consumeBar({});
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:41:22 %s | FileCheck -check-prefix=CHECK-BRACED %s

struct Aggregate {
  // FIXME: no support for aggregates yet.
  // CHECK-AGGREGATE-NOT: OVERLOAD: Aggregate{<#const Aggregate &#>}
  // CHECK-AGGREGATE-NOT: OVERLOAD: {{.*}}first
  int first;
  int second;
};

Aggregate a{};
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:52:13 %s | FileCheck -check-prefix=CHECK-AGGREGATE %s

