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
  int first;
  int second;
  int third;
};

Aggregate a{1, 2, 3};
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:50:13 %s | FileCheck -check-prefix=CHECK-AGGREGATE-1 %s
// CHECK-AGGREGATE-1: OVERLOAD: Aggregate{<#int first#>, int second, int third}
// CHECK-AGGREGATE-1: OVERLOAD: Aggregate{<#const Aggregate &#>}
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:50:16 %s | FileCheck -check-prefix=CHECK-AGGREGATE-2 %s
// CHECK-AGGREGATE-2: OVERLOAD: Aggregate{int first, <#int second#>, int third}
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:50:18 %s | FileCheck -check-prefix=CHECK-AGGREGATE-3 %s
// CHECK-AGGREGATE-3: OVERLOAD: Aggregate{int first, int second, <#int third#>}

Aggregate d{.second=1, .first=2, 3, 4, };
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:59:13 %s | FileCheck -check-prefix=CHECK-DESIG-1 %s
// CHECK-DESIG-1: OVERLOAD: Aggregate{<#int first#>, int second, int third}
// CHECK-DESIG-1: OVERLOAD: Aggregate{<#const Aggregate &#>}
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:59:24 %s | FileCheck -check-prefix=CHECK-DESIG-2 %s
// CHECK-DESIG-2: OVERLOAD: Aggregate{int first, int second, <#int third#>}
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:59:34 %s | FileCheck -check-prefix=CHECK-DESIG-3 %s
// CHECK-DESIG-3: OVERLOAD: Aggregate{int first, <#int second#>, int third}
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:59:37 %s | FileCheck -check-prefix=CHECK-DESIG-4 %s
// CHECK-DESIG-4: OVERLOAD: Aggregate{int first, int second, <#int third#>}
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:59:38 %s | FileCheck -check-prefix=CHECK-DESIG-5 %s --allow-empty
// CHECK-DESIG-5-NOT: OVERLOAD
