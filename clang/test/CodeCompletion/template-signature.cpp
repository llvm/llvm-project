template <int, char y> float overloaded(int);
template <class, int x> bool overloaded(char);

auto m = overloaded<1, 2>(0);
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:4:21 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: OPENING_PAREN_LOC: {{.*}}4:20
// CHECK-CC1-DAG: OVERLOAD: [#float#]overloaded<<#int#>, char y>[#()#]
// CHECK-CC1-DAG: OVERLOAD: [#bool#]overloaded<<#class#>, int x>[#()#]
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:4:24 %s -o - | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2-NOT: OVERLOAD: {{.*}}int x
// CHECK-CC2: OVERLOAD: [#float#]overloaded<int, <#char y#>>[#()#]

template <class T, T... args> int n = 0;
int val = n<int, 1, 2, 3>;
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:14:18 %s -o - | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: OVERLOAD: [#int#]n<class T, <#T ...args#>>
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:14:24 %s -o - | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: OVERLOAD: [#int#]n<class T, T ...args>

template <typename> struct Vector {};
template <typename Element, template <typename E> class Container = Vector>
struct Collection { Container<Element> container; };
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:22:31 %s -o - | FileCheck -check-prefix=CHECK-CC5 %s
// CHECK-CC5: OVERLOAD: [#class#]Container<<#typename E#>>
Collection<int, Vector> collection;
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:25:12 %s -o - | FileCheck -check-prefix=CHECK-CC6 %s
// CHECK-CC6: OVERLOAD: [#struct#]Collection<<#typename Element#>>

