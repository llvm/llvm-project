template <class T, class U>
auto k(T t) -> decltype(t.U::template B<>::MEM);

struct S {};

void f() {
  k(S{});
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-1):5 %s -o - | FileCheck %s
  // CHECK: OVERLOAD: [#decltype(t.U::template B<>::MEM)#]k(<#T t#>)
}
