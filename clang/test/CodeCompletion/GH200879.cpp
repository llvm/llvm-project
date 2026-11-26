// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:6:2 %s
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:10:2 %s

struct A {
  A(int = 0);
}/*invoke completion here*/;

struct B {
  B() noexcept(false);
}/*invoke completion here*/;
