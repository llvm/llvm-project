struct Foo { Foo(int); int abc; };

void test1() {
  Foo foo;
  foo.;
  // RUN: not %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-1):7 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
  // CHECK-CC1: COMPLETION: abc
}

void test2() {
  Foo foo = garbage();
  foo.;
  // RUN: not %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-1):7 %s -o - | FileCheck -check-prefix=CHECK-CC2 %s
  // CHECK-CC2: COMPLETION: abc
}
