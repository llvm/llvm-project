void foo(int a, int b);
void foo(int a, int b, int c);

void test() {
  foo(10, );
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:5:10 %s -o - \
  // RUN: | FileCheck -check-prefix=CHECK-CC1 %s
  // CHECK-CC1: OPENING_PAREN_LOC: {{.*}}paren_locs.cpp:5:6

#define FOO foo(
  FOO 10, );
#undef FOO
  // RUN: not %clang_cc1 -fsyntax-only -code-completion-at=%s:11:10 %s -o - \
  // RUN: | FileCheck -check-prefix=CHECK-CC2 %s
  // CHECK-CC2: OPENING_PAREN_LOC: {{.*}}paren_locs.cpp:11:3

  struct Foo {
    Foo(int a, int b);
    Foo(int a, int b, int c);
  };
  Foo a(10, );
  // RUN: not %clang_cc1 -fsyntax-only -code-completion-at=%s:21:12 %s -o - \
  // RUN: | FileCheck -check-prefix=CHECK-CC3 %s
  // CHECK-CC3: OPENING_PAREN_LOC: {{.*}}paren_locs.cpp:21:8
  Foo(10, );
  // RUN: not %clang_cc1 -fsyntax-only -code-completion-at=%s:25:10 %s -o - \
  // RUN: | FileCheck -check-prefix=CHECK-CC4 %s
  // CHECK-CC4: OPENING_PAREN_LOC: {{.*}}paren_locs.cpp:25:6
  new Foo(10, );
  // RUN: not %clang_cc1 -fsyntax-only -code-completion-at=%s:29:15 %s -o - \
  // RUN: | FileCheck -check-prefix=CHECK-CC5 %s
  // CHECK-CC5: OPENING_PAREN_LOC: {{.*}}paren_locs.cpp:29:10
}
