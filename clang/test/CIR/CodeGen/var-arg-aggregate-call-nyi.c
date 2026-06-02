// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir -verify %s -o -

struct S {
  int a, b, c;
};

int sink(int n, ...);

int call(struct S s) {
  // expected-error@+1 {{ClangIR code gen Not Yet Implemented: aggregate argument passed through variadic ellipsis}}
  return sink(1, s);
}
