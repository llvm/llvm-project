// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only %s

struct Bar {int a;};
const Bar arr[2] = {{1}};

struct Foo {};

const int b = 2;

void foo(int a) {
  Foo *foo_array;
  foo_array = new Foo[arr[0].a];
}
