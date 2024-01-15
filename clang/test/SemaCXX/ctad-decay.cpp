// RUN: %clang_cc1 -fsyntax-only -verify -Wno-unused-value -std=c++20 %s
// expected-no-diagnostics

namespace GH51710 {

template<typename T>
struct A{
  A(T f());
  A(int f(), T);

  A(T array[10]);
  A(int array[10], T);
};

int foo();

void bar() {
  A test1(foo);
  A test2(foo, 1);

  int array[10];
  A test3(array);
  A test4(array, 1);
}

} // namespace GH51710
