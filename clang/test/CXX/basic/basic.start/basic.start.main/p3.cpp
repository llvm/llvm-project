// RUN: %clang_cc1 -fsyntax-only -verify %s -DTEST1
// RUN: %clang_cc1 -fsyntax-only -verify %s -DTEST2
// RUN: %clang_cc1 -fsyntax-only -verify %s -DTEST3
// RUN: %clang_cc1 -fsyntax-only -verify %s -DTEST4
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++14 -DTEST5
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++14 -DTEST6
// RUN: %clang_cc1 -fsyntax-only -verify %s -DTEST7
// RUN: %clang_cc1 -fsyntax-only -verify %s -DTEST8
// RUN: %clang_cc1 -fsyntax-only -verify %s -DTEST9
// RUN: %clang_cc1 -fsyntax-only -verify %s -DTEST10 -ffreestanding

#if TEST1
int main; // expected-error{{main cannot be declared as global variable}}

#elif TEST2
// expected-no-diagnostics
int f () {
  int main;
  return main;
}

#elif TEST3
// expected-no-diagnostics
void x(int main) {};
int y(int main);

#elif TEST4
// expected-no-diagnostics
class A {
  static int main;
};

#elif TEST5
// expected-no-diagnostics
template<class T> constexpr T main;

#elif TEST6
extern template<class T> constexpr T main; //expected-error{{expected unqualified-id}}

#elif TEST7
// expected-no-diagnostics
namespace foo {
  int main;
}

#elif TEST8
void z(void)
{
  extern int main;  // expected-error{{main cannot be declared as global variable}}
}

#elif TEST9
// expected-no-diagnostics
int q(void)
{
  static int main;
  return main;
}

#elif TEST10
// expected-no-diagnostics
int main;

#else
#error Unknown Test
#endif
