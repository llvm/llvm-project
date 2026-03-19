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
// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s -DTEST11
// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s -DTEST12
// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s -DTEST13
// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s -DTEST14
// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm-only -verify -pedantic %s -DTEST15

#if TEST1
int main; // expected-error{{main cannot be declared as a variable in the global scope}}

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
  extern int main;  // expected-error{{main cannot be declared as a variable in the global scope}}}
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

#elif TEST11
extern "C" {
  namespace Y {
    int main; // expected-error {{main cannot be declared as a variable with C language linkage}}}
  }
}
namespace ns {
  extern "C" int main; // expected-error {{main cannot be declared as a variable with C language linkage}}
}

#elif TEST12
extern "C" struct A { int main(); }; // ok

namespace c {
  extern "C" void main(); // expected-error {{'main' must return 'int'}} \
                          // expected-warning {{'main' should not be 'extern "C"'}}
}

extern "C" {
  namespace Z {
    void main(); // expected-error {{'main' must return 'int'}} \
                 // expected-warning {{'main' should not be 'extern "C"'}}
  }
}

namespace ns {
  extern "C" struct A {
    int main; // ok
  };

  extern "C" struct B {
    int main(); // ok
  };
}

#elif TEST13
// expected-no-diagnostics
extern "C++" {
  int main();
}

extern "C++" int main();

namespace ns1 {
  extern "C++" int main(); // ok
  extern "C" {
    extern "C++" {
      int main(void *); // ok
    }
  }
}

namespace ns2 {
  extern "C++" void main() {} // ok
}

#elif TEST14
extern "C" {
  int main(); // expected-warning {{'main' should not be 'extern "C"'}}
}

extern "C" int main(); // expected-warning {{'main' should not be 'extern "C"'}}

#elif TEST15
extern "C" __attribute__((visibility("default"))) __attribute__((weak))
int main(); // expected-warning {{'main' should not be 'extern "C"'}}

unsigned long g() {
  return reinterpret_cast<unsigned long>(&main); // expected-warning {{referring to 'main' within an expression is a Clang extension}}
}

#else
#error Unknown Test
#endif
