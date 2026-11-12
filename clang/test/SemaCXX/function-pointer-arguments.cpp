//RUN: %clang_cc1 -fsyntax-only -verify %s

namespace PR16570 {
  int f1(int, int);
  int f2(const int, int);
  int f3(int&, int);
  int f4(const int&, int);

  void good() {
    int(*g1)(int, int) = f1;
    int(*g2)(const int, int) = f1;
    int(*g3)(volatile int, int) = f1;
    int(*g4)(int, int) = f2;
    int(*g5)(const int, int) = f2;
    int(*g6)(volatile int, int) = f2;
    int(*g7)(int&, int) = f3;
    int(*g8)(const int&, int) = f4;
  }

  void bad() {
    void (*g1)(int, int) = f1;
    // expected-error@-1 {{different return type ('void' vs 'int'}}
    const int (*g2)(int, int) = f1;
    // expected-error@-1 {{different return type ('const int' vs 'int')}}

    int (*g3)(char, int) = f1;
    // expected-error@-1 {{type mismatch at 1st parameter ('char' vs 'int')}}
    int (*g4)(int, char) = f1;
    // expected-error@-1 {{type mismatch at 2nd parameter ('char' vs 'int')}}

    int (*g5)(int) = f1;
    // expected-error@-1 {{different number of parameters (1 vs 2)}}

    int (*g6)(int, int, int) = f1;
    // expected-error@-1 {{different number of parameters (3 vs 2)}}

    int (*g7)(const int, char) = f1;
    // expected-error@-1 {{type mismatch at 2nd parameter ('char' vs 'int')}}
    int (*g8)(int, char) = f2;
    // expected-error@-1 {{type mismatch at 2nd parameter ('char' vs 'int')}}
    int (*g9)(const int&, char) = f3;
    // expected-error@-1 {{type mismatch at 1st parameter ('const int &' vs 'int &')}}
    int (*g10)(int&, char) = f4;
    // expected-error@-1 {{type mismatch at 1st parameter ('int &' vs 'const int &')}}
  }

  typedef void (*F)(const char * __restrict__, int);
  void g(const char *, unsigned);
  F f = g;
  // expected-error@-1 {{type mismatch at 2nd parameter ('int' vs 'unsigned int')}}

}
