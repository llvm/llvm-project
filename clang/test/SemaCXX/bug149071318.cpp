// RUN: %clang_cc1 -std=c++20 -Wno-all -Wunsafe-buffer-usage \
// RUN:            -verify %s

// This example uncovered a bug in UnsafeBufferUsage.cpp, where the
// code assumed that a CXXMethodDecl always have an identifier.

int printf( const char* format, char *); // <-- Fake decl of `printf`; to reproduce the bug, this example needs an implicit cast within a printf call.

namespace std { // fake std namespace; to reproduce the bug, a CXXConversionDecl needs to be in std namespace.
  class X {
    char * p;
  public:    
    operator char*() {return p;}
  };

  class Y {
  public:
    X x;
  };

}

void test(std::Y &y) {
  // Here `y.x` involves an implicit cast and calls the overloaded cast operator, which has no identifier:
  printf("%s", y.x); // expected-warning{{function 'printf' is unsafe}} expected-note{{}}
}
