//RUN: %clang_cc1 %s -cl-std=c++ -pedantic -ast-dump -verify

//expected-no-diagnostics

//CHECK: |-VarDecl  foo {{.*}} 'const __global int' constexpr cinit
constexpr int foo = 0;

class c {
public:
  //CHECK: `-VarDecl {{.*}} foo2 'const __global int' static constexpr cinit
  static constexpr int foo2 = 0;
};
