//RUN: %clang_cc1 %s -cl-std=c++ -pedantic -ast-dump -verify | FileCheck %s

//expected-no-diagnostics

//CHECK: |-VarDecl {{.*}} foo 'const __global int'
constexpr int foo = 0;

class c {
public:
  //CHECK: `-VarDecl {{.*}} foo2 'const __global int'
  static constexpr int foo2 = 0;
};

struct c1 {};

// We only deduce addr space in type alias in pointer types.
//CHECK: TypeAliasDecl {{.*}} alias_c1 'c1'
using alias_c1 = c1;
//CHECK: TypeAliasDecl {{.*}} alias_c1_ptr '__generic c1 *'
using alias_c1_ptr = c1 *;

struct c2 {
  alias_c1 y;
  alias_c1_ptr ptr = &y;
};

