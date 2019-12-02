//RUN: %clang_cc1 %s -cl-std=clc++ -pedantic -ast-dump -verify | FileCheck %s

__constant int i = 1;
//CHECK: |-VarDecl {{.*}} ai '__global int':'__global int'
auto ai = i;

kernel void test() {
  int i;
  //CHECK: VarDecl {{.*}} ai 'int':'int'
  auto ai = i;

  constexpr int c = 1;
  //CHECK: VarDecl {{.*}} used cai '__constant int':'__constant int'
  __constant auto cai = c;
  //CHECK: VarDecl {{.*}} aii 'int':'int'
  auto aii = cai;

  //CHECK: VarDecl {{.*}} ref 'int &'
  auto &ref = i;
  //CHECK: VarDecl {{.*}} ptr 'int *'
  auto *ptr = &i;
  //CHECK: VarDecl {{.*}} ref_c '__constant int &'
  auto &ref_c = cai;

  //CHECK: VarDecl {{.*}} ptrptr 'int *__generic *'
  auto **ptrptr = &ptr;
  //CHECK: VarDecl {{.*}} refptr 'int *__generic &'
  auto *&refptr = ptr;

  //CHECK: VarDecl {{.*}} invalid gref '__global auto &'
  __global auto &gref = i; //expected-error{{variable 'gref' with type '__global auto &' has incompatible initializer of type 'int'}}
  __local int *ptr_l;
  //CHECK: VarDecl {{.*}} invalid gptr '__global auto *'
  __global auto *gptr = ptr_l; //expected-error{{variable 'gptr' with type '__global auto *' has incompatible initializer of type '__local int *'}}
}
