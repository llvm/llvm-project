// Check that the non-clang/non-filechecked runlines execute
// RUN: cp %s %s.copy.c
// RUN: %clang_cc1 -fopenmp %s.copy.c -emit-llvm-bc -o %t-host.bc
// RUN: %clang_cc1 -fopenmp -fopenmp-host-ir-file-path %t-host.bc %s.copy.c -emit-llvm -o - | FileCheck %s

void use(int);

void test(int a)
{
}
