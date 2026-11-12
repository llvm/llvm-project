// Check that the non-clang/non-filechecked runlines execute
// RUN: cp %s %S/../Output/execute-all-runlines.copy.c
// RUN: cp %S/../Output/execute-all-runlines.copy.c %s.copy.c
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp %s.copy.c -emit-llvm-bc -o %t-host.bc
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fopenmp-host-ir-file-path %t-host.bc %s.copy.c -emit-llvm -o - | FileCheck %s --check-prefix=CHECK1
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-pch %s -o %t
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -include-pch %t %s.copy.c -emit-llvm -o - | FileCheck %s --check-prefix=CHECK2
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -emit-pch %s -o %t
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -include-pch %t %s.copy.c -emit-llvm -o - | FileCheck %s --check-prefix=CHECK3


#ifndef HEADER
#define HEADER

void use(int);

void test(int a)
{
}

#endif
