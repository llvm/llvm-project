// RUN: %clang_cc1 -verify -fopenmp -triple x86_64-unknown-unknown %s -ast-dump | FileCheck %s
// expected-no-diagnostics

#pragma omp begin declare variant match(device={arch(x86_64)})

void bar() {}

// CHECK: FunctionDecl {{.*}} bar[device={arch(x86_64)}] 'void ()'

#pragma omp end declare variant
