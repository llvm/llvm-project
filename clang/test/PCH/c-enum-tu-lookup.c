// RUN: %clang_cc1 -x c-header -emit-pch -o %t.pch %s
// RUN: %clang_cc1 -include-pch %t.pch -ast-dump-all -ast-dump-lookups %s | FileCheck %s

#ifndef HEADER
#define HEADER
enum { thrd_success = 4 };
enum E { A = 1 };
extern int v;
#else
int x;
// CHECK-DAG: DeclarationName 'thrd_success'
// CHECK-DAG: DeclarationName 'A'
// CHECK-DAG: DeclarationName 'E'
// CHECK-DAG: DeclarationName 'v'
// CHECK-DAG: DeclarationName 'x'
#endif
