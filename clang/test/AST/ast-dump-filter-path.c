// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ast-dump -ast-dump-filter-path "*ast-dump-filter-path.c" %s | FileCheck %s

int x;
// CHECK: VarDecl {{.*}} x

#define MAKE_VAR(name) int name;
MAKE_VAR(y)
// CHECK: VarDecl {{.*}} y