// RUN: %clang_cc1 -ast-dump -std=c++2c -triple x86_64-unknown-linux-gnu -fms-extensions %s | FileCheck --match-full-lines --check-prefix=CHECK %s

int a1;
// CHECK: |-VarDecl {{.*}} a1 'int' external-linkage

[[gnu::visibility("default")]] int a2;
// CHECK: |-VarDecl {{.*}} a2 'int' external-linkage

__declspec(dllexport) int a3;
// CHECK: |-VarDecl {{.*}} a3 'int' external-linkage

[[gnu::visibility("hidden")]] int b;
// CHECK: |-VarDecl {{.*}} b 'int' external-linkage hidden-visibility

[[gnu::visibility("protected")]] int c;
// CHECK: `-VarDecl {{.*}} c 'int' external-linkage protected-visibility
