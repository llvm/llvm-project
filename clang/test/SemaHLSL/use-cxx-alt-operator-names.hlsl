// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library %s -ast-dump | FileCheck %s

// CHECK: -FunctionDecl {{.*}} and 'void ()'
void and() {}

// CHECK: -FunctionDecl {{.*}} and_eq 'void ()'
void and_eq() {}

// CHECK: -FunctionDecl {{.*}} bitand 'void ()'
void bitand() {}

// CHECK: -FunctionDecl {{.*}} bitor 'void ()'
void bitor() {}

// CHECK: -FunctionDecl {{.*}} compl 'void ()'
void compl() {}

// CHECK: -FunctionDecl {{.*}} not 'void ()'
void not() {}

// CHECK: -FunctionDecl {{.*}} not_eq 'void ()'
void not_eq() {}

// CHECK: -FunctionDecl {{.*}} or 'void ()'
void or() {}

// CHECK: -FunctionDecl {{.*}} or_eq 'void ()'
void or_eq() {}

// CHECK: -FunctionDecl {{.*}} xor 'void ()'
void xor() {}

// CHECK: -FunctionDecl {{.*}} xor_eq 'void ()'
void xor_eq() {}
