// RUN: %clang_cc1 -triple mips64-none-linux-gnu -emit-pch -o %t %s
// REQUIRES: mips-registered-target
// RUN: %clang_cc1 -x ast -ast-print %t | FileCheck %s

// Make sure the semantics of FloatingLiterals are stored correctly in
// the AST. Previously, the ASTWriter didn't store anything and the
// reader assumed PPC 128-bit float semantics, which is incorrect for
// targets with 128-bit IEEE long doubles.

long double foo = 1.0E4000L;
// CHECK: long double foo = 1.00000000000000000000000000000000004E+4000L;

// Just as well check the others are still sane while we're here...

double bar = 1.0E300;
// CHECK: double bar = 1.0000000000000001E+300;

float wibble = 1.0E40;
// CHECK: float wibble = 1.0E+40;
