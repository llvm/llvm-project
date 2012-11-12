// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-pth -o %t %S/pth.h
// RUN: %clang_cc1 -triple i386-unknown-unknown -include-pth %t -fsyntax-only %s 2>&1 | FileCheck %s

#error This is the only diagnostic

// CHECK: This is the only diagnostic
// CHECK: 1 error generated.