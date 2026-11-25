// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c23 -ffixed-point -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c23 -ffixed-point -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s

// C23 6.7.3.3p16: conversion to an enumeration with a fixed bool underlying
// type has the semantics of conversion to bool. Mirrors the cases in
// clang/test/CodeGen/enum-bool.c, restricted to what CIR can lower today.

enum E : bool { Zero, One };

enum E a(int x) { return (enum E)x; }
// CIR-LABEL: cir.func {{.*}} @a(
// CIR:         cir.cast int_to_bool
// LLVM-LABEL: define {{.*}} @a(
// LLVM:         icmp ne i32

enum E c(double x) { return (enum E)x; }
// CIR-LABEL: cir.func {{.*}} @c(
// CIR:         cir.cast float_to_bool
// LLVM-LABEL: define {{.*}} @c(
// LLVM:         fcmp une double

enum E g(double _Complex x) { return (enum E)x; }
// CIR-LABEL: cir.func {{.*}} @g(
// CIR-COUNT-2: cir.cast float_to_bool
// CIR:         cir.const #true
// LLVM-LABEL: define {{.*}} @g(
// LLVM-COUNT-2: fcmp une double

enum E h(enum E e, int x) { e += x; return e; }
// CIR-LABEL: cir.func {{.*}} @h(
// CIR:         cir.cast int_to_bool
// LLVM-LABEL: define {{.*}} @h(

enum E k(enum E e, _Accum x) { e += x; return e; }
// CIR-LABEL: cir.func {{.*}} @k(
// CIR:         cir.cast int_to_bool

enum E l(enum E e) { return ++e; }
// CIR-LABEL: cir.func {{.*}} @l(
// CIR:         cir.const #true

enum E m(enum E e) { e++; return e; }
// CIR-LABEL: cir.func {{.*}} @m(
// CIR:         cir.const #true
