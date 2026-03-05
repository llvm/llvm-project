// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
//
// XFAIL: *
//
// Issue: Static local variable with __used__ attribute
//
// When a function contains a static local variable with the __used__ attribute,
// CIR fails to properly handle the attribute during code generation.
// The __used__ attribute prevents the variable from being optimized away even
// if it appears unused.

void a() { __attribute__((__used__)) static void *b; }
