// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
//
// XFAIL: *
//
// Issue: Static local variable with recursive initialization
//
// When a static local variable is initialized by calling the function that
// contains it, CIR fails during initialization code generation. This pattern
// requires special guard variable handling to prevent infinite recursion at
// runtime and detect the recursion during initialization.

int a() { static int b = a(); }
