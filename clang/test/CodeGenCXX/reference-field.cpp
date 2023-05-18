// RUN: %clang_cc1 -emit-llvm -triple %itanium_abi_triple -o - %s -O2 | FileCheck %s

// Make sure the call to b() doesn't get optimized out.
extern struct x {char& x,y;}y;
int b();      
int a() { if (!&y.x) b(); }

// CHECK: @_Z1bv
