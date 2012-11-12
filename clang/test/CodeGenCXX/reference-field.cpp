// RUN: %clang_cc1 -emit-llvm -o - %s -O2 | grep "@_Z1bv"

// Make sure the call to b() doesn't get optimized out.
extern struct x {char& x,y;}y;
int b();      
int a() { if (!&y.x) b(); }
