// RUN: %clang -target i386-unknown-unknown -S -emit-llvm -std=gnu89 -o - %s | FileCheck %s
// RUN: %clang -target i386-unknown-unknown -S -emit-llvm -fgnu89-inline -o - %s | FileCheck %s
// PR5253

// If an extern inline function is redefined, functions should call the
// redefinition.
extern inline int f(int a) {return a;}
int g(void) {return f(0);}
// CHECK: call i32 @f
int f(int b) {return 1+b;}
// CHECK: load i32* %{{.*}}
// CHECK: add nsw i32 1, %{{.*}}
int h(void) {return f(1);}
// CHECK: call i32 @f

// It shouldn't matter if the function was redefined static.
extern inline int f2(int a, int b) {return a+b;}
int g2(void) {return f2(0,1);}
// CHECK: call i32 @f2
static int f2(int a, int b) {return a*b;}
// CHECK: load i32* %{{.*}}
// CHECK: load i32* %{{.*}}
// CHECK: mul nsw i32 %{{.*}}, %{{.*}}
int h2(void) {return f2(1,2);}
// CHECK: call i32 @f2

