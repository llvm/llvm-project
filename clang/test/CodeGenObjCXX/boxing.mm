// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s

@interface NSNumber
+ (id)numberWithInt:(int)n;
@end

int n = 1;
int m = (@(n++), 0);

// CHECK: define {{.*}} @__cxx_global_var_init()
// CHECK: load i32, ptr @n
// CHECK: store i32 %{{.*}}, ptr @n
