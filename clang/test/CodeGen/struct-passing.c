// This verifies that structs returned from functions by value are passed
// correctly according to their attributes and the ABI.
// SEE: PR3835

// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s

typedef int T0;
typedef struct { int a[16]; } T1;

T0 __attribute__((const)) f0(void);
T0 __attribute__((pure)) f1(void);
T1 __attribute__((const)) f2(void);
T1 __attribute__((pure)) f3(void);
void __attribute__((const)) f4(T1 a);
void __attribute__((pure)) f5(T1 a);

void *ps[] = { f0, f1, f2, f3, f4, f5 };

// CHECK: declare i32 @f0() [[RN:#[0-9]+]]
// CHECK: declare i32 @f1() [[RO:#[0-9]+]]
// CHECK: declare void @f2({{.*}} sret)
// CHECK: declare void @f3({{.*}} sret)
// CHECK: declare void @f4({{.*}} byval({{.*}}) align 4)
// CHECK: declare void @f5({{.*}} byval({{.*}}) align 4)

// CHECK: attributes [[RN]] = { nounwind readnone{{.*}} }
// CHECK: attributes [[RO]] = { nounwind readonly{{.*}} }
