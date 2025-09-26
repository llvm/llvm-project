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
int __attribute__((const)) f4(T1 a);
int __attribute__((pure)) f5(T1 a);

// NOTE: The int parameters verifies non-ptr parameters are not a problem
T1 __attribute__((const)) f6(void*, int);
T1 __attribute__((pure)) f7(void*, int);

void *ps[] = { f0, f1, f2, f3, f4, f5, f6, f7 };

// CHECK: declare i32 @f0() [[RN:#[0-9]+]]
// CHECK: declare i32 @f1() [[RO:#[0-9]+]]
// CHECK: declare void @f2(ptr {{[^,]*}} sret({{[^)]*}}) align 4) [[RNRW:#[0-9]+]]
// CHECK: declare void @f3(ptr {{[^,]*}} sret({{[^)]*}}) align 4) [[RORW:#[0-9]+]]
// CHECK: declare i32 @f4(ptr {{[^,]*}} byval({{[^)]*}}) align 4) [[RNRW:#[0-9]+]]
// CHECK: declare i32 @f5(ptr {{[^,]*}} byval({{[^)]*}}) align 4) [[RORW:#[0-9]+]]
// CHECK: declare void @f6(ptr {{[^,]*}} sret({{[^)]*}}) align 4, ptr {{[^,]*}} readnone, i32 {{[^,]*}}) [[RNRW:#[0-9]+]]
// CHECK: declare void @f7(ptr {{[^,]*}} sret({{[^)]*}}) align 4, ptr {{[^,]*}} readonly, i32 {{[^,]*}}) [[RORW:#[0-9]+]]

// CHECK: attributes [[RN]] = { nounwind willreturn memory(none){{.*}} }
// CHECK: attributes [[RO]] = { nounwind willreturn memory(read){{.*}} }
// CHECK: attributes [[RNRW]] = { nounwind willreturn memory(argmem: readwrite){{.*}} }
// CHECK: attributes [[RORW]] = { nounwind willreturn memory(read, argmem: readwrite){{.*}} }
