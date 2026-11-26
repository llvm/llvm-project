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

T1 __attribute__((const)) f8(int, ...);
int __attribute__((const)) f9(int, ...);

void *ps[] = { f0, f1, f2, f3, f4, f5, f6, f7, f8, f9 };

// Check markings for varargs arguments.
//
// FIXME: We can add markings in more cases.
void test(T1 t1, void *p) {
  f8(1);
  f8(1, t1);
  f8(1, p);

  f9(1);
  f9(1, t1);
  f9(1, p);
}

// CHECK: declare i32 @f0() [[RN:#[0-9]+]]
// CHECK: declare i32 @f1() [[RO:#[0-9]+]]
// CHECK: declare void @f2(ptr {{[^,]*}} sret({{[^)]*}}) align 4) [[RNRW:#[0-9]+]]
// CHECK: declare void @f3(ptr {{[^,]*}} sret({{[^)]*}}) align 4) [[RORW:#[0-9]+]]
// CHECK: declare i32 @f4(ptr {{[^,]*}} byval({{[^)]*}}) align 4) [[RNRW]]
// CHECK: declare i32 @f5(ptr {{[^,]*}} byval({{[^)]*}}) align 4) [[RORW]]
// CHECK: declare void @f6(ptr {{[^,]*}} sret({{[^)]*}}) align 4, ptr {{[^,]*}} readnone, i32 {{[^,]*}}) [[RNRW]]
// CHECK: declare void @f7(ptr {{[^,]*}} sret({{[^)]*}}) align 4, ptr {{[^,]*}} readonly, i32 {{[^,]*}}) [[RORW]]
// CHECK: declare void @f8(ptr dead_on_unwind writable sret(%struct.T1) align 4, i32 noundef, ...) [[RNRW]]
// CHECK: declare i32 @f9(i32 noundef, ...) [[RNRW]]


// CHECK: call void (ptr, i32, ...) @f8(ptr dead_on_unwind writable sret(%struct.T1) align 4 {{.*}}, i32 noundef 1) [[RNRW_CALL:#[0-9]+]]
// CHECK: call void (ptr, i32, ...) @f8(ptr dead_on_unwind writable sret(%struct.T1) align 4 {{.*}}, i32 noundef 1, ptr noundef byval(%struct.T1) align 4 {{.*}}) [[RNRW_CALL]]
// CHECK: call void (ptr, i32, ...) @f8(ptr dead_on_unwind writable sret(%struct.T1) align 4 {{.*}}, i32 noundef 1, ptr noundef %0) [[RNRW_CALL]]
// CHECK: call i32 (i32, ...) @f9(i32 noundef 1) [[RN_CALL:#[0-9]+]]
// CHECK: call i32 (i32, ...) @f9(i32 noundef 1, ptr noundef byval(%struct.T1) align 4 {{.*}}) [[RNRW_CALL]]
// CHECK: call i32 (i32, ...) @f9(i32 noundef 1, ptr noundef %1) [[RN_CALL]]


// CHECK: attributes [[RN]] = { nounwind willreturn memory(none){{.*}} }
// CHECK: attributes [[RO]] = { nounwind willreturn memory(read){{.*}} }
// CHECK: attributes [[RNRW]] = { nounwind willreturn memory(argmem: readwrite){{.*}} }
// CHECK: attributes [[RORW]] = { nounwind willreturn memory(read, argmem: readwrite){{.*}} }
// CHECK: attributes [[RNRW_CALL]] = { nounwind willreturn memory(argmem: readwrite) }
// CHECK: attributes [[RN_CALL]] = { nounwind willreturn memory(none) }
