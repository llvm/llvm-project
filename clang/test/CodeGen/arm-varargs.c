// RUN: %clang_cc1 -triple arm-none-eabi -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple armeb-none-eabi -emit-llvm -o - %s | FileCheck %s

#include <stdarg.h>

// Obviously there's more than one way to implement va_arg. This test should at
// least prevent unintentional regressions caused by refactoring.

va_list the_list;

int simple_int(void) {
// CHECK-LABEL: define{{.*}} i32 @simple_int
  return va_arg(the_list, int);
// CHECK: [[CUR:%[a-z0-9._]+]] = load ptr, ptr @the_list, align 4
// CHECK: [[NEXT:%[a-z0-9._]+]] = getelementptr inbounds i8, ptr [[CUR]], i32 4
// CHECK: store ptr [[NEXT]], ptr @the_list, align 4
// CHECK: [[RESULT:%[a-z0-9._]+]] = load i32, ptr [[CUR]]
// CHECK: ret i32 [[RESULT]]
}

struct bigstruct {
  int a[10];
};

struct bigstruct simple_struct(void) {
// CHECK-LABEL: define{{.*}} void @simple_struct(ptr dead_on_unwind noalias writable sret(%struct.bigstruct) align 4 %agg.result)
  return va_arg(the_list, struct bigstruct);
// CHECK: [[CUR:%[a-z0-9._]+]] = load ptr, ptr @the_list, align 4
// CHECK: [[NEXT:%[a-z0-9._]+]] = getelementptr inbounds i8, ptr [[CUR]], i32 40
// CHECK: store ptr [[NEXT]], ptr @the_list, align 4
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 4 %agg.result, ptr align 4 [[CUR]], i32 40, i1 false)
// CHECK: ret void
}

struct aligned_bigstruct {
  float a;
  long double b;
};

struct aligned_bigstruct simple_aligned_struct(void) {
// CHECK-LABEL: define{{.*}} void @simple_aligned_struct(ptr dead_on_unwind noalias writable sret(%struct.aligned_bigstruct) align 8 %agg.result)
  return va_arg(the_list, struct aligned_bigstruct);
// CHECK: [[CUR:%[a-z0-9._]+]] = load ptr, ptr @the_list, align 4
// CHECK: [[CUR_INT_ADD:%[a-z0-9._]+]] = getelementptr inbounds i8, ptr [[CUR]], i32 7
// CHECK: [[CUR_ALIGNED:%[a-z0-9._]+]] = call ptr @llvm.ptrmask.p0.i32(ptr [[CUR_INT_ADD]], i32 -8)
// CHECK: [[NEXT:%[a-z0-9._]+]] = getelementptr inbounds i8, ptr [[CUR_ALIGNED]], i32 16
// CHECK: store ptr [[NEXT]], ptr @the_list, align 4
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 8 %agg.result, ptr align 8 [[CUR_ALIGNED]], i32 16, i1 false)
// CHECK: ret void
}

double simple_double(void) {
// CHECK-LABEL: define{{.*}} double @simple_double
  return va_arg(the_list, double);
// CHECK: [[CUR:%[a-z0-9._]+]] = load ptr, ptr @the_list, align 4
// CHECK: [[CUR_INT_ADD:%[a-z0-9._]+]] = getelementptr inbounds i8, ptr [[CUR]], i32 7
// CHECK: [[CUR_ALIGNED:%[a-z0-9._]+]] = call ptr @llvm.ptrmask.p0.i32(ptr [[CUR_INT_ADD]], i32 -8)
// CHECK: [[NEXT:%[a-z0-9._]+]] = getelementptr inbounds i8, ptr [[CUR_ALIGNED]], i32 8
// CHECK: store ptr [[NEXT]], ptr @the_list, align 4
// CHECK: [[RESULT:%[a-z0-9._]+]] = load double, ptr [[CUR_ALIGNED]]
// CHECK: ret double [[RESULT]]
}

struct hfa {
  float a, b;
};

struct hfa simple_hfa(void) {
// CHECK-LABEL: define{{.*}} void @simple_hfa(ptr dead_on_unwind noalias writable sret(%struct.hfa) align 4 %agg.result)
  return va_arg(the_list, struct hfa);
// CHECK: [[CUR:%[a-z0-9._]+]] = load ptr, ptr @the_list, align 4
// CHECK: [[NEXT:%[a-z0-9._]+]] = getelementptr inbounds i8, ptr [[CUR]], i32 8
// CHECK: store ptr [[NEXT]], ptr @the_list, align 4
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 4 %agg.result, ptr align 4 [[CUR]], i32 8, i1 false)
// CHECK: ret void
}

// Over and under alignment on fundamental types has no effect on parameter
// passing, so the code generated for va_arg should be the same as for
// non-aligned fundamental types.

typedef int underaligned_int __attribute__((packed,aligned(2)));
underaligned_int underaligned_int_test(void) {
// CHECK-LABEL: define{{.*}} i32 @underaligned_int_test()
  return va_arg(the_list, underaligned_int);
// CHECK: [[CUR:%[a-z0-9._]+]] = load ptr, ptr @the_list, align 4
// CHECK: [[NEXT:%[a-z0-9._]+]] = getelementptr inbounds i8, ptr [[CUR]], i32 4
// CHECK: store ptr [[NEXT]], ptr @the_list, align 4
// CHECK: [[RESULT:%[a-z0-9._]+]] = load i32, ptr [[CUR]]
// CHECK: ret i32 [[RESULT]]
}

typedef int overaligned_int __attribute__((aligned(32)));
overaligned_int overaligned_int_test(void) {
// CHECK-LABEL: define{{.*}} i32 @overaligned_int_test()
  return va_arg(the_list, overaligned_int);
// CHECK: [[CUR:%[a-z0-9._]+]] = load ptr, ptr @the_list, align 4
// CHECK: [[NEXT:%[a-z0-9._]+]] = getelementptr inbounds i8, ptr [[CUR]], i32 4
// CHECK: store ptr [[NEXT]], ptr @the_list, align 4
// CHECK: [[RESULT:%[a-z0-9._]+]] = load i32, ptr [[CUR]]
// CHECK: ret i32 [[RESULT]]
}

typedef long long underaligned_long_long  __attribute__((packed,aligned(2)));
underaligned_long_long underaligned_long_long_test(void) {
// CHECK-LABEL: define{{.*}} i64 @underaligned_long_long_test()
  return va_arg(the_list, underaligned_long_long);
// CHECK: [[CUR:%[a-z0-9._]+]] = load ptr, ptr @the_list, align 4
// CHECK: [[CUR_INT_ADD:%[a-z0-9._]+]] = getelementptr inbounds i8, ptr [[CUR]], i32 7
// CHECK: [[CUR_ALIGNED:%[a-z0-9._]+]] = call ptr @llvm.ptrmask.p0.i32(ptr [[CUR_INT_ADD]], i32 -8)
// CHECK: [[NEXT:%[a-z0-9._]+]] = getelementptr inbounds i8, ptr [[CUR_ALIGNED]], i32 8
// CHECK: store ptr [[NEXT]], ptr @the_list, align 4
// CHECK: [[RESULT:%[a-z0-9._]+]] = load i64, ptr [[CUR_ALIGNED]]
// CHECK: ret i64 [[RESULT]]
}

typedef long long overaligned_long_long  __attribute__((aligned(32)));
overaligned_long_long overaligned_long_long_test(void) {
// CHECK-LABEL: define{{.*}} i64 @overaligned_long_long_test()
  return va_arg(the_list, overaligned_long_long);
// CHECK: [[CUR:%[a-z0-9._]+]] = load ptr, ptr @the_list, align 4
// CHECK: [[CUR_INT_ADD:%[a-z0-9._]+]] = getelementptr inbounds i8, ptr [[CUR]], i32 7
// CHECK: [[CUR_ALIGNED:%[a-z0-9._]+]] = call ptr @llvm.ptrmask.p0.i32(ptr [[CUR_INT_ADD]], i32 -8)
// CHECK: [[NEXT:%[a-z0-9._]+]] = getelementptr inbounds i8, ptr [[CUR_ALIGNED]], i32 8
// CHECK: store ptr [[NEXT]], ptr @the_list, align 4
// CHECK: [[RESULT:%[a-z0-9._]+]] = load i64, ptr [[CUR_ALIGNED]]
// CHECK: ret i64 [[RESULT]]
}

// The way that attributes applied to a struct change parameter passing is a
// little strange, in that the alignment due to attributes is used when
// calculating the size of the struct, but the alignment is based only on the
// alignment of the members (which can be affected by attributes). What this
// means is:
//  * The only effect of the aligned attribute on a struct is to increase its
//    size if the alignment is greater than the member alignment.
//  * The packed attribute is considered as applying to the members, so it will
//    affect the alignment.
// Additionally the alignment can't go below 4 or above 8, so it's only
// long long and double that can be affected by a change in alignment.

typedef struct __attribute__((packed,aligned(2))) {
  int val;
} underaligned_int_struct;
underaligned_int_struct underaligned_int_struct_test(void) {
// CHECK-LABEL: define{{.*}} i32 @underaligned_int_struct_test()
  return va_arg(the_list, underaligned_int_struct);
// CHECK: [[RETVAL:%[a-z0-9._]+]] = alloca %struct.underaligned_int_struct, align 2
// CHECK: [[CUR:%[a-z0-9._]+]] = load ptr, ptr @the_list, align 4
// CHECK: [[NEXT:%[a-z0-9._]+]] = getelementptr inbounds i8, ptr [[CUR]], i32 4
// CHECK: store ptr [[NEXT]], ptr @the_list, align 4
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 2 [[RETVAL]], ptr align 4 [[CUR]], i32 4, i1 false)
// CHECK: [[COERCE:%[a-z0-9._]+]] = getelementptr inbounds %struct.underaligned_int_struct, ptr [[RETVAL]], i32 0, i32 0
// CHECK: [[RESULT:%[a-z0-9._]+]] = load i32, ptr [[COERCE]]
// CHECK: ret i32 [[RESULT]]
}

typedef struct __attribute__((aligned(16))) {
  int val;
} overaligned_int_struct;
overaligned_int_struct overaligned_int_struct_test(void) {
// CHECK-LABEL: define{{.*}} void @overaligned_int_struct_test(ptr dead_on_unwind noalias writable sret(%struct.overaligned_int_struct) align 16 %agg.result)
  return va_arg(the_list, overaligned_int_struct);
// CHECK: [[CUR:%[a-z0-9._]+]] = load ptr, ptr @the_list, align 4
// CHECK: [[NEXT:%[a-z0-9._]+]] = getelementptr inbounds i8, ptr [[CUR]], i32 16
// CHECK: store ptr [[NEXT]], ptr @the_list, align 4
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 16 %agg.result, ptr align 4 [[CUR]], i32 16, i1 false)
// CHECK: ret void
}

typedef struct __attribute__((packed,aligned(2))) {
  long long val;
} underaligned_long_long_struct;
underaligned_long_long_struct underaligned_long_long_struct_test(void) {
// CHECK-LABEL: define{{.*}} void @underaligned_long_long_struct_test(ptr dead_on_unwind noalias writable sret(%struct.underaligned_long_long_struct) align 2 %agg.result)
  return va_arg(the_list, underaligned_long_long_struct);
// CHECK: [[CUR:%[a-z0-9._]+]] = load ptr, ptr @the_list, align 4
// CHECK: [[NEXT:%[a-z0-9._]+]] = getelementptr inbounds i8, ptr [[CUR]], i32 8
// CHECK: store ptr [[NEXT]], ptr @the_list, align 4
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 2 %agg.result, ptr align 4 [[CUR]], i32 8, i1 false)
// CHECK: ret void
}

typedef struct __attribute__((aligned(16))) {
  long long val;
} overaligned_long_long_struct;
overaligned_long_long_struct overaligned_long_long_struct_test(void) {
// CHECK-LABEL: define{{.*}} void @overaligned_long_long_struct_test(ptr dead_on_unwind noalias writable sret(%struct.overaligned_long_long_struct) align 16 %agg.result)
  return va_arg(the_list, overaligned_long_long_struct);
// CHECK: [[CUR:%[a-z0-9._]+]] = load ptr, ptr @the_list, align 4
// CHECK: [[CUR_INT_ADD:%[a-z0-9._]+]] = getelementptr inbounds i8, ptr [[CUR]], i32 7
// CHECK: [[CUR_ALIGNED:%[a-z0-9._]+]] = call ptr @llvm.ptrmask.p0.i32(ptr [[CUR_INT_ADD]], i32 -8)
// CHECK: [[NEXT:%[a-z0-9._]+]] = getelementptr inbounds i8, ptr [[CUR_ALIGNED]], i32 16
// CHECK: store ptr [[NEXT]], ptr @the_list, align 4
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 16 %agg.result, ptr align 8 [[CUR_ALIGNED]], i32 16, i1 false)
// CHECK: ret void
}

// Overaligning or underaligning a struct member changes both its alignment and
// size when passed as an argument.

typedef struct {
  int val __attribute__((packed,aligned(2)));
} underaligned_int_struct_member;
underaligned_int_struct_member underaligned_int_struct_member_test(void) {
// CHECK-LABEL: define{{.*}} i32 @underaligned_int_struct_member_test()
  return va_arg(the_list, underaligned_int_struct_member);
// CHECK: [[RETVAL:%[a-z0-9._]+]] = alloca %struct.underaligned_int_struct_member, align 2
// CHECK: [[CUR:%[a-z0-9._]+]] = load ptr, ptr @the_list, align 4
// CHECK: [[NEXT:%[a-z0-9._]+]] = getelementptr inbounds i8, ptr [[CUR]], i32 4
// CHECK: store ptr [[NEXT]], ptr @the_list, align 4
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 2 [[RETVAL]], ptr align 4 [[CUR]], i32 4, i1 false)
// CHECK: [[COERCE:%[a-z0-9._]+]] = getelementptr inbounds %struct.underaligned_int_struct_member, ptr [[RETVAL]], i32 0, i32 0
// CHECK: [[RESULT:%[a-z0-9._]+]] = load i32, ptr [[COERCE]]
// CHECK: ret i32 [[RESULT]]
}

typedef struct {
  int val __attribute__((aligned(16)));
} overaligned_int_struct_member;
overaligned_int_struct_member overaligned_int_struct_member_test(void) {
// CHECK-LABEL: define{{.*}} void @overaligned_int_struct_member_test(ptr dead_on_unwind noalias writable sret(%struct.overaligned_int_struct_member) align 16 %agg.result)
  return va_arg(the_list, overaligned_int_struct_member);
// CHECK: [[CUR:%[a-z0-9._]+]] = load ptr, ptr @the_list, align 4
// CHECK: [[CUR_INT_ADD:%[a-z0-9._]+]] = getelementptr inbounds i8, ptr [[CUR]], i32 7
// CHECK: [[CUR_ALIGNED:%[a-z0-9._]+]] = call ptr @llvm.ptrmask.p0.i32(ptr [[CUR_INT_ADD]], i32 -8)
// CHECK: [[NEXT:%[a-z0-9._]+]] = getelementptr inbounds i8, ptr [[CUR_ALIGNED]], i32 16
// CHECK: store ptr [[NEXT]], ptr @the_list, align 4
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 16 %agg.result, ptr align 8 [[CUR_ALIGNED]], i32 16, i1 false)
// CHECK: ret void
}

typedef struct {
  long long val __attribute__((packed,aligned(2)));
} underaligned_long_long_struct_member;
underaligned_long_long_struct_member underaligned_long_long_struct_member_test(void) {
// CHECK-LABEL: define{{.*}} void @underaligned_long_long_struct_member_test(ptr dead_on_unwind noalias writable sret(%struct.underaligned_long_long_struct_member) align 2 %agg.result)
  return va_arg(the_list, underaligned_long_long_struct_member);
// CHECK: [[CUR:%[a-z0-9._]+]] = load ptr, ptr @the_list, align 4
// CHECK: [[NEXT:%[a-z0-9._]+]] = getelementptr inbounds i8, ptr [[CUR]], i32 8
// CHECK: store ptr [[NEXT]], ptr @the_list, align 4
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 2 %agg.result, ptr align 4 [[CUR]], i32 8, i1 false)
// CHECK: ret void
}

typedef struct {
  long long val __attribute__((aligned(16)));
} overaligned_long_long_struct_member;
overaligned_long_long_struct_member overaligned_long_long_struct_member_test(void) {
// CHECK-LABEL: define{{.*}} void @overaligned_long_long_struct_member_test(ptr dead_on_unwind noalias writable sret(%struct.overaligned_long_long_struct_member) align 16 %agg.result)
  return va_arg(the_list, overaligned_long_long_struct_member);
// CHECK: [[CUR:%[a-z0-9._]+]] = load ptr, ptr @the_list, align 4
// CHECK: [[CUR_INT_ADD:%[a-z0-9._]+]] = getelementptr inbounds i8, ptr [[CUR]], i32 7
// CHECK: [[CUR_ALIGNED:%[a-z0-9._]+]] = call ptr @llvm.ptrmask.p0.i32(ptr [[CUR_INT_ADD]], i32 -8)
// CHECK: [[NEXT:%[a-z0-9._]+]] = getelementptr inbounds i8, ptr [[CUR_ALIGNED]], i32 16
// CHECK: store ptr [[NEXT]], ptr @the_list, align 4
// CHECK: call void @llvm.memcpy.p0.p0.i32(ptr align 16 %agg.result, ptr align 8 [[CUR_ALIGNED]], i32 16, i1 false)
// CHECK: ret void
}

void check_start(int n, ...) {
// CHECK-LABEL: define{{.*}} void @check_start(i32 noundef %n, ...)

  va_list the_list;
  va_start(the_list, n);
// CHECK: [[THE_LIST:%[a-z0-9._]+]] = alloca %struct.__va_list
// CHECK: call void @llvm.va_start.p0(ptr [[THE_LIST]])
}
