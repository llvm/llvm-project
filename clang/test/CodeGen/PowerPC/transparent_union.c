// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -O2 -target-cpu pwr7 \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-64
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -O2 -target-cpu pwr7 \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-64
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -O2 -target-cpu pwr7 \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-64
// RUN: %clang_cc1 -triple powerpc-unknown-aix -O2 -target-cpu pwr7 \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-AIX-32

typedef union tu_c {
	char a;
	char b;
} tu_c_t __attribute__((transparent_union));

typedef union tu_s {
	short a;
} tu_s_t __attribute__((transparent_union));

typedef union tu_us {
	unsigned short a;
} tu_us_t __attribute__((transparent_union));

typedef union tu_l {
	long a;
} tu_l_t __attribute__((transparent_union));

// CHECK-LABEL: define{{.*}} void @ftest0(
// CHECK-SAME: i8 noundef signext [[UC_COERCE:%.*]]) local_unnamed_addr #[[ATTR0:[0-9]+]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    ret void
void ftest0(tu_c_t uc) { }

// CHECK-LABEL: define{{.*}} void @ftest1(
// CHECK-SAME: i16 noundef signext [[UC_COERCE:%.*]]) local_unnamed_addr #[[ATTR0]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    ret void
void ftest1(tu_s_t uc) { }

// CHECK-LABEL: define{{.*}} void @ftest2(
// CHECK-SAME: i16 noundef zeroext [[UC_COERCE:%.*]]) local_unnamed_addr #[[ATTR0]] {
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    ret void
void ftest2(tu_us_t uc) { }

// CHECK-64-LABEL: define{{.*}} void @ftest3(
// CHECK-64-SAME: i64 [[UC_COERCE:%.*]]) local_unnamed_addr #[[ATTR0]] {
// CHECK-64-NEXT:  [[ENTRY:.*:]]
// CHECK-64-NEXT:    ret void
//
// CHECK-AIX-32-LABEL: define void @ftest3(
// CHECK-AIX-32-SAME: i32 [[UC_COERCE:%.*]]) local_unnamed_addr #[[ATTR0]] {
// CHECK-AIX-32-NEXT:  [[ENTRY:.*:]]
// CHECK-AIX-32-NEXT:    ret void
void ftest3(tu_l_t uc) { }
