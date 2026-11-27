// REQUIRES: bpf-registered-target
// RUN: %clang_cc1 -triple bpf -emit-llvm -debug-info-kind=limited -disable-llvm-passes \
// RUN:   %s -o - | FileCheck --check-prefixes=CHECK,LOCAL %s
// RUN: %clang_cc1 -triple bpf -emit-llvm -disable-llvm-passes %s -o - \
// RUN:   | FileCheck --check-prefixes=NODBG,LOCAL %s
// RUN: %clang_cc1 -triple bpf -emit-llvm -debug-info-kind=limited -disable-llvm-passes \
// RUN:   -fbpf-no-enum-reloc=e1,e2,e3 -DNO_ENUM_RELOC %s -o - \
// RUN:   | FileCheck --check-prefix=CONST %s

// A case value with an enumerator of a preserve_access_index enum is relocated
// and compared before the switch. Without debug info, e.g., in a nodebug
// function, its local value is compared before the switch. With
// -fbpf-no-enum-reloc=, the case values are constants in the switch.

#define __reloc__ __attribute__((preserve_access_index))

enum e1 { E1_A = 1, E1_B = 4, E1_C = 8 } __reloc__;
typedef enum : signed char { E2_A = 3 } __reloc__ e2;
enum e3 : long { E3_A = 5 } __reloc__;

void h(int);

int test1(int x) {
  switch (x) {
  case 2:
    return 1;
  case E1_A:
    return 2;
  case E1_B ... E1_C:
    return 3;
  case E2_A + 100:
    return 4;
  case 10 ... E3_A + 10:
    return 5;
  default:
    return 6;
  }
}

// CHECK-LABEL: define dso_local i32 @test1
// CHECK: [[X:%[0-9]+]] = load i32, ptr %x.addr
// CHECK-NEXT: [[V:%[0-9]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E1:[0-9]+]]
// CHECK-NEXT: [[T:%[0-9a-z]+]] = trunc i64 [[V]] to i32
// CHECK-NEXT: [[C:%[0-9]+]] = icmp eq i32 [[X]], [[T]]
// CHECK-NEXT: br i1 [[C]], label %[[BB_A:[0-9a-z.]+]], label %[[D1:[0-9a-z.]+]]
// CHECK: [[D1]]:
// CHECK-NEXT: [[V:%[0-9]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E1]]
// CHECK-NEXT: [[T:%[0-9a-z]+]] = trunc i64 [[V]] to i32
// CHECK-NEXT: [[LO:%[0-9]+]] = icmp sge i32 [[X]], [[T]]
// CHECK-NEXT: [[V:%[0-9]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E1]]
// CHECK-NEXT: [[T:%[0-9a-z]+]] = trunc i64 [[V]] to i32
// CHECK-NEXT: [[HI:%[0-9]+]] = icmp sle i32 [[X]], [[T]]
// CHECK-NEXT: [[C:%[0-9]+]] = and i1 [[LO]], [[HI]]
// CHECK-NEXT: br i1 [[C]], label %[[BB_BC:[0-9a-z.]+]], label %[[D2:[0-9a-z.]+]]
// CHECK: [[D2]]:
// CHECK-NEXT: [[V:%[0-9]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E2:[0-9]+]]
// CHECK-NEXT: [[T:%[0-9a-z]+]] = trunc i64 [[V]] to i8
// CHECK-NEXT: [[S:%[0-9a-z]+]] = sext i8 [[T]] to i32
// CHECK-NEXT: [[ADD:%[0-9a-z]+]] = add nsw i32 [[S]], 100
// CHECK-NEXT: [[C:%[0-9]+]] = icmp eq i32 [[X]], [[ADD]]
// CHECK-NEXT: br i1 [[C]], label %[[BB_A2:[0-9a-z.]+]], label %[[D3:[0-9a-z.]+]]
// CHECK: [[D3]]:
// CHECK-NEXT: [[LO:%[0-9]+]] = icmp sge i32 [[X]], 10
// CHECK-NEXT: [[V:%[0-9]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E3:[0-9]+]]
// CHECK-NEXT: [[ADD:%[0-9a-z]+]] = add nsw i64 [[V]], 10
// CHECK-NEXT: [[T:%[0-9a-z]+]] = trunc i64 [[ADD]] to i32
// CHECK-NEXT: [[HI:%[0-9]+]] = icmp sle i32 [[X]], [[T]]
// CHECK-NEXT: [[C:%[0-9]+]] = and i1 [[LO]], [[HI]]
// CHECK-NEXT: br i1 [[C]], label %[[BB_R:[0-9a-z.]+]], label %[[D4:[0-9a-z.]+]]
// CHECK: [[D4]]:
// CHECK-NEXT: switch i32 [[X]], label %sw.default [
// CHECK-NEXT: i32 2, label %sw.bb
// CHECK-NEXT: ]
// CHECK: [[BB_A]]:
// CHECK-NEXT: store i32 2, ptr %retval
// CHECK: [[BB_BC]]:
// CHECK-NEXT: store i32 3, ptr %retval
// CHECK: [[BB_A2]]:
// CHECK-NEXT: store i32 4, ptr %retval
// CHECK: [[BB_R]]:
// CHECK-NEXT: store i32 5, ptr %retval

// CONST-LABEL: define dso_local i32 @test1
// CONST-NOT: @llvm.bpf.preserve.enum.value
// CONST: switch i32 %{{[0-9]+}}, label %sw.default [
// CONST-NEXT: i32 2, label %sw.bb
// CONST-NEXT: i32 1, label %sw.bb1
// CONST-NEXT: i32 4, label %sw.bb2
// CONST-NEXT: i32 5, label %sw.bb2
// CONST-NEXT: i32 6, label %sw.bb2
// CONST-NEXT: i32 7, label %sw.bb2
// CONST-NEXT: i32 8, label %sw.bb2
// CONST-NEXT: i32 103, label %sw.bb3
// CONST-NEXT: i32 10, label %sw.bb4
// CONST-NEXT: i32 11, label %sw.bb4
// CONST-NEXT: i32 12, label %sw.bb4
// CONST-NEXT: i32 13, label %sw.bb4
// CONST-NEXT: i32 14, label %sw.bb4
// CONST-NEXT: i32 15, label %sw.bb4
// CONST-NEXT: ]

int test2(int x) {
  switch (4) {
  case E1_B:
    switch (x) {
    case E3_A:
      h(1);
    }
    return 1;
  case 6:
    return 2;
  }
  return 0;
}

// CHECK-LABEL: define dso_local i32 @test2
// CHECK: [[V:%[0-9]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E1]]
// CHECK-NEXT: [[T:%[0-9a-z]+]] = trunc i64 [[V]] to i32
// CHECK-NEXT: [[C:%[0-9]+]] = icmp eq i32 4, [[T]]
// CHECK-NEXT: br i1 [[C]], label %[[BB_B:[0-9a-z.]+]], label %[[D1:[0-9a-z.]+]]
// CHECK: [[D1]]:
// CHECK-NEXT: switch i32 4, label %{{[0-9a-z.]+}} [
// CHECK-NEXT: i32 6, label %{{[0-9a-z.]+}}
// CHECK-NEXT: ]
// CHECK: [[BB_B]]:
// CHECK-NEXT: [[X:%[0-9]+]] = load i32, ptr %x.addr
// CHECK-NEXT: [[V:%[0-9]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E3]]
// CHECK-NEXT: [[T:%[0-9a-z]+]] = trunc i64 [[V]] to i32
// CHECK-NEXT: [[C:%[0-9]+]] = icmp eq i32 [[X]], [[T]]
// CHECK-NEXT: br i1 [[C]], label %{{[0-9a-z.]+}}, label %[[D2:[0-9a-z.]+]]
// CHECK: [[D2]]:
// CHECK-NEXT: switch i32 [[X]], label %{{[0-9a-z.]+}} [
// CHECK-NEXT: ]

// CONST-LABEL: define dso_local i32 @test2
// CONST-NOT: @llvm.bpf.preserve.enum.value
// CONST: switch i32 %{{[0-9]+}}, label %sw.epilog [
// CONST-NEXT: i32 5, label %sw.bb
// CONST-NEXT: ]
// CONST-NOT: switch
// CONST: ret i32 1

void test3(long x) {
  switch (x) {
  case 1:
  case E2_A:
  case 9:
    h(x);
  }
}

// CHECK-LABEL: define dso_local void @test3
// CHECK: [[X:%[0-9]+]] = load i64, ptr %x.addr
// CHECK-NEXT: [[V:%[0-9]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E2]]
// CHECK-NEXT: [[T:%[0-9a-z]+]] = trunc i64 [[V]] to i8
// CHECK-NEXT: [[S:%[0-9]+]] = sext i8 [[T]] to i64
// CHECK-NEXT: [[C:%[0-9]+]] = icmp eq i64 [[X]], [[S]]
// CHECK-NEXT: br i1 [[C]], label %[[BB_C:[0-9a-z.]+]], label %[[D1:[0-9a-z.]+]]
// CHECK: [[D1]]:
// CHECK-NEXT: switch i64 [[X]], label %sw.epilog [
// CHECK-NEXT: i64 1, label %[[BB_1:[0-9a-z.]+]]
// CHECK-NEXT: i64 9, label %[[BB_9:[0-9a-z.]+]]
// CHECK-NEXT: ]
// CHECK: [[BB_1]]:
// CHECK-NEXT: br label %[[BB_C]]
// CHECK: [[BB_C]]:
// CHECK-NEXT: br label %[[BB_9]]
// CHECK: [[BB_9]]:
// CHECK: call void @h

// CONST-LABEL: define dso_local void @test3
// CONST: switch i64 %{{[0-9]+}}, label %sw.epilog [
// CONST-NEXT: i64 1, label %sw.bb
// CONST-NEXT: i64 3, label %sw.bb
// CONST-NEXT: i64 9, label %sw.bb
// CONST-NEXT: ]

__attribute__((nodebug)) int test4(int x) {
  switch (x) {
  case E1_A:
    return 1;
  }
  return 0;
}

// LOCAL-LABEL: define dso_local i32 @test4
// LOCAL-NOT: @llvm.bpf.preserve.enum.value
// LOCAL: [[X:%[0-9]+]] = load i32, ptr %x.addr
// LOCAL-NEXT: [[C:%[0-9]+]] = icmp eq i32 [[X]], 1
// LOCAL-NEXT: br i1 [[C]], label %{{[0-9a-z.]+}}, label %[[D1:[0-9a-z.]+]]
// LOCAL: [[D1]]:
// LOCAL-NEXT: switch i32 [[X]], label %sw.epilog [
// LOCAL-NEXT: ]

#ifndef NO_ENUM_RELOC
// A relocated case value is not checked for duplicates or overlaps, and is not
// added to the switch. The first relocated case that matches, in source order,
// is taken before the other cases.
int test5(int x) {
  switch (x) {
  case E1_A:
    return 1;
  case 1:
    return 2;
  case E1_A:
    return 3;
  case E1_B ... E1_C:
    return 4;
  case 5:
    return 5;
  }
  return 0;
}
#endif

// CHECK-LABEL: define dso_local i32 @test5
// CHECK: [[X:%[0-9]+]] = load i32, ptr %x.addr
// CHECK-NEXT: [[V:%[0-9]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E1]]
// CHECK-NEXT: [[T:%[0-9a-z]+]] = trunc i64 [[V]] to i32
// CHECK-NEXT: [[C:%[0-9]+]] = icmp eq i32 [[X]], [[T]]
// CHECK-NEXT: br i1 [[C]], label %[[BB_A1:[0-9a-z.]+]], label %[[D1:[0-9a-z.]+]]
// CHECK: [[D1]]:
// CHECK-NEXT: [[V:%[0-9]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[E1]]
// CHECK-NEXT: [[T:%[0-9a-z]+]] = trunc i64 [[V]] to i32
// CHECK-NEXT: [[C:%[0-9]+]] = icmp eq i32 [[X]], [[T]]
// CHECK-NEXT: br i1 [[C]], label %[[BB_A2:[0-9a-z.]+]], label %[[D2:[0-9a-z.]+]]
// CHECK: [[D2]]:
// CHECK: [[C:%[0-9]+]] = and i1
// CHECK-NEXT: br i1 [[C]], label %[[BB_BC:[0-9a-z.]+]], label %[[D3:[0-9a-z.]+]]
// CHECK: [[D3]]:
// CHECK-NEXT: switch i32 [[X]], label %sw.epilog [
// CHECK-NEXT: i32 1, label %[[BB_1:[0-9a-z.]+]]
// CHECK-NEXT: i32 5, label %[[BB_5:[0-9a-z.]+]]
// CHECK-NEXT: ]
// CHECK: [[BB_A1]]:
// CHECK-NEXT: store i32 1, ptr %retval
// CHECK: [[BB_1]]:
// CHECK-NEXT: store i32 2, ptr %retval
// CHECK: [[BB_A2]]:
// CHECK-NEXT: store i32 3, ptr %retval
// CHECK: [[BB_BC]]:
// CHECK-NEXT: store i32 4, ptr %retval
// CHECK: [[BB_5]]:
// CHECK-NEXT: store i32 5, ptr %retval

// NODBG-LABEL: define dso_local i32 @test5
// NODBG: [[X:%[0-9]+]] = load i32, ptr %x.addr
// NODBG-NEXT: [[C:%[0-9]+]] = icmp eq i32 [[X]], 1
// NODBG-NEXT: br i1 [[C]], label %[[BB_A1:[0-9a-z.]+]], label %[[D1:[0-9a-z.]+]]
// NODBG: [[D1]]:
// NODBG-NEXT: [[C:%[0-9]+]] = icmp eq i32 [[X]], 1
// NODBG-NEXT: br i1 [[C]], label %[[BB_A2:[0-9a-z.]+]], label %[[D2:[0-9a-z.]+]]
// NODBG: [[D2]]:
// NODBG-NEXT: [[LO:%[0-9]+]] = icmp sge i32 [[X]], 4
// NODBG-NEXT: [[HI:%[0-9]+]] = icmp sle i32 [[X]], 8
// NODBG-NEXT: [[C:%[0-9]+]] = and i1 [[LO]], [[HI]]
// NODBG-NEXT: br i1 [[C]], label %[[BB_BC:[0-9a-z.]+]], label %[[D3:[0-9a-z.]+]]
// NODBG: [[D3]]:
// NODBG-NEXT: switch i32 [[X]], label %sw.epilog [
// NODBG-NEXT: i32 1, label %[[BB_1:[0-9a-z.]+]]
// NODBG-NEXT: i32 5, label %[[BB_5:[0-9a-z.]+]]
// NODBG-NEXT: ]
// NODBG: [[BB_A1]]:
// NODBG-NEXT: store i32 1, ptr %retval
// NODBG: [[BB_1]]:
// NODBG-NEXT: store i32 2, ptr %retval
// NODBG: [[BB_A2]]:
// NODBG-NEXT: store i32 3, ptr %retval
// NODBG: [[BB_BC]]:
// NODBG-NEXT: store i32 4, ptr %retval
// NODBG: [[BB_5]]:
// NODBG-NEXT: store i32 5, ptr %retval

// CHECK: ![[E1]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "e1"
// CHECK: ![[E3]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "e3"
// CHECK: ![[E2]] = !DIDerivedType(tag: DW_TAG_typedef, name: "e2"
