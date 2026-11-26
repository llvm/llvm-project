// REQUIRES: bpf-registered-target
// RUN: %clang_cc1 -triple bpf -emit-llvm -debug-info-kind=limited -disable-llvm-passes \
// RUN:   %s -o - | FileCheck %s
// RUN: not %clang_cc1 -triple bpf -emit-llvm -disable-llvm-passes %s -o /dev/null 2>&1 \
// RUN:   | FileCheck --check-prefix=NODBG %s

// An explicit __builtin_preserve_enum_value() call in a case label is relocated
// and compared before the switch, like an enumerator of a preserve_access_index
// enum. The existence check guards the relocated value, so it is not reached
// if the enumerator does not exist. Without debug info, the call is an error as
// elsewhere.

#define bpf_core_enum_value_exists(enum_type, enum_value)                      \
  __builtin_preserve_enum_value(*(typeof(enum_type) *)enum_value, 0)
#define bpf_core_enum_value(enum_type, enum_value)                             \
  __builtin_preserve_enum_value(*(typeof(enum_type) *)enum_value, 1)

enum cgroup_subsys_id { cpu_cgrp_id, memory_cgrp_id = 4, hugetlb_cgrp_id = 9 };

int test(int id) {
  switch (id) {
  case 1:
    return 1;
  case bpf_core_enum_value_exists(enum cgroup_subsys_id, hugetlb_cgrp_id)
      ? bpf_core_enum_value(enum cgroup_subsys_id, hugetlb_cgrp_id)
      : -1:
    return 2;
  case bpf_core_enum_value(enum cgroup_subsys_id, memory_cgrp_id) ... 6:
    return 3;
  }
  return 0;
}

// CHECK-LABEL: define dso_local i32 @test
// CHECK: [[X:%[0-9]+]] = load i32, ptr %id.addr
// CHECK-NEXT: [[E:%[0-9]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 0), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[ENUM:[0-9]+]]
// CHECK-NEXT: [[B:%[0-9a-z]+]] = icmp ne i64 [[E]], 0
// CHECK-NEXT: br i1 [[B]], label %[[TRUE:[0-9a-z.]+]], label %[[FALSE:[0-9a-z.]+]]
// CHECK: [[TRUE]]:
// CHECK-NEXT: [[V:%[0-9]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[ENUM]]
// CHECK: [[P:%[0-9a-z]+]] = phi i64 [ [[V]], %[[TRUE]] ], [ -1, %[[FALSE]] ]
// CHECK-NEXT: [[T:%[0-9]+]] = trunc i64 [[P]] to i32
// CHECK-NEXT: [[C:%[0-9]+]] = icmp eq i32 [[X]], [[T]]
// CHECK-NEXT: br i1 [[C]], label %[[BB_H:[0-9a-z.]+]], label %[[D1:[0-9a-z.]+]]
// CHECK: [[D1]]:
// CHECK-NEXT: [[V:%[0-9]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[ENUM]]
// CHECK-NEXT: [[T:%[0-9]+]] = trunc i64 [[V]] to i32
// CHECK-NEXT: [[LO:%[0-9]+]] = icmp sge i32 [[X]], [[T]]
// CHECK-NEXT: [[HI:%[0-9]+]] = icmp sle i32 [[X]], 6
// CHECK-NEXT: [[C:%[0-9]+]] = and i1 [[LO]], [[HI]]
// CHECK-NEXT: br i1 [[C]], label %[[BB_M:[0-9a-z.]+]], label %[[D2:[0-9a-z.]+]]
// CHECK: [[D2]]:
// CHECK-NEXT: switch i32 [[X]], label %sw.epilog [
// CHECK-NEXT: i32 1, label %sw.bb
// CHECK-NEXT: ]
// CHECK: [[BB_H]]:
// CHECK-NEXT: store i32 2, ptr %retval
// CHECK: [[BB_M]]:
// CHECK-NEXT: store i32 3, ptr %retval

// NODBG: error: using builtin function without -g
