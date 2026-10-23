subroutine if_then(a, l)
  integer :: a(4)
  logical :: l
  integer :: x
  if (l) x = 1
  a = 11
end subroutine

subroutine if_else(a, b, l)
  integer :: a(4), b(4)
  logical :: l
  integer :: x
  if (l) then
    x = 1
  else
    x = 2
  end if
  a = 11
  b = a + x
end subroutine

subroutine computed_goto(a, k)
  integer :: a(4), k
  goto (10,20), k
  a = 33
  return
10 a = 11
  return
20 a = 22
end subroutine

! Expand assignments explicitly to exercise the O0 pipeline independently of
! which HLFIR expansions are enabled by default.
! RUN: %flang_fc1 -emit-hlfir -mmlir --mlir-print-debuginfo -o - %s | fir-opt --inline-elementals --inline-hlfir-assign --mlir-print-debuginfo -o %t.fir && %flang_fc1 -O0 -debug-info-kind=line-tables-only -emit-llvm %t.fir -o - | FileCheck %s

! Every path to an expanded assignment must go through its entry branch.
! In particular, keeping only the conditional edge is insufficient: the
! unconditional edge from the THEN arm must also go through the preheader.
! CHECK-LABEL: define void @if_then_(
! CHECK: br i1 {{.*}}, label %[[THEN:[0-9]+]], label %[[ENTRY:[0-9]+]]
! CHECK: [[THEN]]:
! CHECK: store i32 1,
! CHECK-NEXT: br label %[[ENTRY]],
! CHECK: [[ENTRY]]:
! CHECK-NEXT: br label %[[HEADER:[0-9]+]], !dbg ![[THEN_LOC:[0-9]+]]
! CHECK: [[HEADER]]:
! CHECK-NEXT: {{.*}} = phi i64 {{.*}}[ 1, %[[ENTRY]] ]

! Both arms must converge before initializing the loop PHIs. Merely finding
! one branch with the assignment location misses a breakpoint on the other arm.
! CHECK-LABEL: define void @if_else_(
! CHECK: br i1 {{.*}}, label %[[THEN:[0-9]+]], label %[[ELSE:[0-9]+]]
! CHECK: [[THEN]]:
! CHECK: store i32 1,
! CHECK-NEXT: br label %[[ENTRY:[0-9]+]],
! CHECK: [[ELSE]]:
! CHECK: store i32 2,
! CHECK-NEXT: br label %[[ENTRY]],
! CHECK: [[ENTRY]]:
! CHECK-NEXT: br label %[[HEADER:[0-9]+]], !dbg ![[ELSE_LOC:[0-9]+]]
! CHECK: [[HEADER]]:
! CHECK-NEXT: {{.*}} = phi i64 {{.*}}[ 1, %[[ENTRY]] ]

! A computed GOTO lowers to cf.switch. Check both case destinations and the
! default destination: none may skip the corresponding assignment preheader.
! CHECK-LABEL: define void @computed_goto_(
! CHECK: switch i64 {{.*}}, label %[[DEFAULT:[0-9]+]] [
! CHECK-NEXT: i64 1, label %[[CASE1:[0-9]+]]
! CHECK-NEXT: i64 2, label %[[CASE2:[0-9]+]]
! CHECK: [[DEFAULT]]:
! CHECK-NEXT: br label %[[HEADER:[0-9]+]], !dbg ![[DEFAULT_LOC:[0-9]+]]
! CHECK: [[HEADER]]:
! CHECK-NEXT: {{.*}} = phi i64 {{.*}}[ 1, %[[DEFAULT]] ]
! CHECK: [[CASE1]]:
! CHECK-NEXT: br label %[[HEADER:[0-9]+]], !dbg ![[CASE1_LOC:[0-9]+]]
! CHECK: [[HEADER]]:
! CHECK-NEXT: {{.*}} = phi i64 {{.*}}[ 1, %[[CASE1]] ]
! CHECK: [[CASE2]]:
! CHECK-NEXT: br label %[[HEADER:[0-9]+]], !dbg ![[CASE2_LOC:[0-9]+]]
! CHECK: [[HEADER]]:
! CHECK-NEXT: {{.*}} = phi i64 {{.*}}[ 1, %[[CASE2]] ]

! CHECK-DAG: ![[THEN_LOC]] = !DILocation(line: 6, column: 3,
! CHECK-DAG: ![[ELSE_LOC]] = !DILocation(line: 18, column: 3,
! CHECK-DAG: ![[DEFAULT_LOC]] = !DILocation(line: 25, column: 3,
! CHECK-DAG: ![[CASE1_LOC]] = !DILocation(line: 27, column: 1,
! CHECK-DAG: ![[CASE2_LOC]] = !DILocation(line: 29, column: 1,
