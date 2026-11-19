subroutine arrays(a, b, c)
  integer :: a(4), b(4), c(4)
  b = 1
  c = 2
  a = 11
  a = b + c
end subroutine

! Expand the assignments explicitly, without enabling other optimizations.
! Each following assignment must retain a separate entry branch carrying its
! source location, outside both the preceding and following loops.
! RUN: %flang_fc1 -emit-hlfir -mmlir --mlir-print-debuginfo -o - %s | fir-opt --inline-elementals --inline-hlfir-assign --mlir-print-debuginfo -o %t.fir && %flang_fc1 -O0 -debug-info-kind=line-tables-only -emit-llvm %t.fir -o - | FileCheck %s

! CHECK-LABEL: define void @arrays_(
! CHECK: br label %[[FIRST:[0-9]+]], !dbg ![[LOC1:[0-9]+]]
! CHECK: [[FIRST]]:
! CHECK: br i1 {{.*}}, label %{{[0-9]+}}, label %[[ENTRY2:[0-9]+]], !dbg ![[LOC1]]
! CHECK: [[ENTRY2]]:
! CHECK-NEXT: br label %[[SECOND:[0-9]+]], !dbg ![[LOC2:[0-9]+]]
! CHECK: [[SECOND]]:
! CHECK-NEXT: {{.*}} = phi i64 {{.*}}[ {{[01]}}, %[[ENTRY2]] ]
! CHECK: br i1 {{.*}}, label %{{[0-9]+}}, label %[[ENTRY3:[0-9]+]], !dbg ![[LOC2]]
! CHECK: [[ENTRY3]]:
! CHECK-NEXT: br label %[[THIRD:[0-9]+]], !dbg ![[LOC3:[0-9]+]]
! CHECK: [[THIRD]]:
! CHECK-NEXT: {{.*}} = phi i64 {{.*}}[ {{[01]}}, %[[ENTRY3]] ]
! CHECK: br i1 {{.*}}, label %{{[0-9]+}}, label %[[ENTRY4:[0-9]+]], !dbg ![[LOC3]]
! CHECK: [[ENTRY4]]:
! CHECK-NEXT: br label %[[FOURTH:[0-9]+]], !dbg ![[LOC4:[0-9]+]]
! CHECK: [[FOURTH]]:
! CHECK-NEXT: {{.*}} = phi i64 {{.*}}[ {{[01]}}, %[[ENTRY4]] ]
! CHECK: br i1 {{.*}}, !dbg ![[LOC4]]
! CHECK-DAG: ![[LOC1]] = !DILocation(line: 3, column: 3,
! CHECK-DAG: ![[LOC2]] = !DILocation(line: 4, column: 3,
! CHECK-DAG: ![[LOC3]] = !DILocation(line: 5, column: 3,
! CHECK-DAG: ![[LOC4]] = !DILocation(line: 6, column: 3,
