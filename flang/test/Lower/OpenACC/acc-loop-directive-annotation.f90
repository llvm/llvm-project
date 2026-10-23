! Test that !dir$ loop directives are applied as loopAnnotation on acc.loop,
! including combined constructs and loops inside a compute construct.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPacc_loop_unroll
! CHECK: acc.loop {{.*}} {
! CHECK: } {{.*}}attributes {{{.*}}llvm.loop_annotation = #llvm.loop_annotation<unroll = <disable = false, full = true>>}

subroutine acc_loop_unroll(a, n)
  real :: a(n)
  integer :: i, n
  !$acc loop
  !dir$ unroll
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

! CHECK-LABEL: func.func @_QPunroll_acc_loop
! CHECK: acc.loop {{.*}} {
! CHECK: } {{.*}}attributes {{{.*}}llvm.loop_annotation = #llvm.loop_annotation<unroll = <disable = false, full = true>>}

subroutine unroll_acc_loop(a, n)
  real :: a(n)
  integer :: i, n
  !dir$ unroll
  !$acc loop
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_loop_unroll_count
! CHECK: acc.loop {{.*}} {
! CHECK: } {{.*}}attributes {{{.*}}llvm.loop_annotation = #llvm.loop_annotation<unroll = <disable = false, count = 4 : i64>>}

subroutine acc_loop_unroll_count(a, n)
  real :: a(n)
  integer :: i, n
  !$acc loop
  !dir$ unroll 4
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

! CHECK-LABEL: func.func @_QPunroll_count_acc_loop
! CHECK: acc.loop {{.*}} {
! CHECK: } {{.*}}attributes {{{.*}}llvm.loop_annotation = #llvm.loop_annotation<unroll = <disable = false, count = 4 : i64>>}

subroutine unroll_count_acc_loop(a, n)
  real :: a(n)
  integer :: i, n
  !dir$ unroll 4
  !$acc loop
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_parallel_loop_unroll
! CHECK: acc.loop {{.*}} {
! CHECK: } {{.*}}attributes {{{.*}}llvm.loop_annotation = #llvm.loop_annotation<unroll = <disable = false, full = true>>}

subroutine acc_parallel_loop_unroll(a, n)
  real :: a(n)
  integer :: i, n
  !$acc parallel loop
  !dir$ unroll
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

! CHECK-LABEL: func.func @_QPunroll_acc_parallel_loop
! CHECK: acc.loop {{.*}} {
! CHECK: } {{.*}}attributes {{{.*}}llvm.loop_annotation = #llvm.loop_annotation<unroll = <disable = false, full = true>>}

subroutine unroll_acc_parallel_loop(a, n)
  real :: a(n)
  integer :: i, n
  !dir$ unroll
  !$acc parallel loop
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_parallel_loop_unroll_count
! CHECK: acc.loop {{.*}} {
! CHECK: } {{.*}}attributes {{{.*}}llvm.loop_annotation = #llvm.loop_annotation<unroll = <disable = false, count = 4 : i64>>}

subroutine acc_parallel_loop_unroll_count(a, n)
  real :: a(n)
  integer :: i, n
  !$acc parallel loop
  !dir$ unroll 4
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

! CHECK-LABEL: func.func @_QPunroll_count_acc_parallel_loop
! CHECK: acc.loop {{.*}} {
! CHECK: } {{.*}}attributes {{{.*}}llvm.loop_annotation = #llvm.loop_annotation<unroll = <disable = false, count = 4 : i64>>}

subroutine unroll_count_acc_parallel_loop(a, n)
  real :: a(n)
  integer :: i, n
  !dir$ unroll 4
  !$acc parallel loop
  do i = 1, n
    a(i) = a(i) + 1
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_parallel_unroll
! CHECK: acc.parallel {
! CHECK: acc.loop {{.*}} {
! CHECK: } {{.*}}attributes {{{.*}}llvm.loop_annotation = #llvm.loop_annotation<unroll = <disable = false, full = true>>}

subroutine acc_parallel_unroll(a, n)
  real :: a(n)
  integer :: i, n
  !$acc parallel
  !dir$ unroll
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPunroll_acc_parallel
! CHECK: acc.parallel {
! CHECK: acc.loop {{.*}} {
! CHECK: } {{.*}}attributes {{{.*}}llvm.loop_annotation = #llvm.loop_annotation<unroll = <disable = false, full = true>>}

subroutine unroll_acc_parallel(a, n)
  real :: a(n)
  integer :: i, n
  !dir$ unroll
  !$acc parallel
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPacc_parallel_unroll_count
! CHECK: acc.parallel {
! CHECK: acc.loop {{.*}} {
! CHECK: } {{.*}}attributes {{{.*}}llvm.loop_annotation = #llvm.loop_annotation<unroll = <disable = false, count = 4 : i64>>}

subroutine acc_parallel_unroll_count(a, n)
  real :: a(n)
  integer :: i, n
  !$acc parallel
  !dir$ unroll 4
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPunroll_count_acc_parallel
! CHECK: acc.parallel {
! CHECK: acc.loop {{.*}} {
! CHECK: } {{.*}}attributes {{{.*}}llvm.loop_annotation = #llvm.loop_annotation<unroll = <disable = false, count = 4 : i64>>}

subroutine unroll_count_acc_parallel(a, n)
  real :: a(n)
  integer :: i, n
  !dir$ unroll 4
  !$acc parallel
  do i = 1, n
    a(i) = a(i) + 1
  end do
  !$acc end parallel
end subroutine