! This test checks lowering of OpenACC cache directive.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: acc.private.recipe @privatization_ref_i32 : !fir.ref<i32> init {

! CHECK-LABEL: func.func @_QPtest_cache_basic()
subroutine test_cache_basic()
  integer, parameter :: n = 10
  real, dimension(n) :: a, b
  integer :: i

  !$acc loop
  do i = 1, n
    !$acc cache(b)
    a(i) = b(i)
  end do

! CHECK: acc.loop
! CHECK: %[[CACHE:.*]] = acc.cache varPtr(%{{.*}} : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {{{.*}}name = "b"
! CHECK: hlfir.declare %[[CACHE]](%{{.*}}) {uniq_name = "_QFtest_cache_basicEb"}
end subroutine

! CHECK-LABEL: func.func @_QPtest_cache_readonly()
subroutine test_cache_readonly()
  integer, parameter :: n = 10
  real, dimension(n) :: a, b
  integer :: i

  !$acc loop
  do i = 1, n
    !$acc cache(readonly: b)
    a(i) = b(i)
  end do

! CHECK: acc.loop
! CHECK: %[[CACHE:.*]] = acc.cache varPtr(%{{.*}} : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {dataClause = #acc<data_clause acc_cache_readonly>, name = "b"
! CHECK: hlfir.declare %[[CACHE]](%{{.*}}) {uniq_name = "_QFtest_cache_readonlyEb"}
end subroutine

! CHECK-LABEL: func.func @_QPtest_cache_array_section()
! For b(2:5): lowerbound = 2-1 = 1, extent = 5-2+1 = 4
subroutine test_cache_array_section()
  integer, parameter :: n = 10
  real, dimension(n) :: a, b
  integer :: i

  !$acc loop
  do i = 1, n
    !$acc cache(b(2:5))
    a(i) = b(i)
  end do

! CHECK: acc.loop
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[C2:.*]] = arith.constant 2 : index
! CHECK: %[[C5:.*]] = arith.constant 5 : index
! CHECK: %[[LB:.*]] = arith.subi %[[C2]], %[[C1]] : index
! CHECK: %[[TMP1:.*]] = arith.subi %[[C5]], %[[C1]] : index
! CHECK: %[[TMP2:.*]] = arith.subi %[[TMP1]], %[[LB]] : index
! CHECK: %[[EXT:.*]] = arith.addi %[[TMP2]], %[[C1]] : index
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) extent(%[[EXT]] : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
! CHECK: %[[CACHE:.*]] = acc.cache varPtr(%{{.*}} : !fir.ref<!fir.array<10xf32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<10xf32>> {{{.*}}name = "b
! CHECK: hlfir.declare %[[CACHE]](%{{.*}}) {uniq_name = "_QFtest_cache_array_sectionEb"}
end subroutine

! CHECK-LABEL: func.func @_QPtest_cache_multiple()
subroutine test_cache_multiple()
  integer, parameter :: n = 10
  real, dimension(n) :: a, b, c
  integer :: i

  !$acc loop
  do i = 1, n
    !$acc cache(b, c)
    a(i) = b(i) + c(i)
  end do

! CHECK: acc.loop
! CHECK: %[[CACHE_B:.*]] = acc.cache varPtr(%{{.*}} : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {{{.*}}name = "b"
! CHECK: hlfir.declare %[[CACHE_B]](%{{.*}}) {uniq_name = "_QFtest_cache_multipleEb"}
! CHECK: %[[CACHE_C:.*]] = acc.cache varPtr(%{{.*}} : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {{{.*}}name = "c"
! CHECK: hlfir.declare %[[CACHE_C]](%{{.*}}) {uniq_name = "_QFtest_cache_multipleEc"}
end subroutine

! CHECK-LABEL: func.func @_QPtest_cache_2d_array()
! For b(1:5, 1:5): each dimension has lowerbound = 0, extent = 5
subroutine test_cache_2d_array()
  integer, parameter :: n = 10
  real, dimension(n, n) :: a, b
  integer :: i, j

  !$acc loop
  do i = 1, n
    !$acc cache(b(1:5, 1:5))
    do j = 1, n
      a(i,j) = b(i,j)
    end do
  end do

! CHECK: acc.loop
! Dimension 1: lowerbound = 1-1 = 0, extent = 5-1+1 = 5
! CHECK: arith.constant 1 : index
! CHECK: arith.constant 5 : index
! CHECK: arith.subi
! CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) startIdx(%{{.*}} : index)
! Dimension 2: lowerbound = 1-1 = 0, extent = 5-1+1 = 5
! CHECK: arith.constant 5 : index
! CHECK: arith.subi
! CHECK: %[[BOUND2:.*]] = acc.bounds lowerbound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) startIdx(%{{.*}} : index)
! CHECK: %[[CACHE:.*]] = acc.cache varPtr(%{{.*}} : !fir.ref<!fir.array<10x10xf32>>) bounds(%[[BOUND1]], %[[BOUND2]]) -> !fir.ref<!fir.array<10x10xf32>> {{{.*}}name = "b
! CHECK: hlfir.declare %[[CACHE]](%{{.*}}) {uniq_name = "_QFtest_cache_2d_arrayEb"}
end subroutine
