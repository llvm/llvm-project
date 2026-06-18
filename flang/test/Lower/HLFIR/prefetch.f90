! Test lowering of prefetch directive
! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s --check-prefixes=HLFIR

module test_prefetch_mod
  implicit none
  type :: t
    integer :: a(256, 256)
  end type t
end module test_prefetch_mod

subroutine test_prefetch_01()
  ! HLFIR: %[[H_A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFtest_prefetch_01Ea"} : (!fir.ref<!fir.array<256xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<256xi32>>, !fir.ref<!fir.array<256xi32>>)
  ! HLFIR: %[[H_I:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFtest_prefetch_01Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  ! HLFIR: %[[H_J:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFtest_prefetch_01Ej"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

  integer :: i, j
  integer :: a(256)

  a = 23

  ! HLFIR: fir.prefetch %[[H_A]]#0 {read, data, localityHint = 3 : i32} : !fir.ref<!fir.array<256xi32>>
  !dir$ prefetch a
  i = sum(a)

  ! HLFIR: %[[H_LOAD:.*]] = fir.load %[[H_I]]#0 : !fir.ref<i32>
  ! HLFIR: %[[H_C64:.*]] = arith.constant 64 : i32
  ! HLFIR: %[[H_ADD:.*]] = arith.addi %[[H_LOAD]], %[[H_C64]] overflow<nsw> : i32
  ! HLFIR: %[[H_CON:.*]] = fir.convert %[[H_ADD]] : (i32) -> i64
  ! HLFIR: %[[H_DESIG:.*]] = hlfir.designate %[[H_A]]#0 (%[[H_CON]])  : (!fir.ref<!fir.array<256xi32>>, i64) -> !fir.ref<i32>

  ! HLFIR: fir.prefetch %[[H_DESIG]] {read, data, localityHint = 3 : i32} : !fir.ref<i32>
  ! HLFIR: fir.prefetch %[[H_J]]#0 {read, data, localityHint = 3 : i32} : !fir.ref<i32>

  do i = 1, (256 - 64)
    !dir$ prefetch a(i+64), j
    a(i) = a(i-32) + a(i+32) + j
  end do
end subroutine test_prefetch_01

subroutine test_prefetch_02(t1)
  use test_prefetch_mod
  ! HLFIR: %[[H_A:.*]]:2 = hlfir.declare {{.*}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_prefetch_02Ea"}
  ! HLFIR: %[[H_ARG0:.*]]:2 = hlfir.declare {{.*}} dummy_scope {{.*}} {fortran_attrs = #fir.var_attrs<intent_inout>, uniq_name = "_QFtest_prefetch_02Et1"}
  type(t), intent(inout) :: t1
  integer, allocatable :: a(:, :)

  ! HLFIR: %[[H_DESIG_01:.*]] = hlfir.designate %[[H_ARG0]]#0{"a"}   shape {{.*}}
  ! HLFIR: fir.prefetch %[[H_DESIG_01]] {read, data, localityHint = 3 : i32} : !fir.ref<!fir.array<256x256xi32>>
  !dir$ prefetch t1%a
  a = t1%a ** 2

  do i = 1, 256
    ! HLFIR: %[[A_LOAD:.*]] = fir.load %[[H_A]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
    ! HLFIR: %[[A_BOX:.*]] = fir.box_addr %[[A_LOAD]] : (!fir.box<!fir.heap<!fir.array<?x?xi32>>>) -> !fir.heap<!fir.array<?x?xi32>>
    ! HLFIR: fir.prefetch %[[A_BOX]] {read, data, localityHint = 3 : i32} : !fir.heap<!fir.array<?x?xi32>>
    !dir$ prefetch a
    a(i, :) = a(i, :) + i
    do j = 1, 256
      ! HLFIR: %[[H_DESIG_02:.*]] = hlfir.designate %[[H_ARG0]]#0{"a"} {{.*}}
      ! HLFIR: fir.prefetch %[[H_DESIG_02]] {read, data, localityHint = 3 : i32} : !fir.ref<i32>
      !dir$ prefetch t1%a(i, j)
      t1%a(i, j) = (a(i, j) + i*j) / t1%a(i, j)
    end do
  end do
end subroutine test_prefetch_02

subroutine test_prefetch_03(a)
  integer :: a(:)
  ! HLFIR: %[[BOX:.*]] = fir.box_addr {{.*}} : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
  ! HLFIR: fir.prefetch %[[BOX]] {read, data, localityHint = 3 : i32} : !fir.ref<!fir.array<?xi32>>
  !dir$ prefetch a
  a = sum(a)
end subroutine test_prefetch_03
