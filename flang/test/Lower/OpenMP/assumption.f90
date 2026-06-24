! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 -o - %s 2>&1 | FileCheck %s

! Lowering of the OpenMP ASSUME construct and ASSUMES directive. Most assumption
! clauses are optimization hints with no representation in the OpenMP dialect, so
! the ASSUME construct lowers its associated structured block and the ASSUMES
! directive lowers as a no-op. The HOLDS clause asserts a scalar logical
! expression, which is lowered to an `llvm.intr.assume` intrinsic.

! CHECK-LABEL: func.func @_QPassumes_decl
subroutine assumes_decl(x)
  !$omp assumes no_openmp
  integer :: x
  ! CHECK: %[[DECL:.*]]:2 = hlfir.declare
  ! CHECK: %[[C0:.*]] = arith.constant 0 : i32
  ! CHECK: hlfir.assign %[[C0]] to %[[DECL]]#0
  x = 0
end subroutine assumes_decl

! CHECK-LABEL: func.func @_QPassume_construct
subroutine assume_construct(x)
  integer :: x
  ! CHECK: %[[DECL:.*]]:2 = hlfir.declare
  !$omp assume no_openmp
  ! CHECK: %[[C1:.*]] = arith.constant 1 : i32
  ! CHECK: hlfir.assign %[[C1]] to %[[DECL]]#0
  x = 1
  !$omp end assume
end subroutine assume_construct

! CHECK-LABEL: func.func @_QPassume_block
subroutine assume_block(x)
  integer :: x
  ! CHECK: %[[DECL:.*]]:2 = hlfir.declare
  !$omp assume holds(x > 0)
  block
    ! CHECK: %[[LOAD:.*]] = fir.load %[[DECL]]#0
    ! CHECK: %[[C0:.*]] = arith.constant 0 : i32
    ! CHECK: %[[CMP:.*]] = arith.cmpi sgt, %[[LOAD]], %[[C0]] : i32
    ! CHECK: llvm.intr.assume %[[CMP]] : i1
    ! CHECK: %[[C2:.*]] = arith.constant 2 : i32
    ! CHECK: hlfir.assign %[[C2]] to %[[DECL]]#0
    x = 2
  end block
end subroutine assume_block
