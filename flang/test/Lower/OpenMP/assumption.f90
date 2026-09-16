! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 -o - %s 2>&1 | FileCheck %s

! Lowering of the OpenMP ASSUME construct. The HOLDS clause asserts a scalar
! logical expression, which is lowered to an llvm.assume intrinsic. The block is
! always lowered.

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
