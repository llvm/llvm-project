! Test lowering of assumed-size cray pointee. This is an
! odd case where an assumed-size symbol is not a dummy.
! Test that no bogus stack allocation is created for it
! (it will take its address from the cray pointer when used).
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

subroutine assumed_size_cray_ptr
  implicit none
  pointer(ivar,var)
  real :: var(*)
end subroutine
! CHECK-LABEL: func.func @_QPassumed_size_cray_ptr
! CHECK-NOT: fir.alloca !fir.array<?xf32>
! CHECK: return
