!RUN: %flang -g -fopenmp -S -emit-llvm %s -o - | FileCheck %s

!CHECK: define internal void @main_sub
!CHECK: define internal void @__nv_main_sub_
!CHECK:  call void @llvm.dbg.declare(metadata ptr %"res$p
!CHECK-NEXT:  call void @llvm.dbg.declare(metadata ptr %"res$p
!CHECK-NEXT:  call void @llvm.dbg.declare(metadata ptr %"res$sd

program main
  type :: dtype
    integer(4) :: fdim
    real(8), pointer :: fld_ptr(:)
  end type dtype
  type(dtype) :: dvar
  allocate(dvar%fld_ptr(100))
  call sub(dvar)
  deallocate(dvar%fld_ptr)

contains

  subroutine sub(arg)
    type(dtype),intent(inout) :: arg
    integer:: count               ! indices
    real(8), pointer :: res(:)
!$OMP PARALLEL DO PRIVATE (COUNT, RES)
    do count=1, 100
      res  => arg%fld_ptr(1:10)
    end do
  end subroutine sub
end program main
