! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -fopenmp-version=52 -cpp -DDEPEND -o - %s 2>&1 | FileCheck %s --check-prefix=DEPEND
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -fopenmp-version=52 -cpp -DDEVICE -o - %s 2>&1 | FileCheck %s --check-prefix=DEVICE
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -fopenmp-version=52 -cpp -DIS_DEVICE_PTR -o - %s 2>&1 | FileCheck %s --check-prefix=IS_DEVICE_PTR
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -fopenmp-version=52 -cpp -DNOCONTEXT -o - %s 2>&1 | FileCheck %s --check-prefix=NOCONTEXT

! DEPEND: not yet implemented: Unhandled clause DEPEND in DISPATCH construct
! DEVICE: not yet implemented: Unhandled clause DEVICE in DISPATCH construct
! IS_DEVICE_PTR: not yet implemented: Unhandled clause IS_DEVICE_PTR in DISPATCH construct
! NOCONTEXT: not yet implemented: Unhandled clause NOCONTEXT in DISPATCH construct

subroutine sub()
#ifdef IS_DEVICE_PTR
  use iso_c_binding
  type(c_ptr) :: x
#endif
  integer :: r
#ifdef DEPEND
!$omp dispatch depend(inout: r)
#endif
#ifdef DEVICE
!$omp dispatch device(0)
#endif
#ifdef IS_DEVICE_PTR
!$omp dispatch is_device_ptr(x)
#endif
#ifdef NOCONTEXT
!$omp dispatch nocontext(.false.)
#endif
  call foo()
contains
  subroutine foo
  end subroutine
end subroutine sub
