! RUN: %flang -O0 -fopenmp -S -emit-llvm %s -o - 2>&1 | FileCheck %s

program subcallx

  implicit none

  call sub2

contains

subroutine sub1(x)

  implicit none

  integer :: x

  print *, "Hello from sub1", x

end subroutine sub1

! CHECK: define internal void @subcallx_sub2
subroutine sub2

  implicit none

  interface
    subroutine simple_sub(x)
      implicit none
      integer :: x
    end subroutine
  end interface

  procedure(simple_sub), pointer :: pptr => null() ! CHECK: %"pptr$p_{{[0-9]+}}" = alloca void

  pptr => sub1

! CHECK: @subcallx__{{.+}}_ to i64*
! CHECK: @__kmpc_fork_call
! CHECK: define internal void @subcallx__{{.+}}_
!$OMP PARALLEL
  call pptr(123) ! CHECK-NOT: %"pptr${{.+}}_{{[0-9]+}}" = alloca
!$OMP END PARALLEL

end subroutine sub2

end program subcallx
