! Test lowering of firstprivate on derived type with pointer components.
! No deep copy must be done.

! TODO: ensure pointer components are initialized to NULL for private.

! RUN: not bbc -fopenacc -emit-hlfir %s -o - 2>&1 | FileCheck %s

! CHECK: not yet implemented: OpenACC: privatizing derived type with pointer components

module m_firstprivate_derived_ptr_comp
 type point
   real, pointer :: x(:)
 end type point
 contains
   subroutine test(a)
     type(point) :: a

     !$acc parallel loop firstprivate(a)
     do i = 1, n
      a%x(10) = 1
     enddo
   end
end module
