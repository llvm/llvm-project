! Regression test for #192581:
! DECLARE SIMD applies to the external procedure declared via the interface
! body, not to the enclosing program unit or procedure. Lowering must not emit
! an omp.declare_simd op in the enclosing scope.

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

interface
  subroutine add2(i)
  !$omp declare simd(add2) linear(i:1)
    integer :: i
  end subroutine
end interface

! Implicit (no procedure name), same scope situation.
interface
  subroutine add3(i)
  !$omp declare simd
    integer :: i
  end subroutine
end interface

end

! Interfaces in subroutine / function; multiple nesting levels.
subroutine a(x)
  implicit none
  integer :: x

  interface
    subroutine b(y)
      !$omp declare simd linear(y:1)
      integer :: y
      interface
        function c(z)
          !$omp declare simd linear(z:1)
          integer :: z
        end function
      end interface
    end subroutine
  end interface

  call b(x)
end subroutine

! CHECK-NOT:   omp.declare_simd
