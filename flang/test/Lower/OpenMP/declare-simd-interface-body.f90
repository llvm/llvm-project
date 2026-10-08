! Regression test for #192581:
! DECLARE SIMD applies to the external procedure declared via the interface
! body, not to the enclosing program unit.
! Lowering must not emit an omp.declare_simd op in the enclosing
! scope (the op-based form requires operands referring to the procedure's
! dummy arguments, which are not in scope here, and creating it would yield
! null operands).

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

print *, 'pass'
end

! CHECK-LABEL: func.func @_QQmain()
! CHECK-NOT:   omp.declare_simd
! CHECK:       return

