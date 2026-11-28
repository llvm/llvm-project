! RUN: %flang_fc1 -emit-hlfir %s -o - | fir-opt --cse --opt-bufferization | FileCheck %s

! Constant size constructor of a trivial type can have its temporary removed.
! CHECK-LABEL: func.func @_QPbasic_scalar(
! CHECK-NOT: fir.allocmem
! CHECK: return
subroutine basic_scalar(dest, source)
  real :: dest(4)
  integer :: source
  dest = [real(source), real(2*source), real(3*source), real(4*source)]
end subroutine

! CHECK-LABEL: func.func @_QPbasic_implied_do(
! CHECK-NOT: fir.allocmem
! CHECK: return
subroutine basic_implied_do(dest, source)
  real :: dest(4)
  integer :: source
  dest = [(real(i*source), i = 1, 4)]
end subroutine

! Logical constructors should also be eligible.
! CHECK-LABEL: func.func @_QPlogical(
! CHECK-NOT: fir.allocmem
! CHECK: return
subroutine logical(dest, source)
  logical :: dest(3)
  integer :: source
  dest = [source > 1, source < 1, source == 3]
end subroutine

! A function reference in the constructor keeps the temporary.
! CHECK-LABEL: func.func @_QPconstructor_call(
! CHECK: fir.allocmem
! CHECK: hlfir.as_expr
! CHECK: hlfir.destroy
subroutine constructor_call(dest, source)
  interface
    real function f(i)
      integer :: i
    end function
  end interface
  real :: dest(4)
  integer :: source
  dest = [f(source), f(2*source), f(3*source), f(4*source)]
end subroutine

! A section destination can be hoisted to eliminate the constructor temporary.
! Even when the intial code doesn't match the pattern.
! CHECK-LABEL: func.func @_QPsection_non_default_lb(
! CHECK-NOT: fir.allocmem
! CHECK: return
subroutine section_non_default_lb(dest, source)
  real :: dest(-2:5), source(4)
  dest(2:5) = [source(4), source(3), source(2), source(1)]
end subroutine
