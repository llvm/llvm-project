! Ensure that func.func arguments are given the Fortran attributes
! even if their first use is in a procedure designator reference
! and not a call.

! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

subroutine test(x)
  interface
    subroutine foo(x)
      integer, optional, target :: x
    end subroutine
  end interface
  integer, optional, target :: x
  call takes_proc(foo)
  call foo(x)
end subroutine
! CHECK: func.func private @_QPfoo(!fir.ref<i32> {fir.optional, fir.target})
