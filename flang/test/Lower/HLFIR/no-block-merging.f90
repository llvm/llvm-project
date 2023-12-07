! Test that dead code elimination after lowering does not merge
! blocks. This test was added because block merging on the
! code below created a block with a fir.shape<> block argument
! which FIR codegen is not intended to support.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

subroutine test(res, selector, x1, x2, x3, x4, x5, x6, x7, x8)
  integer :: res
  character(*) :: selector
  real(8), dimension(7) :: x1, x2, x3, x4, x5, x6, x7, x8
  interface
   integer function foo(x)
     real(8) :: x(:)
   end function
  end interface
  select case (selector)
    case ("01")
      res = foo(x1)
      res = foo(x1)
    case ("02")
      res = foo(x2)
      res = foo(x2)
    case ("03")
      res = foo(x3)
      res = foo(x3)
    case ("04")
      res = foo(x4)
      res = foo(x4)
    case ("05")
      res = foo(x5)
      res = foo(x5)
    case ("06")
      res = foo(x6)
      res = foo(x6)
    case ("07")
      res = foo(x7)
      res = foo(x7)
    case ("08")
      res = foo(x8)
      res = foo(x8)
    case default
   end select
end subroutine
! CHECK: fir.call @_QPfoo
! CHECK: fir.call @_QPfoo
! CHECK: fir.call @_QPfoo
! CHECK: fir.call @_QPfoo
! CHECK: fir.call @_QPfoo
! CHECK: fir.call @_QPfoo
! CHECK: fir.call @_QPfoo
! CHECK: fir.call @_QPfoo
! CHECK: fir.call @_QPfoo
! CHECK: fir.call @_QPfoo
! CHECK: fir.call @_QPfoo
! CHECK: fir.call @_QPfoo
! CHECK: fir.call @_QPfoo
! CHECK: fir.call @_QPfoo
! CHECK: fir.call @_QPfoo
! CHECK: fir.call @_QPfoo
