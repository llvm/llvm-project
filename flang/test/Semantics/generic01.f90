! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
! Tests rules of 15.5.5.2 for generics and explicit intrinsics
! competing at various scoping levels.
module m1
  private
  public abs
  interface abs
    module procedure :: abs_int_redef, abs_noargs
  end interface
contains
  integer function abs_int_redef(j)
    integer, intent(in) :: j
    abs_int_redef = j
  end function
  integer function abs_noargs()
    abs_noargs = 0
  end function
end module

module m2
  private
  public abs
  interface abs
    module procedure abs_real_redef
  end interface
contains
  real function abs_real_redef(x)
    real, intent(in) :: x
    abs_real_redef = x
  end function
end module

module m3
  use m1, only: abs
  implicit none
contains
  subroutine test1
    use m2, only: abs
    !CHECK: abs_int_redef(
    print *, abs(1)
    !CHECK: abs_real_redef(
    print *, abs(1.)
    !CHECK: 1.41421353816986083984375_4
    print *, abs((1,1))
    !CHECK: abs_noargs(
    print *, abs()
  end subroutine
  subroutine test2
    intrinsic abs ! override some of module's use of m1
    block
      use m2, only: abs
      !CHECK: 1_4
      print *, abs(1)
      !CHECK: abs_real_redef(
      print *, abs(1.)
      !CHECK: 1.41421353816986083984375_4
      print *, abs((1,1))
      !CHECK: abs_noargs(
      print *, abs()
    end block
  end subroutine
  subroutine test3
    interface abs
      module procedure abs_int_redef2 ! override module's use of m1
    end interface
    !CHECK: abs_int_redef2(
    print *, abs(1)
    !CHECK: 1._4
    print *, abs(1.)
    !CHECK: 1.41421353816986083984375_4
    print *, abs((1,1))
    !CHECK: abs_noargs(
    print *, abs()
    block
      use m1, only: abs ! override the override
      !CHECK: abs_int_redef(
      print *, abs(1)
    end block
  end subroutine
  integer function abs_int_redef2(j)
    integer, intent(in) :: j
    abs_int_redef2 = j
  end function
end module
