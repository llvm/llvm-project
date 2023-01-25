! RUN: %python %S/test_errors.py %s %flang_fc1
! Exercise ways to define and extend non-type-bound generics
! TODO: crashes compiler (infinite recursion) when build with MSVC
! XFAIL: system-windows

module m1
  type :: t1; end type
  type :: t2; end type
  interface operator(.eq.)
    module procedure :: eq1
  end interface
  generic :: operator(==) => eq2
 contains
  logical function eq1(x, y)
    type(t1), intent(in) :: x
    type(t2), intent(in) :: y
    eq1 = .true.
  end function
  logical function eq2(y, x)
    type(t2), intent(in) :: y
    type(t1), intent(in) :: x
    eq2 = .true.
  end function
  subroutine test1
    type(t1) :: a
    type(t2) :: b
    if (a == b .and. b .eq. a) print *, 'ok'
  end subroutine
end module

module m2
  use m1
  type :: t3; end type
  interface operator(==)
    module procedure eq3
  end interface
  generic :: operator(.eq.) => eq4
 contains
  logical function eq3(x, y)
    type(t1), intent(in) :: x
    type(t3), intent(in) :: y
    eq3 = .true.
  end function
  logical function eq4(y, x)
    type(t3), intent(in) :: y
    type(t1), intent(in) :: x
    eq4 = .true.
  end function
  subroutine test2
    type(t1) :: a
    type(t2) :: b
    type(t3) :: c
    if (a == b .and. b .eq. a .and. a == c .and. c .eq. a) print *, 'ok'
  end subroutine
end module

module m3
  use m2
 contains
  logical function eq5(x, y)
    type(t2), intent(in) :: x
    type(t3), intent(in) :: y
    eq5 = .true.
  end function
  logical function eq6(y, x)
    type(t3), intent(in) :: y
    type(t2), intent(in) :: x
    eq6 = .true.
  end function
  subroutine test3
    interface operator(==)
      module procedure :: eq5
    end interface
    type(t1) :: a
    type(t2) :: b
    type(t3) :: c
    if (a == b .and. b .eq. a .and. a == c .and. c .eq. a .and. b == c) print *, 'ok'
    block
      generic :: operator(.eq.) => eq6
      if (a == b .and. b .eq. a .and. a == c .and. c .eq. a .and. b == c .and. c .eq. b) print *, 'ok'
    end block
   contains
    subroutine inner
      interface operator(.eq.)
        module procedure :: eq6
      end interface
      if (a == b .and. b .eq. a .and. a == c .and. c .eq. a .and. b == c .and. c .eq. b) print *, 'ok'
    end subroutine
  end subroutine
end module
