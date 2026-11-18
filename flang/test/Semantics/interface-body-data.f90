! RUN: %python %S/test_errors.py %s %flang_fc1

module m
  interface
    subroutine ms(x)
      integer :: x
      integer :: y
      !ERROR: A DATA statement may not appear in an interface body
      data y /1/
    end subroutine
  end interface
end module

module mop
  type t
    integer :: i
  end type
  interface operator(+)
    function add(a, b) result(r)
      import t
      type(t), intent(in) :: a, b
      type(t) :: r
      integer :: w
      !ERROR: A DATA statement may not appear in an interface body
      data w /3/
    end function
  end interface
end module

program p
  interface
    subroutine s
      integer :: x
      !ERROR: A DATA statement may not appear in an interface body
      data x /1/
    end subroutine
    pure subroutine sp
      integer :: z
      !ERROR: A DATA statement may not appear in an interface body
      data z /2/
    end subroutine
    subroutine cb
      real :: v
      common /blk/ v
      !ERROR: A DATA statement may not appear in an interface body
      data v /0./
    end subroutine
    subroutine ok
      ! Legal type declaration initialization is not a DATA statement.
      ! it is simply a specification with no effect (F2023 15.4.3.2 p6).
      integer :: v = 1
    end subroutine
  end interface
end program
