! RUN: %S/test_modfile.sh %s %t %f18
! Check modfile generation for generic interfaces
module m1
  interface foo
    real function s1(x,y)
      real, intent(in) :: x
      logical, intent(in) :: y
    end function
    complex function s2(x,y)
      complex, intent(in) :: x
      logical, intent(in) :: y
    end function
  end interface
  generic :: operator ( + ) => s1, s2
  interface operator ( /= )
    logical function f1(x, y)
      real, intent(in) :: x
      logical, intent(in) :: y
    end function
  end interface
  interface
    logical function f2(x, y)
      complex, intent(in) :: x
      logical, intent(in) :: y
    end function
    logical function f3(x, y)
      integer, intent(in) :: x
      logical, intent(in) :: y
    end function
  end interface
  generic :: operator(.ne.) => f2
  generic :: operator(<>) => f3
  private :: operator( .ne. )
  interface bar
    procedure :: s1
    procedure :: s2
    procedure :: s3
    procedure :: s4
  end interface
  interface operator( .bar.)
    procedure :: s1
    procedure :: s2
    procedure :: s3
    procedure :: s4
  end interface
contains
  logical function s3(x,y)
    logical, intent(in) :: x,y
  end function
  integer function s4(x,y)
    integer, intent(in) :: x,y
  end function
end
!Expect: m1.mod
!module m1
! interface foo
!  procedure::s1
!  procedure::s2
! end interface
! interface
!  function s1(x,y)
!   real(4),intent(in)::x
!   logical(4),intent(in)::y
!   real(4)::s1
!  end
! end interface
! interface
!  function s2(x,y)
!   complex(4),intent(in)::x
!   logical(4),intent(in)::y
!   complex(4)::s2
!  end
! end interface
! interface operator(+)
!  procedure::s1
!  procedure::s2
! end interface
! interface operator(/=)
!  procedure::f1
!  procedure::f2
!  procedure::f3
! end interface
! private::operator(/=)
! interface
!  function f1(x,y)
!   real(4),intent(in)::x
!   logical(4),intent(in)::y
!   logical(4)::f1
!  end
! end interface
! interface
!  function f2(x,y)
!   complex(4),intent(in)::x
!   logical(4),intent(in)::y
!   logical(4)::f2
!  end
! end interface
! interface
!  function f3(x,y)
!   integer(4),intent(in)::x
!   logical(4),intent(in)::y
!   logical(4)::f3
!  end
! end interface
! interface bar
!  procedure::s1
!  procedure::s2
!  procedure::s3
!  procedure::s4
! end interface
! interface operator(.bar.)
!  procedure::s1
!  procedure::s2
!  procedure::s3
!  procedure::s4
! end interface
!contains
! function s3(x,y)
!  logical(4),intent(in)::x
!  logical(4),intent(in)::y
!  logical(4)::s3
! end
! function s4(x,y)
!  integer(4),intent(in)::x
!  integer(4),intent(in)::y
!  integer(4)::s4
! end
!end

module m1b
  use m1
end
!Expect: m1b.mod
!module m1b
! use m1,only:foo
! use m1,only:s1
! use m1,only:s2
! use m1,only:operator(+)
! use m1,only:f1
! use m1,only:f2
! use m1,only:f3
! use m1,only:bar
! use m1,only:operator(.bar.)
! use m1,only:s3
! use m1,only:s4
!end

module m1c
  use m1, only: myfoo => foo
  use m1, only: operator(.bar.)
  use m1, only: operator(.mybar.) => operator(.bar.)
  use m1, only: operator(+)
end
!Expect: m1c.mod
!module m1c
! use m1,only:myfoo=>foo
! use m1,only:operator(.bar.)
! use m1,only:operator(.mybar.)=>operator(.bar.)
! use m1,only:operator(+)
!end

module m2
  interface foo
    procedure foo
  end interface
contains
  complex function foo()
    foo = 1.0
  end
end
!Expect: m2.mod
!module m2
! interface foo
!  procedure::foo
! end interface
!contains
! function foo()
!  complex(4)::foo
! end
!end

module m2b
  type :: foo
    real :: x
  end type
  interface foo
  end interface
  private :: bar
  interface bar
  end interface
end
!Expect: m2b.mod
!module m2b
! interface foo
! end interface
! type::foo
!  real(4)::x
! end type
! interface bar
! end interface
! private::bar
!end

! Test interface nested inside another interface
module m3
  interface g
    subroutine s1(f)
      interface
        real function f(x)
          interface
            subroutine x()
            end subroutine
          end interface
        end function
      end interface
    end subroutine
  end interface
end
!Expect: m3.mod
!module m3
! interface g
!  procedure::s1
! end interface
! interface
!  subroutine s1(f)
!   interface
!    function f(x)
!     interface
!      subroutine x()
!      end
!     end interface
!     real(4)::f
!    end
!   end interface
!  end
! end interface
!end

module m4
  interface foo
    integer function foo()
    end function
    integer function f(x)
    end function
  end interface
end
subroutine s4
  use m4
  i = foo()
end
!Expect: m4.mod
!module m4
! interface foo
!  procedure::foo
!  procedure::f
! end interface
! interface
!  function foo()
!   integer(4)::foo
!  end
! end interface
! interface
!  function f(x)
!   real(4)::x
!   integer(4)::f
!  end
! end interface
!end

! Compile contents of m4.mod and verify it gets the same thing again.
module m5
 interface foo
  procedure::foo
  procedure::f
 end interface
 interface
  function foo()
   integer(4)::foo
  end
 end interface
 interface
  function f(x)
   integer(4)::f
   real(4)::x
  end
 end interface
end
!Expect: m5.mod
!module m5
! interface foo
!  procedure::foo
!  procedure::f
! end interface
! interface
!  function foo()
!   integer(4)::foo
!  end
! end interface
! interface
!  function f(x)
!   real(4)::x
!   integer(4)::f
!  end
! end interface
!end

module m6a
  interface operator(<)
    logical function lt(x, y)
      logical, intent(in) :: x, y
    end function
  end interface
end
!Expect: m6a.mod
!module m6a
! interface operator(<)
!  procedure::lt
! end interface
! interface
!  function lt(x,y)
!   logical(4),intent(in)::x
!   logical(4),intent(in)::y
!   logical(4)::lt
!  end
! end interface
!end

module m6b
  use m6a, only: operator(.lt.)
end
!Expect: m6b.mod
!module m6b
! use m6a,only:operator(.lt.)
!end
