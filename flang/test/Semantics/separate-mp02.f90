! RUN: %python %S/test_errors.py %s %flang_fc1

! When a module subprogram has the MODULE prefix the following must match
! with the corresponding separate module procedure interface body:
! - C1549: characteristics and dummy argument names
! - C1550: binding label
! - C1551: NON_RECURSIVE prefix

module m1
  interface
    module subroutine s4(x)
      real, intent(in) :: x
    end
    module subroutine s5(x, y)
      real, pointer :: x
      real, value :: y
    end
    module subroutine s6(x, y)
      real :: x
      real :: y
    end
    module subroutine s7(x, y, z)
      real :: x(8)
      real :: y(8)
      real :: z(8)
    end
    module subroutine s8(x, y, z)
      real :: x(8)
      real :: y(*)
      real :: z(*)
    end
    module subroutine s9(x, y, z, w)
      character(len=4) :: x
      character(len=4) :: y
      character(len=*) :: z
      character(len=*) :: w
    end
  end interface
end

submodule(m1) sm1
contains
  module subroutine s4(x)
    !ERROR: The intent of dummy argument 'x' does not match the intent of the corresponding argument in the interface body
    real, intent(out) :: x
  end
  module subroutine s5(x, y)
    !ERROR: Dummy argument 'x' has the OPTIONAL attribute; the corresponding argument in the interface body does not
    real, pointer, optional :: x
    !ERROR: Dummy argument 'y' does not have the VALUE attribute; the corresponding argument in the interface body does
    real :: y
  end
  module subroutine s6(x, y)
    !ERROR: Dummy argument 'x' has type INTEGER(4); the corresponding argument in the interface body has type REAL(4)
    integer :: x
    !ERROR: Dummy argument 'y' has type REAL(8); the corresponding argument in the interface body has type REAL(4)
    real(8) :: y
  end
  module subroutine s7(x, y, z)
    integer, parameter :: n = 8
    real :: x(n)
    real :: y(2:n+1)
    !ERROR: The shape of dummy argument 'z' does not match the shape of the corresponding argument in the interface body
    real :: z(n+1)
  end
  module subroutine s8(x, y, z)
    !ERROR: The shape of dummy argument 'x' does not match the shape of the corresponding argument in the interface body
    real :: x(*)
    real :: y(*)
    !ERROR: The shape of dummy argument 'z' does not match the shape of the corresponding argument in the interface body
    real :: z(8)
  end
  module subroutine s9(x, y, z, w)
    character(len=4) :: x
    !ERROR: Dummy argument 'y' has type CHARACTER(KIND=1,LEN=5_8); the corresponding argument in the interface body has type CHARACTER(KIND=1,LEN=4_8)
    character(len=5) :: y
    character(len=*) :: z
    !ERROR: Dummy argument 'w' has type CHARACTER(KIND=1,LEN=4_8); the corresponding argument in the interface body has type CHARACTER(KIND=1,LEN=*)
    character(len=4) :: w
  end
end

module m2
  interface
    module subroutine s1(x, y)
      real, intent(in) :: x
      real, intent(out) :: y
    end
    module subroutine s2(x, y)
      real, intent(in) :: x
      real, intent(out) :: y
    end
    module subroutine s3(x, y)
      real(4) :: x
      procedure(real) :: y
    end
    module subroutine s4()
    end
    non_recursive module subroutine s5()
    end
  end interface
end

submodule(m2) sm2
contains
  !ERROR: Module subprogram 's1' has 3 args but the corresponding interface body has 2
  module subroutine s1(x, y, z)
    real, intent(in) :: x
    real, intent(out) :: y
    real :: z
  end
  module subroutine s2(x, z)
    real, intent(in) :: x
  !ERROR: Dummy argument name 'z' does not match corresponding name 'y' in interface body
    real, intent(out) :: z
  end
  module subroutine s3(x, y)
    !ERROR: Dummy argument 'x' is a procedure; the corresponding argument in the interface body is not
    procedure(real) :: x
    !ERROR: Dummy argument 'y' is a data object; the corresponding argument in the interface body is not
    real :: y
  end
  !ERROR: Module subprogram 's4' has NON_RECURSIVE prefix but the corresponding interface body does not
  non_recursive module subroutine s4()
  end
  !ERROR: Module subprogram 's5' does not have NON_RECURSIVE prefix but the corresponding interface body does
  module subroutine s5()
  end
end

module m2b
  interface
    module subroutine s1()
    end
    module subroutine s2() bind(c, name="s2")
    end
    module subroutine s3() bind(c, name="s3")
    end
    module subroutine s4() bind(c, name=" s4")
    end
    module subroutine s5() bind(c)
    end
    module subroutine s6() bind(c)
    end
  end interface
end

submodule(m2b) sm2b
  character(*), parameter :: suffix = "_xxx"
contains
  !ERROR: Module subprogram 's1' has a binding label but the corresponding interface body does not
  module subroutine s1() bind(c, name="s1")
  end
  !ERROR: Module subprogram 's2' does not have a binding label but the corresponding interface body does
  module subroutine s2()
  end
  !ERROR: Module subprogram 's3' has binding label 's3_xxx' but the corresponding interface body has 's3'
  module subroutine s3() bind(c, name="s3" // suffix)
  end
  module subroutine s4() bind(c, name="s4  ")
  end
  module subroutine s5() bind(c, name=" s5")
  end
  !ERROR: Module subprogram 's6' has binding label 'not_s6' but the corresponding interface body has 's6'
  module subroutine s6() bind(c, name="not_s6")
  end
end


module m3
  interface
    module subroutine s1(x, y, z)
      procedure(real), pointer, intent(in) :: x
      procedure(real), pointer, intent(out) :: y
      procedure(real), pointer, intent(out) :: z
    end
    module subroutine s2(x, y)
      procedure(real), pointer :: x
      procedure(real) :: y
    end
  end interface
end

submodule(m3) sm3
contains
  module subroutine s1(x, y, z)
    procedure(real), pointer, intent(in) :: x
    !ERROR: The intent of dummy argument 'y' does not match the intent of the corresponding argument in the interface body
    procedure(real), pointer, intent(inout) :: y
    !ERROR: The intent of dummy argument 'z' does not match the intent of the corresponding argument in the interface body
    procedure(real), pointer :: z
  end
  module subroutine s2(x, y)
    !ERROR: Dummy argument 'x' has the OPTIONAL attribute; the corresponding argument in the interface body does not
    !ERROR: Dummy argument 'x' does not have the POINTER attribute; the corresponding argument in the interface body does
    procedure(real), optional :: x
    !ERROR: Dummy argument 'y' has the POINTER attribute; the corresponding argument in the interface body does not
    procedure(real), pointer :: y
  end
end

module m4
  interface
    subroutine s_real(x)
      real :: x
    end
    subroutine s_real2(x)
      real :: x
    end
    subroutine s_integer(x)
      integer :: x
    end
    module subroutine s1(x)
      procedure(s_real) :: x
    end
    module subroutine s2(x)
      procedure(s_real) :: x
    end
  end interface
end

submodule(m4) sm4
contains
  module subroutine s1(x)
    !OK
    procedure(s_real2) :: x
  end
  module subroutine s2(x)
    !ERROR: Dummy procedure 'x' does not match the corresponding argument in the interface body
    procedure(s_integer) :: x
  end
end

module m5
  interface
    module function f1()
      real :: f1
    end
    module subroutine s2()
    end
  end interface
end

submodule(m5) sm5
contains
  !ERROR: Module subroutine 'f1' was declared as a function in the corresponding interface body
  module subroutine f1()
  end
  !ERROR: Module function 's2' was declared as a subroutine in the corresponding interface body
  module function s2()
  end
end

module m6
  interface
    module function f1()
      real :: f1
    end
    module function f2()
      real :: f2
    end
    module function f3()
      real :: f3
    end
  end interface
end

submodule(m6) ms6
contains
  !OK
  real module function f1()
  end
  !ERROR: Return type of function 'f2' does not match return type of the corresponding interface body
  integer module function f2()
  end
  !ERROR: Return type of function 'f3' does not match return type of the corresponding interface body
  module function f3()
    real :: f3
    pointer :: f3
  end
end

module m7
  interface
    module subroutine s1(x, *)
      real :: x
    end
  end interface
end

submodule(m7) sm7
contains
  !ERROR: Dummy argument 1 of 's1' is an alternate return indicator but the corresponding argument in the interface body is not
  !ERROR: Dummy argument 2 of 's1' is not an alternate return indicator but the corresponding argument in the interface body is
  module subroutine s1(*, x)
    real :: x
  end
end
