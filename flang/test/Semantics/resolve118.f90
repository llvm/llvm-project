! RUN: %python %S/test_errors.py %s %flang_fc1
! USE vs IMPORT
module m1
  type t
    integer n
  end type
end module

module m2
  type t
    real x
  end type
end module

module m3
  use m1
  interface
    subroutine s1(x)
      use m1
      !PORTABILITY: The same 't' is already present in this scope
      import t
      type(t) x
    end
    subroutine s2(x)
      use m2
      !ERROR: A distinct 't' is already present in this scope
      import t
      type(t) x
    end
  end interface
end module

module m4
  type t
    complex z
  end type
  interface
    subroutine s3(x)
      use m1
      !ERROR: A distinct 't' is already present in this scope
      import t
      type(t) x
    end
  end interface
end module

module m5
  interface
    subroutine s4(x)
      use m1
      !ERROR: A distinct 't' is already present in this scope
      import t
      type(t) x
    end
  end interface
 contains
  subroutine t
  end
end module
