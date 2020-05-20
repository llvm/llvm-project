! RUN: %S/test_errors.sh %s %t %f18
module m
  public
  type t
    integer, private :: i
  end type
  !ERROR: The default accessibility of this module has already been declared
  private  !C869
end

subroutine s1
  !ERROR: PUBLIC statement may only appear in the specification part of a module
  public  !C869
end

subroutine s2
  !ERROR: PRIVATE attribute may only appear in the specification part of a module
  integer, private :: i  !C817
end

subroutine s3
  type t
    !ERROR: PUBLIC attribute may only appear in the specification part of a module
    integer, public :: i  !C817
  end type
end

module m4
  interface
    module subroutine s()
    end subroutine
  end interface
end
submodule(m4) sm4
  !ERROR: PUBLIC statement may only appear in the specification part of a module
  public  !C869
  !ERROR: PUBLIC attribute may only appear in the specification part of a module
  real, public :: x  !C817
  type :: t
    !ERROR: PRIVATE attribute may only appear in the specification part of a module
    real, private :: y  !C817
  end type
end
