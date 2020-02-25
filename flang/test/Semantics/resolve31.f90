subroutine s1
  integer :: t0
  !ERROR: 't0' is not a derived type
  type(t0) :: x
  type :: t1
  end type
  type, extends(t1) :: t2
  end type
  !ERROR: Derived type 't3' not found
  type, extends(t3) :: t4
  end type
  !ERROR: 't0' is not a derived type
  type, extends(t0) :: t5
  end type
end subroutine

module m1
  type t0
  end type
end
module m2
  type t
  end type
end
module m3
  type t0
  end type
end
subroutine s2
  use m1
  use m2, t0 => t
  use m3
  !ERROR: Reference to 't0' is ambiguous
  type, extends(t0) :: t1
  end type
end subroutine

module m4
  type :: t1
    private
    sequence
    private  ! not a fatal error
  end type
  type :: t1a
  end type
  !ERROR: A sequence type may not have the EXTENDS attribute
  type, extends(t1a) :: t2
    sequence
    integer i
  end type
  type :: t3
    sequence
    integer i
  !ERROR: A sequence type may not have a CONTAINS statement
  contains
  end type
contains
  subroutine s3
    type :: t1
      !ERROR: PRIVATE is only allowed in a derived type that is in a module
      private
    contains
      !ERROR: PRIVATE is only allowed in a derived type that is in a module
      private
    end type
  end
end
