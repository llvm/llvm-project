! RUN: %S/test_errors.sh %s %t %f18

! C1568 The procedure-name shall have been declared to be a separate module
! procedure in the containing program unit or an ancestor of that program unit.
module m1
  interface
    module subroutine sub1(arg1)
      integer, intent(inout) :: arg1
    end subroutine
    module integer function fun1()
    end function
  end interface
  type t
  end type
  integer i
end module

submodule(m1) s1
contains
  !ERROR: 'missing1' was not declared a separate module procedure
  module procedure missing1
  end
  !ERROR: 'missing2' was not declared a separate module procedure
  module subroutine missing2
  end
  !ERROR: 't' was not declared a separate module procedure
  module procedure t
  end
  !ERROR: 'i' was not declared a separate module procedure
  module subroutine i
  end
end submodule

module m2
  interface
    module subroutine sub1(arg1)
      integer, intent(inout) :: arg1
    end subroutine
    module integer function fun1()
    end function
  end interface
  type t
  end type
  !ERROR: Declaration of 'i' conflicts with its use as module procedure
  integer i
contains
  !ERROR: 'missing1' was not declared a separate module procedure
  module procedure missing1
  end
  !ERROR: 'missing2' was not declared a separate module procedure
  module subroutine missing2
  end
  !ERROR: 't' is already declared in this scoping unit
  !ERROR: 't' was not declared a separate module procedure
  module procedure t
  end
  !ERROR: 'i' was not declared a separate module procedure
  module subroutine i
  end
end module

! Separate module procedure defined in same module as declared
module m3
  interface
    module subroutine sub
    end subroutine
  end interface
contains
  module procedure sub
  end procedure
end module

! Separate module procedure defined in a submodule
module m4
  interface
    module subroutine a
    end subroutine
    module subroutine b
    end subroutine
  end interface
end module
submodule(m4) s4a
contains
  module procedure a
  end procedure
end submodule
submodule(m4:s4a) s4b
contains
  module procedure b
  end procedure
end
