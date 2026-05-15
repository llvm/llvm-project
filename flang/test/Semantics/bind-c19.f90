! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for C1807: A procedure defined in a submodule shall not have a
! binding label unless its interface is declared in the ancestor module.

module m1
  implicit none
end module

! Submodule with BIND(C) procedures that have no interface in the ancestor
! module - these violate C1807.
submodule(m1) sm1
  implicit none
contains
  !ERROR: A procedure defined in a submodule shall not have a binding label unless its interface is declared in the ancestor module
  subroutine sub1() bind(c)
  end subroutine
  !ERROR: A procedure defined in a submodule shall not have a binding label unless its interface is declared in the ancestor module
  subroutine sub2() bind(c, name="my_sub2")
  end subroutine
  !ERROR: A procedure defined in a submodule shall not have a binding label unless its interface is declared in the ancestor module
  function func1() bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    integer(c_int) :: func1
    func1 = 0
  end function
end submodule

! Valid: interfaces declared in ancestor module.
module m2
  implicit none
  interface
    module subroutine sub3() bind(c, name="sub3")
    end subroutine
    module function func2() bind(c) result(res)
      use, intrinsic :: iso_c_binding, only: c_int
      integer(c_int) :: res
    end function
  end interface
end module

submodule(m2) sm2
  implicit none
contains
  module subroutine sub3() bind(c, name="sub3")
  end subroutine
  module function func2() bind(c) result(res)
    use, intrinsic :: iso_c_binding, only: c_int
    integer(c_int) :: res
    res = 0
  end function
end submodule

! Valid: BIND(C,NAME="") gives no binding label.
module m3
  implicit none
end module

submodule(m3) sm3
  implicit none
contains
  subroutine sub4() bind(c, name="")
  end subroutine
end submodule

! Valid: BIND(C,NAME=" ") gives no binding label (blanks are discarded).
module m4
  implicit none
end module

submodule(m4) sm4
  implicit none
contains
  subroutine sub5() bind(c, name=" ")
  end subroutine
end submodule

! Invalid: interface declared in a parent submodule, not the ancestor module.
! C1807 requires the interface be in the ancestor module.
module m5
  implicit none
end module

submodule(m5) sm5parent
  implicit none
  interface
    module subroutine sub6() bind(c, name="sub6")
    end subroutine
  end interface
end submodule

submodule(m5:sm5parent) sm5child
  implicit none
contains
  !ERROR: A procedure defined in a submodule shall not have a binding label unless its interface is declared in the ancestor module
  module subroutine sub6() bind(c, name="sub6")
  end subroutine
end submodule

! Valid: interface declared in the ancestor module, definition in a child
! submodule - C1807 is satisfied because the interface is in the module.
module m6
  implicit none
  interface
    module subroutine sub7() bind(c, name="sub7")
    end subroutine
  end interface
end module

submodule(m6) sm6parent
end submodule

submodule(m6:sm6parent) sm6child
  implicit none
contains
  module subroutine sub7() bind(c, name="sub7")
  end subroutine
end submodule
