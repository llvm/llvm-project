! RUN: %python %S/test_errors.py %s %flang_fc1 -funderscoring

subroutine conflict1()
end subroutine

!ERROR: BIND(C) procedure assembly name conflicts with non BIND(C) procedure assembly name
subroutine foo(x)  bind(c, name="conflict1_")
  real :: x
end subroutine

subroutine no_conflict1() bind(c, name="")
end subroutine
subroutine foo2() bind(c, name="conflict2_")
end subroutine

subroutine bar()
  interface
    subroutine no_conflict1() bind(c, name="")
    end subroutine
    ! ERROR: Non BIND(C) procedure assembly name conflicts with BIND(C) procedure assembly name
    subroutine conflict2()
    end subroutine
  end interface
  call no_conflict1()
  call conflict2
end subroutine

subroutine no_conflict2() bind(c, name="no_conflict2_")
end subroutine

subroutine _()
end subroutine

subroutine dash_no_conflict() bind(c, name="")
end subroutine
