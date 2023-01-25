! RUN: %python %S/test_errors.py %s %flang_fc1 -Werror
! Catch discrepancies between a local interface and a global definition

subroutine global1(x)
  integer, intent(in) :: x
end subroutine

subroutine global2(x) bind(c,name="xyz")
  integer, intent(in) :: x
end subroutine

subroutine global3(x)
  integer, intent(in) :: x
end subroutine

pure subroutine global4(x)
  integer, intent(in) :: x
end subroutine

subroutine global5(x)
  integer, intent(in) :: x
end subroutine

program test
  interface
    !WARNING: The global subprogram 'global1' is not compatible with its local procedure declaration (incompatible dummy argument #1: incompatible dummy data object types: REAL(4) vs INTEGER(4))
    subroutine global1(x)
      real, intent(in) :: x
    end subroutine
    subroutine global2(x)
      real, intent(in) :: x
    end subroutine
    subroutine global3(x) bind(c,name="abc")
      real, intent(in) :: x
    end subroutine
    subroutine global4(x) ! not PURE, but that's ok
      integer, intent(in) :: x
    end subroutine
    !WARNING: The global subprogram 'global5' is not compatible with its local procedure declaration (incompatible procedure attributes: Pure)
    pure subroutine global5(x)
      integer, intent(in) :: x
    end subroutine
  end interface
end

