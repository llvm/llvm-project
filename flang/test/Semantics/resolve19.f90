module m
  interface a
    subroutine s(x)
      real :: x
    end subroutine
    !ERROR: 's' is already declared in this scoping unit
    subroutine s(x)
      integer :: x
    end subroutine
  end interface
end module

module m2
  interface s
    subroutine s(x)
      real :: x
    end subroutine
    !ERROR: 's' is already declared in this scoping unit
    subroutine s(x)
      integer :: x
    end subroutine
  end interface
end module
