! RUN: %python %S/test_errors.py %s %flang_fc1

! Test that the interface of specific intrinsics passed as dummy arguments
! are correctly validated against actual arguments explicit interface.

  intrinsic :: abs, dabs
  interface
    subroutine foo(f)
      interface
        function f(x)
          real :: f
          real, intent(in) :: x
        end function
      end interface
    end subroutine

    subroutine foo2(f)
      interface
        function f(x)
          double precision :: f
          double precision, intent(in) :: x
        end function
      end interface
    end subroutine
  end interface

  ! OK
  call foo(abs)

  ! OK
  call foo2(dabs)

  !ERROR: Actual procedure argument has interface incompatible with dummy argument 'f='
  call foo(dabs)

  !ERROR: Actual procedure argument has interface incompatible with dummy argument 'f='
  call foo2(abs)
end
