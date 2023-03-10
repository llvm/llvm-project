! RUN: %python %S/test_errors.py %s %flang_fc1
! Detect incompatible argument shapes
module m
 contains
  subroutine s1(a)
    real, intent(in) :: a(2,3)
  end
  subroutine s2(a)
    real, intent(in) :: a(3,2)
  end
  subroutine s1c(s)
    procedure(s1) :: s
  end
  subroutine s2c(s)
    procedure(s2) :: s
  end
end

program main
  use m
  procedure(s1), pointer :: ps1
  procedure(s2), pointer :: ps2
  call s1c(s1)
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 's=': incompatible dummy argument #1: incompatible dummy data object shapes
  call s1c(s2)
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 's=': incompatible dummy argument #1: incompatible dummy data object shapes
  call s2c(s1)
  call s2c(s2)
  ps1 => s1
  !ERROR: Procedure pointer 'ps1' associated with incompatible procedure designator 's2': incompatible dummy argument #1: incompatible dummy data object shapes
  ps1 => s2
  !ERROR: Procedure pointer 'ps2' associated with incompatible procedure designator 's1': incompatible dummy argument #1: incompatible dummy data object shapes
  ps2 => s1
  ps2 => s2
  call s1c(ps1)
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 's=': incompatible dummy argument #1: incompatible dummy data object shapes
  call s1c(ps2)
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 's=': incompatible dummy argument #1: incompatible dummy data object shapes
  call s2c(ps1)
  call s2c(ps2)
end
