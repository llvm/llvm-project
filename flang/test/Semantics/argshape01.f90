! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
! Detect incompatible argument shapes
module m
  integer :: ha = 1
 contains
  subroutine s1(a)
    real, intent(in) :: a(2,3)
  end
  subroutine s2(a)
    real, intent(in) :: a(3,2)
  end
  subroutine s3(a)
    real, intent(in) :: a(3,*)
  end
  subroutine s4(a)
    real, intent(in) :: a(:,:)
  end
  subroutine s5(a)
    real, intent(in) :: a(..)
  end
  subroutine s6(a,n,m)
    integer, intent(in) :: n, m
    real, intent(in) :: a(n, m)
  end
  subroutine s6b(a,nn,mm)
    integer, intent(in) :: nn, mm
    real, intent(in) :: a(nn, mm)
  end
  subroutine s7(a,n,m)
    integer, intent(in) :: n, m
    real, intent(in) :: a(m, n)
  end
  subroutine s8(a,n,m)
    integer, intent(in) :: n, m
    real, intent(in) :: a(n+1,m+1)
  end
  subroutine s8b(a,n,m)
    integer, intent(in) :: n, m
    real, intent(in) :: a(n-1,m+2)
  end
  subroutine s9(a)
    real, intent(in) :: a(ha,ha)
  end
  subroutine s9b(a)
    real, intent(in) :: a(ha,ha)
  end
  subroutine s1c(s)
    procedure(s1) :: s
  end
  subroutine s2c(s)
    procedure(s2) :: s
  end
  subroutine s3c(s)
    procedure(s3) :: s
  end
  subroutine s4c(s)
    procedure(s4) :: s
  end
  subroutine s5c(s)
    procedure(s5) :: s
  end
  subroutine s6c(s)
    procedure(s6) :: s
  end
  subroutine s7c(s)
    procedure(s7) :: s
  end
  subroutine s8c(s)
    procedure(s8) :: s
  end
  subroutine s9c(s)
    procedure(s9) :: s
  end
end

program main
  use m
  procedure(s1), pointer :: ps1
  procedure(s2), pointer :: ps2
  procedure(s3), pointer :: ps3
  procedure(s4), pointer :: ps4
  procedure(s5), pointer :: ps5
  procedure(s6), pointer :: ps6
  procedure(s7), pointer :: ps7
  procedure(s8), pointer :: ps8
  procedure(s9), pointer :: ps9
  call s1c(s1)
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 's=': incompatible dummy argument #1: incompatible dummy data object shapes
  call s1c(s2)
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 's=': incompatible dummy argument #1: incompatible dummy data object shapes
  call s1c(s3)
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 's=': incompatible dummy argument #1: incompatible dummy data object shapes
  call s1c(s4)
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 's=': incompatible dummy argument #1: incompatible dummy data object shapes
  call s1c(s5)
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 's=': distinct numbers of dummy arguments
  call s1c(s6)
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 's=': incompatible dummy argument #1: incompatible dummy data object shapes
  call s2c(s1)
  call s2c(s2)
  call s6c(s6)
  call s6c(s6b)
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 's=': incompatible dummy argument #1: incompatible dummy data object shapes
  call s6c(s7)
  !WARNING: Actual procedure argument has interface incompatible with dummy argument 's=': incompatible dummy argument #1: incompatible dummy data object shapes
  call s6c(s8)
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 's=': incompatible dummy argument #1: incompatible dummy data object shapes
  call s7c(s6)
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 's=': incompatible dummy argument #1: incompatible dummy data object shapes
  call s7c(s8)
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 's=': incompatible dummy argument #1: incompatible dummy data object shapes
  call s8c(s6)
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 's=': incompatible dummy argument #1: incompatible dummy data object shapes
  call s8c(s7)
  call s8c(s8)
  !WARNING: Actual procedure argument has possible interface incompatibility with dummy argument 's=': possibly incompatible dummy argument #1: distinct dummy data object shapes
  call s8c(s8b)
  call s9c(s9)
  call s9c(s9b)
  ps1 => s1
  !ERROR: Procedure pointer 'ps1' associated with incompatible procedure designator 's2': incompatible dummy argument #1: incompatible dummy data object shapes
  ps1 => s2
  !ERROR: Procedure pointer 'ps1' associated with incompatible procedure designator 's3': incompatible dummy argument #1: incompatible dummy data object shapes
  ps1 => s3
  !ERROR: Procedure pointer 'ps1' associated with incompatible procedure designator 's4': incompatible dummy argument #1: incompatible dummy data object shapes
  ps1 => s4
  !ERROR: Procedure pointer 'ps1' associated with incompatible procedure designator 's5': incompatible dummy argument #1: incompatible dummy data object shapes
  ps1 => s5
  !ERROR: Procedure pointer 'ps1' associated with incompatible procedure designator 's6': distinct numbers of dummy arguments
  ps1 => s6
  !ERROR: Procedure pointer 'ps2' associated with incompatible procedure designator 's1': incompatible dummy argument #1: incompatible dummy data object shapes
  ps2 => s1
  ps2 => s2
  call s1c(ps1)
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 's=': incompatible dummy argument #1: incompatible dummy data object shapes
  call s1c(ps2)
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 's=': incompatible dummy argument #1: incompatible dummy data object shapes
  call s1c(ps3)
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 's=': incompatible dummy argument #1: incompatible dummy data object shapes
  call s1c(ps4)
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 's=': incompatible dummy argument #1: incompatible dummy data object shapes
  call s1c(ps5)
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 's=': incompatible dummy argument #1: incompatible dummy data object shapes
  call s2c(ps1)
  call s2c(ps2)
  ps6 => s6
  ps6 => s6b
  !ERROR: Procedure pointer 'ps6' associated with incompatible procedure designator 's7': incompatible dummy argument #1: incompatible dummy data object shapes
  ps6 => s7
  !ERROR: Procedure pointer 'ps6' associated with incompatible procedure designator 's8': incompatible dummy argument #1: incompatible dummy data object shapes
  ps6 => s8
  !ERROR: Procedure pointer 'ps7' associated with incompatible procedure designator 's6': incompatible dummy argument #1: incompatible dummy data object shapes
  ps7 => s6
  !ERROR: Procedure pointer 'ps7' associated with incompatible procedure designator 's8': incompatible dummy argument #1: incompatible dummy data object shapes
  ps7 => s8
  ps8 => s8
  !WARNING: pointer 'ps8' and s8b may not be completely compatible procedures: possibly incompatible dummy argument #1: distinct dummy data object shapes
  ps8 => s8b
  !ERROR: Procedure pointer 'ps8' associated with incompatible procedure designator 's6': incompatible dummy argument #1: incompatible dummy data object shapes
  ps8 => s6
  !WARNING: Procedure pointer 'ps8' associated with incompatible procedure designator 's7': incompatible dummy argument #1: incompatible dummy data object shapes
  ps8 => s7
end
