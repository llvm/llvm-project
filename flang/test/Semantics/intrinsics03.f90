! RUN: %python %S/test_errors.py %s %flang_fc1
! Ensure that INDEX is a usable specific intrinsic procedure.

program test
  interface
    pure integer function index1(string, substring)
      character(*), intent(in) :: string, substring ! ok
    end
    pure integer function index2(x1, x2)
      character(*), intent(in) :: x1, x2 ! ok
    end
    pure integer function index3(string, substring)
      character, intent(in) :: string, substring ! not assumed length
    end
    pure integer function index4(string, substring, back)
      character(*), intent(in) :: string, substring
      logical, optional, intent(in) :: back ! not ok
    end
    subroutine s0(ix)
      procedure(index) :: ix
    end
    subroutine s1(ix)
      import index1
      procedure(index1) :: ix
    end
    subroutine s2(ix)
      import index2
      procedure(index2) :: ix
    end
    subroutine s3(ix)
      import index3
      procedure(index3) :: ix
    end
    subroutine s4(ix)
      import index4
      procedure(index4) :: ix
    end
  end interface

  procedure(index), pointer :: p0
  procedure(index1), pointer :: p1
  procedure(index2), pointer :: p2
  procedure(index3), pointer :: p3
  procedure(index4), pointer :: p4

  p0 => index ! ok
  p0 => index1 ! ok
  p0 => index2 ! ok
  !ERROR: Procedure pointer 'p0' associated with incompatible procedure designator 'index3': incompatible dummy argument #1: assumed-length character vs explicit-length character
  p0 => index3
  !ERROR: Procedure pointer 'p0' associated with incompatible procedure designator 'index4': distinct numbers of dummy arguments
  p0 => index4
  p1 => index ! ok
  p1 => index1 ! ok
  p1 => index2 ! ok
  !ERROR: Procedure pointer 'p1' associated with incompatible procedure designator 'index3': incompatible dummy argument #1: assumed-length character vs explicit-length character
  p1 => index3
  !ERROR: Procedure pointer 'p1' associated with incompatible procedure designator 'index4': distinct numbers of dummy arguments
  p1 => index4
  p2 => index ! ok
  p2 => index1 ! ok
  p2 => index2 ! ok
  !ERROR: Procedure pointer 'p2' associated with incompatible procedure designator 'index3': incompatible dummy argument #1: assumed-length character vs explicit-length character
  p2 => index3
  !ERROR: Procedure pointer 'p2' associated with incompatible procedure designator 'index4': distinct numbers of dummy arguments
  p2 => index4
  !ERROR: Procedure pointer 'p3' associated with incompatible procedure designator 'index': incompatible dummy argument #1: assumed-length character vs explicit-length character
  p3 => index
  !ERROR: Procedure pointer 'p3' associated with incompatible procedure designator 'index1': incompatible dummy argument #1: assumed-length character vs explicit-length character
  p3 => index1
  !ERROR: Procedure pointer 'p3' associated with incompatible procedure designator 'index2': incompatible dummy argument #1: assumed-length character vs explicit-length character
  p3 => index2
  p3 => index3 ! ok
  !ERROR: Procedure pointer 'p3' associated with incompatible procedure designator 'index4': distinct numbers of dummy arguments
  p3 => index4
  !ERROR: Procedure pointer 'p4' associated with incompatible procedure designator 'index': distinct numbers of dummy arguments
  p4 => index
  !ERROR: Procedure pointer 'p4' associated with incompatible procedure designator 'index1': distinct numbers of dummy arguments
  p4 => index1
  !ERROR: Procedure pointer 'p4' associated with incompatible procedure designator 'index2': distinct numbers of dummy arguments
  p4 => index2
  !ERROR: Procedure pointer 'p4' associated with incompatible procedure designator 'index3': distinct numbers of dummy arguments
  p4 => index3
  p4 => index4 ! ok

  call s0(index) ! ok
  call s0(index1) ! ok
  call s0(index2)
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 'ix=': incompatible dummy argument #1: assumed-length character vs explicit-length character
  call s0(index3)
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 'ix=': distinct numbers of dummy arguments
  call s0(index4)
  call s1(index) ! ok
  call s1(index1) ! ok
  call s1(index2) ! ok
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 'ix=': incompatible dummy argument #1: assumed-length character vs explicit-length character
  call s1(index3)
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 'ix=': distinct numbers of dummy arguments
  call s1(index4)
  call s2(index) ! ok
  call s2(index1) ! ok
  call s2(index2) ! ok
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 'ix=': incompatible dummy argument #1: assumed-length character vs explicit-length character
  call s2(index3)
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 'ix=': distinct numbers of dummy arguments
  call s2(index4)
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 'ix=': incompatible dummy argument #1: assumed-length character vs explicit-length character
  call s3(index)
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 'ix=': incompatible dummy argument #1: assumed-length character vs explicit-length character
  call s3(index1)
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 'ix=': incompatible dummy argument #1: assumed-length character vs explicit-length character
  call s3(index2)
  call s3(index3) ! ok
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 'ix=': distinct numbers of dummy arguments
  call s3(index4)
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 'ix=': distinct numbers of dummy arguments
  call s4(index)
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 'ix=': distinct numbers of dummy arguments
  call s4(index1)
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 'ix=': distinct numbers of dummy arguments
  call s4(index2)
  !ERROR: Actual procedure argument has interface incompatible with dummy argument 'ix=': distinct numbers of dummy arguments
  call s4(index3)
  call s4(index4) ! ok
end
