! RUN: %python %S/test_errors.py %s %flang_fc1
! Ensures that things that aren't procedures aren't allowed to be called.
module m
  integer :: i
  integer, pointer :: ip
  type :: t
  end type
  type :: pdt(k,len)
    integer, kind :: k
    integer, len :: len
  end type
  type(pdt(1,2)) :: x
  !ERROR: 'i' is not a variable
  namelist /nml/i
 contains
  subroutine s(d)
    real d
    !ERROR: 'm' is not a callable procedure
    call m
    !ERROR: Cannot call function 'i' like a subroutine
    call i
    !ERROR: Cannot call function 'ip' like a subroutine
    call ip
    !ERROR: 't' is not a callable procedure
    call t
    !ERROR: 'k' is not a procedure
    call x%k
    !ERROR: 'len' is not a procedure
    call x%len
    !ERROR: Use of 'nml' as a procedure conflicts with its declaration
    call nml
    !ERROR: Cannot call function 'd' like a subroutine
    call d
  end subroutine
end
