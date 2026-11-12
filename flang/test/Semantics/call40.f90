! RUN: %python %S/test_errors.py %s %flang_fc1
! %VAL en %REF legacy extension semantic tests.

subroutine val_errors(array, string, polymorphic, derived)
  type t
    integer :: t
  end type
  integer :: array(10)
  character(*) :: string
  type(t) :: derived
  type(*) :: polymorphic
  interface
    subroutine foo5(a)
      integer a(:)
    end
  end interface
  !ERROR: %VAL argument must be a scalar numeric or logical expression
  call foo1(%val(array))
  !ERROR: %VAL argument must be a scalar numeric or logical expression
  call foo2(%val(string))
  !ERROR: %VAL argument must be a scalar numeric or logical expression
  call foo3(%val(derived))
  !ERROR: Assumed type actual argument requires an explicit interface
  !ERROR: %VAL argument must be a scalar numeric or logical expression
  call foo4(%val(polymorphic))
  !ERROR: %VAL or %REF are not allowed for dummy argument 'a=' that must be passed by means of a descriptor
  call foo5(%ref(array))
end subroutine

subroutine val_ok()
  integer :: array(10)
  real :: x
  logical :: l
  complex :: c
  call ok1(%val(array(1)))
  call ok2(%val(x))
  call ok3(%val(l))
  call ok4(%val(c))
  call ok5(%val(42))
  call ok6(%val(x+x))
end subroutine

subroutine ref_ok(array, string, derived)
  type t
    integer :: t
  end type
  integer :: array(10)
  character(*) :: string
  type(t) :: derived
  call rok1(%ref(array))
  call rok2(%ref(string))
  call rok3(%ref(derived))
end subroutine
