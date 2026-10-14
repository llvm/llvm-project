! RUN: %python %S/test_errors.py %s %flang_fc1

module bind_c_conflicting_type_name
  type :: x
  end type
  !ERROR: 'x' is already declared in this scoping unit
  integer, bind(c) :: x
end module
