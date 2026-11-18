! RUN: %python %S/test_errors.py %s %flang_fc1

module bind_c_derived_type_conflict
  type :: x
  end type
  !ERROR: 'x' is already declared in this scoping unit
  integer, bind(c) :: x
end module

module bind_c_multiple_binding_names
  !ERROR: The entity 'x' has multiple BIND names ('a' and 'b')
  integer, bind(c, name="a") :: x
  !ERROR: BIND_C attribute was already specified on 'x'
  !ERROR: The type of 'x' has already been declared as INTEGER(4)
  real, bind(c, name="b") :: x
end module

module bind_c_parameter_conflict
  integer, parameter :: n = 10
  !ERROR: The type of 'n' has already been declared as INTEGER(4)
  integer, bind(c) :: n
end module

module bind_c_generic_interface_conflict
  interface g
  end interface
  !ERROR: 'g' is already declared in this scoping unit
  integer, bind(c) :: g
end module

module bind_c_namelist_conflict
  integer :: a
  namelist /nl/ a
  !ERROR: 'nl' is already declared in this scoping unit
  integer, bind(c) :: nl
end module

module bind_c_subsequent_module_procedure_conflict
  !ERROR: Declaration of 's' conflicts with its use as module procedure
  integer, bind(c) :: s
contains
  subroutine s()
  end subroutine
end module

module bind_c_derived_type_array_conflict
  type :: t
  end type
  !ERROR: 't' is already declared in this scoping unit
  integer, bind(c) :: t(2)
end module
