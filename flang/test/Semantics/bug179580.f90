!RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Werror
type t
  real v
  integer, pointer :: p
end type
type(t), allocatable :: a
!WARNING: Structure constructor lacks a value for pointer component 'p', NULL() assumed [-Wdefault-struct-constructor-null-pointer]
allocate(a, source=t(v=3.3))
end
