! RUN: %python %S/test_errors.py %s %flang_fc1
type A (p, r)
  integer, kind :: p, r
  !ERROR: KIND parameter expression (int(selected_real_kind(six,twenty_three),kind=8)) of intrinsic type REAL did not resolve to a constant value
  real (selected_real_kind(p, r)) :: data
end type
   integer :: six = 6, twenty_three = 23
   type(a(6,23)) :: a1
   !ERROR: Value of KIND type parameter 'p' must be constant
   !ERROR: Value of KIND type parameter 'r' must be constant
   !WARNING: specification expression refers to local object 'six' (initialized and saved) [-Wsaved-local-in-spec-expr]
   !WARNING: specification expression refers to local object 'twenty_three' (initialized and saved) [-Wsaved-local-in-spec-expr]
   type(a(six, twenty_three)) :: a2
   print *, a1%data%kind
end
