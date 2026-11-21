! RUN: %python %S/test_errors.py %s %flang_fc1
! Type parameters may not redeclare inherited type parameters.

module m
  type base(k)
    integer, kind :: k = 4
  end type

  !ERROR: Type parameter 'k' is already declared in a parent of this derived type
  type, extends(base) :: child(K)
    integer, len :: K = 10
  end type

  type, extends(base) :: valid_child(l)
    integer, len :: l = 10
  end type

  !ERROR: Type parameter 'k' is already declared in a parent of this derived type
  type, extends(valid_child) :: grandchild(k)
    integer, kind :: k = 4
  end type

  type(valid_child(4, 20)) :: valid_object
end module
