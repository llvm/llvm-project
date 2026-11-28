! RUN: %python %S/test_errors.py %s %flang_fc1
! Type parameters must not reuse accessible inherited names.

module m
  type base(k)
    integer, kind :: k = 4
  end type

  !ERROR: Type parameter 'k' conflicts with an accessible name inherited from a parent type
  type, extends(base) :: child(K)
    integer, len :: K = 10
  end type

  type, extends(base) :: valid_child(l)
    integer, len :: l = 10
  end type

  !ERROR: Type parameter 'k' conflicts with an accessible name inherited from a parent type
  type, extends(valid_child) :: grandchild(k)
    integer, kind :: k = 4
  end type

  type(valid_child(4, 20)) :: valid_object
end module

module inherited_names
  type base
    integer :: c
  contains
    procedure, nopass :: f
    generic :: g => f
  end type

  !ERROR: Type parameter 'c' conflicts with an accessible name inherited from a parent type
  type, extends(base) :: t1(c)
    integer, len :: c
  end type

  !ERROR: Type parameter 'f' conflicts with an accessible name inherited from a parent type
  type, extends(base) :: t2(f)
    integer, len :: f
  end type

  !ERROR: Type parameter 'g' conflicts with an accessible name inherited from a parent type
  type, extends(base) :: t3(g)
    integer, len :: g
  end type

  type, extends(base) :: mid
  end type

  !ERROR: Type parameter 'base' conflicts with an accessible name inherited from a parent type
  type, extends(mid) :: t4(base)
    integer, len :: base
  end type
contains
  subroutine f
  end subroutine
end module

module m1
  type base
    integer, private :: c
  contains
    procedure, nopass, private :: f
    generic, private :: g => f
  end type
contains
  subroutine f
  end subroutine
end module

module m2
  use m1
  type, extends(base) :: t(c)
    integer, len :: c
  end type

  type, extends(base) :: private_bindings(f, g)
    integer, len :: f, g
  end type
end module

module accessible_private
  type base
    integer, private :: c
  end type

  !ERROR: Type parameter 'c' conflicts with an accessible name inherited from a parent type
  type, extends(base) :: child(c)
    integer, len :: c
  end type
end module

module missing_definition
  type base(k)
    integer, kind :: k = 4
  end type

  !ERROR: No definition found for type parameter 'k'
  type, extends(base) :: child(k)
  end type
end module

! A nearer inaccessible name must not hide an accessible ancestor's name.
module hidden_root
  type root
    integer, private :: p
  end type
  interface
    module subroutine anchor
    end subroutine
  end interface
end module

module hidden_near
  use hidden_root, only: root
  type, extends(root) :: near
    !WARNING: Component 'p' is inaccessibly declared in or as a parent of this derived type [-Wredeclared-inaccessible-component]
    integer, private :: p
  end type
end module

submodule(hidden_root) hidden_descendant
  use hidden_near, only: near
  !ERROR: Type parameter 'p' conflicts with an accessible name inherited from a parent type
  type, extends(near) :: child(p)
    integer, kind :: p = 4
  end type
contains
  module procedure anchor
  end procedure
end submodule
