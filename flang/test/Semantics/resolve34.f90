! Extended derived types

module m1
  type :: t1
    integer :: x
    !ERROR: Component 'x' is already declared in this derived type
    real :: x
  end type
end

module m2
  type :: t1
    integer :: i
  end type
  type, extends(t1) :: t2
    !ERROR: Component 'i' is already declared in a parent of this derived type
    integer :: i
  end type
end

module m3
  type :: t1
  end type
  type, extends(t1) :: t2
    integer :: i
    !ERROR: 't1' is a parent type of this type and so cannot be a component
    real :: t1
  end type
  type, extends(t2) :: t3
    !ERROR: 't1' is a parent type of this type and so cannot be a component
    real :: t1
  end type
end

module m4
  type :: t1
    integer :: t1
  end type
  !ERROR: Type cannot be extended as it has a component named 't1'
  type, extends(t1) :: t2
  end type
end

module m5
  type :: t1
    integer :: t2
  end type
  type, extends(t1) :: t2
  end type
  !ERROR: Type cannot be extended as it has a component named 't2'
  type, extends(t2) :: t3
  end type
end

module m6
  ! t1 can be extended if it is known as anything but t3
  type :: t1
    integer :: t3
  end type
  type, extends(t1) :: t2
  end type
end
subroutine s6
  use :: m6, only: t3 => t1
  !ERROR: Type cannot be extended as it has a component named 't3'
  type, extends(t3) :: t4
  end type
end
subroutine r6
  use :: m6, only: t5 => t1
  type, extends(t5) :: t6
  end type
end

module m7
  type, private :: t1
    integer :: i1
  end type
  type, extends(t1) :: t2
    integer :: i2
    integer, private :: i3
  end type
end
subroutine s7
  use m7
  type(t2) :: x
  integer :: j
  j = x%i2
  !ERROR: PRIVATE component 'i3' is only accessible within module 'm7'
  j = x%i3
  !ERROR: PRIVATE component 't1' is only accessible within module 'm7'
  j = x%t1%i1
end
