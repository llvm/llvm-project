! RUN: %python %S/test_errors.py %s %flang_fc1
! Test 15.7 C1594 - prohibited assignments in pure subprograms

module used
  real :: useassociated
end module

module m
  type :: t
    sequence
    real :: a
  end type
  type(t), target :: x
  type :: hasPtr
    real, pointer :: p
  end type
  type :: hasCoarray
    real, allocatable :: co[:]
  end type
 contains
  integer pure function purefunc(x)
    integer, intent(in) :: x
    purefunc = x
  end function
  integer pure function f00(p0)
    procedure(purefunc) :: p0
    f00 = p0(1)
  end function
  pure function test(ptr, in, hpd)
    use used
    type(t), pointer :: ptr, ptr2
    type(t), target, intent(in) :: in
    type(t), target :: y, z
    type(hasPtr) :: hp
    type(hasPtr), intent(in) :: hpd
    type(hasPtr), allocatable :: alloc
    type(hasCoarray), pointer :: hcp
    integer :: n
    common /block/ y
    external :: extfunc
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'x' may not be defined in pure subprogram 'test' because it is host-associated
    x%a = 0.
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'y' may not be defined in pure subprogram 'test' because it is in a COMMON block
    y%a = 0. ! C1594(1)
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'useassociated' may not be defined in pure subprogram 'test' because it is USE-associated
    useassociated = 0.  ! C1594(1)
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'ptr' is externally visible via 'ptr' and not definable in a pure subprogram
    ptr%a = 0. ! C1594(1)
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: 'in' is an INTENT(IN) dummy argument
    in%a = 0. ! C1594(1)
    !ERROR: Left-hand side of assignment is not definable
    !BECAUSE: A pure subprogram may not define the coindexed object 'hcp%co[1_8]'
    hcp%co[1] = 0. ! C1594(1)
    !ERROR: The left-hand side of a pointer assignment is not definable
    !BECAUSE: 'ptr' may not be defined in pure subprogram 'test' because it is a POINTER dummy argument of a pure function
    ptr => z ! C1594(2)
    !ERROR: 'ptr' may not appear in NULLIFY
    !BECAUSE: 'ptr' may not be defined in pure subprogram 'test' because it is a POINTER dummy argument of a pure function
    nullify(ptr) ! C1594(2), 19.6.8
    !ERROR: A pure subprogram may not use 'ptr' as the target of pointer assignment because it is a POINTER dummy argument of a pure function
    ptr2 => ptr ! C1594(3)
    !ERROR: A pure subprogram may not use 'in' as the target of pointer assignment because it is an INTENT(IN) dummy argument
    ptr2 => in ! C1594(3)
    !ERROR: A pure subprogram may not use 'y' as the target of pointer assignment because it is in a COMMON block
    ptr2 => y ! C1594(2)
    !ERROR: Externally visible object 'block' may not be associated with pointer component 'p' in a pure procedure
    n = size([hasPtr(y%a)]) ! C1594(4)
    !ERROR: Externally visible object 'x' may not be associated with pointer component 'p' in a pure procedure
    n = size([hasPtr(x%a)]) ! C1594(4)
    !ERROR: Externally visible object 'ptr' may not be associated with pointer component 'p' in a pure procedure
    n = size([hasPtr(ptr%a)]) ! C1594(4)
    !ERROR: Externally visible object 'in' may not be associated with pointer component 'p' in a pure procedure
    n = size([hasPtr(in%a)]) ! C1594(4)
    !ERROR: A pure subprogram may not copy the value of 'hpd' because it is an INTENT(IN) dummy argument and has the POINTER component '%p'
    hp = hpd ! C1594(5)
    !ERROR: A pure subprogram may not copy the value of 'hpd' because it is an INTENT(IN) dummy argument and has the POINTER component '%p'
    allocate(alloc, source=hpd)
    !ERROR: Actual procedure argument for dummy argument 'p0=' of a PURE procedure must have an explicit interface
    n = f00(extfunc)
   contains
    pure subroutine internal
      type(hasPtr) :: localhp
      !ERROR: Left-hand side of assignment is not definable
      !BECAUSE: 'z' may not be defined in pure subprogram 'internal' because it is host-associated
      z%a = 0.
      !ERROR: Externally visible object 'z' may not be associated with pointer component 'p' in a pure procedure
      localhp = hasPtr(z%a)
    end subroutine
  end function
end module
