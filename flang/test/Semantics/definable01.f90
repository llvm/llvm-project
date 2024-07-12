! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
! Test WhyNotDefinable() explanations

module prot
  real, protected :: prot
  type :: ptype
    real, pointer :: ptr
    real :: x
  end type
  type(ptype), protected :: protptr
 contains
  subroutine ok
    prot = 0. ! ok
  end subroutine
end module

module m
  use iso_fortran_env
  use prot
  type :: t1
    type(lock_type) :: lock
  end type
  type :: t2
    type(t1) :: x1
    real :: x2
  end type
  type(t2) :: t2static
  type list
    real a
    type(list), pointer :: prev, next
  end type
  character(*), parameter :: internal = '0'
 contains
  subroutine test1(dummy)
    real :: arr(2)
    integer, parameter :: j3 = 666
    type(ptype), intent(in) :: dummy
    type(t2) :: t2var
    associate (a => 3+4)
      !CHECK: error: Input variable 'a' is not definable
      !CHECK: because: 'a' is construct associated with an expression
      read(internal,*) a
    end associate
    associate (a => arr([1])) ! vector subscript
      !CHECK: error: Input variable 'a' is not definable
      !CHECK: because: Construct association 'a' has a vector subscript
      read(internal,*) a
    end associate
    associate (a => arr(2:1:-1))
      read(internal,*) a ! ok
    end associate
    !CHECK: error: Input variable 'j3' is not definable
    !CHECK: because: '666_4' is not a variable
    read(internal,*) j3
    !CHECK: error: Left-hand side of assignment is not definable
    !CHECK: because: 't2var' is an entity with either an EVENT_TYPE or LOCK_TYPE
    t2var = t2static
    t2var%x2 = 0. ! ok
    !CHECK: error: Left-hand side of assignment is not definable
    !CHECK: because: 'prot' is protected in this scope
    prot = 0.
    protptr%ptr = 0. ! ok
    !CHECK: error: Left-hand side of assignment is not definable
    !CHECK: because: 'dummy' is an INTENT(IN) dummy argument
    dummy%x = 0.
    dummy%ptr = 0. ! ok
  end subroutine
  pure subroutine test2(ptr)
    integer, pointer, intent(in) :: ptr
    !CHECK: error: Input variable 'ptr' is not definable
    !CHECK: because: 'ptr' is externally visible via 'ptr' and not definable in a pure subprogram
    read(internal,*) ptr
  end subroutine
  subroutine test3(objp, procp)
    real, intent(in), pointer :: objp
    procedure(sin), pointer, intent(in) :: procp
    !CHECK: error: Actual argument associated with INTENT(IN OUT) dummy argument 'op=' is not definable
    !CHECK: because: 'objp' is an INTENT(IN) dummy argument
    call test3a(objp)
    !CHECK: error: Actual argument associated with procedure pointer dummy argument 'pp=' may not be INTENT(IN)
    call test3b(procp)
  end subroutine
  subroutine test3a(op)
    real, intent(in out), pointer :: op
  end subroutine
  subroutine test3b(pp)
    procedure(sin), pointer, intent(in out) :: pp
  end subroutine
  subroutine test4(p)
    type(ptype), pointer, intent(in) :: p
    p%x = 1.
    p%ptr = 1. ! ok
    nullify(p%ptr) ! ok
    !CHECK: error: 'p' may not appear in NULLIFY
    !CHECK: because: 'p' is an INTENT(IN) dummy argument
    nullify(p)
  end
  subroutine test5(np)
    type(ptype), intent(in) :: np
    !CHECK: error: 'ptr' may not appear in NULLIFY
    !CHECK: because: 'np' is an INTENT(IN) dummy argument
    nullify(np%ptr)
  end
  pure function test6(lp)
    type(list), pointer :: lp
    !CHECK: error: The left-hand side of a pointer assignment is not definable
    !CHECK: because: 'lp' may not be defined in pure subprogram 'test6' because it is a POINTER dummy argument of a pure function
    lp%next%next => null()
  end
  pure subroutine test7(lp)
    type(list), pointer :: lp
    !CHECK-NOT: error:
    lp%next%next => null()
  end
end module
