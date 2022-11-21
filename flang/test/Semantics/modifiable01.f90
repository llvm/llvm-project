! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
! Test WhyNotModifiable() explanations

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
end module
