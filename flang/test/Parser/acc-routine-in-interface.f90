! RUN: %flang_fc1 -fopenacc -fdebug-unparse %s | FileCheck %s
! RUN: %flang_fc1 -fopenacc -fdebug-dump-parse-tree %s | FileCheck %s --check-prefix=TREE

! Flang extension: an !$acc routine directive is accepted directly within an
! interface block (as an interface-specification), e.g. preceding the named
! interface body it applies to.  The OpenACC specification only permits the
! named routine directive in the specification part of a subroutine, function,
! or module.

program p
  implicit none
  interface
  !$acc routine (foo) seq
  subroutine foo() bind(c, name="foo")
  end subroutine foo
  end interface
end program p

! CHECK: INTERFACE
! CHECK: !$ACC ROUTINE(foo) SEQ
! CHECK: SUBROUTINE foo () BIND(C, NAME="foo")
! CHECK: END SUBROUTINE foo
! CHECK: END INTERFACE

! TREE: InterfaceSpecification -> OpenACCRoutineConstruct
! TREE: InterfaceSpecification -> InterfaceBody -> Subroutine
