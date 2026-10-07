! RUN: %flang_fc1 -fopenacc -fdebug-dump-symbols %s 2>&1 | FileCheck %s

! Flang extension: a named !$acc routine directive placed directly within an
! interface block applies to the interface body it names.  Verify that the
! ROUTINE information is attached to the named procedure's symbol and that no
! diagnostic is produced.

program p
  implicit none
  interface
  !$acc routine (foo) seq
  subroutine foo() bind(c, name="foo")
  end subroutine foo
  end interface
end program p

! CHECK: foo, BIND(C), EXTERNAL (Subroutine){{.*}}openACCRoutineInfos: seq
