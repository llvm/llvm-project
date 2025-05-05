! RUN: %flang_fc1 -fdebug-dump-symbols %s 2>&1 | FileCheck %s

!CHECK: Module scope: m1
!CHECK:       i, PUBLIC size={{[0-9]+}} offset={{[0-9]+}}: ObjectEntity type: REAL({{[0-9]+}}) init:{{.+}}
!CHECK:       init, PUBLIC (Subroutine): Subprogram ()
!CHECK:       o, PUBLIC (CrayPointee) size={{[0-9]+}} offset={{[0-9]+}}: ObjectEntity type: REAL({{[0-9]+}})
!CHECK:       ptr, PUBLIC (CrayPointer) size={{[0-9]+}} offset={{[0-9]+}}: ObjectEntity type: INTEGER({{[0-9]+}})
module m1
  implicit none
  real:: o
  real:: i = 42.0
  pointer (ptr, o)
contains
  !CHECK: Subprogram scope: init
  subroutine init
    implicit none
    ptr=loc(i)
    print *, "init : o= ", o
  end subroutine init
end module m1

!CHECK: Module scope: m2
!CHECK:       i, PUBLIC: Use from i in m1
!CHECK:       i2, PUBLIC size={{[0-9]+}} offset={{[0-9]+}}: ObjectEntity type: REAL({{[0-9]+}}) init:{{.+}}
!CHECK:       init, PUBLIC (Subroutine): Use from init in m1
!CHECK:       o, PUBLIC (CrayPointee): Use from o in m1
!CHECK:       ptr, PUBLIC (CrayPointer): Use from ptr in m1
!CHECK:       reset, PUBLIC (Subroutine): Subprogram ()
module m2
  use m1
  implicit none
  real:: i2 = 777.0
contains
  !CHECK: Subprogram scope: reset
  !CHECK:       o2 (CrayPointee) size={{[0-9]+}} offset={{[0-9]+}}: ObjectEntity type: REAL({{[0-9]+}})
  !CHECK:       ptr (CrayPointer) size={{[0-9]+}} offset={{[0-9]+}}: ObjectEntity type: INTEGER({{[0-9]+}})
  subroutine reset
    real::o2
    pointer (ptr, o2)
    ptr=loc(i2)
    print *, "reset : o= ", o, " o2 = ", o2
    o2 = 666.0
  end subroutine reset
end module m2

!CHECK: MainProgram scope: main
!CHECK:       i: Use from i in m2
!CHECK:       i2: Use from i2 in m2
!CHECK:       init (Subroutine): Use from init in m2
!CHECK:       o (CrayPointee): Use from o in m2
!CHECK:       ptr (CrayPointer): Use from ptr in m2
!CHECK:       reset (Subroutine): Use from reset in m2
program main
  use m2
  implicit none
  call init
  call reset
  write(6,*) "main : o = ", o
  if (o == 42.0) then
    print *, "pass"
  else
    print *, "fail"
  end if
end program main