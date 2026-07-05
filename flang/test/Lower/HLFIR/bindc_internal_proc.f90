! Test that internal procedure with BIND(C) do not have binding labels,
! that is, that they are generated using usual flang mangling for non BIND(C)
! internal procedures.
! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

!CHECK: func.func private @_QFsub1Pfoo(%{{.*}}: i32
subroutine sub1()
  call foo(42)
contains
  subroutine foo(i) bind(c)
    integer, value :: i
    print *, i
  end subroutine
end subroutine

!CHECK: func.func private @_QFsub2Pfoo(%{{.*}}: i64
subroutine sub2()
  call foo(42_8)
contains
  subroutine foo(i) bind(c)
    integer(8), value :: i
    print *, i
  end subroutine
end subroutine
