! RUN: bbc %s -o - | tco | llc --relocation-model=pic --filetype=obj -o %t.o
! RUN: %CC %t.o -L%L -Wl,-rpath=%L -lFortran_main -lFortranRuntime -lFortranDecimal -lm -o %t.out
! RUN: %t.out | FileCheck %s

program p
  integer :: n, foo
  n = 0
  associate (i => n, j => n + 10, k => foo(20))
!   CHECK: 0 0 10 20
    print*, n, i, j, k
    n = n + 1
!   CHECK: 1 1 10 20
    print*, n, i, j, k
    i = i + 1
!   CHECK: 2 2 10 20
    print*, n, i, j, k
  end associate
! CHECK: 2
  print*, n
end

integer function foo(x)
  integer x
  integer, save :: i = 0
  i = i + x
  foo = i
end function foo
