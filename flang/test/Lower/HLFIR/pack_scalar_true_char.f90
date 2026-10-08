! RUN: %flang_fc1 %s -o %t.o
! RUN: %flang_fc1 -O2 %s -o %t2.o

program p
  character(3) :: a(4) = ['abc', 'def', 'ghi', 'jkl']
  character(3), allocatable :: b(:)
  b = pack(a, .true.)
  print *, b
end
