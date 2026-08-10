! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
program p
  type t(k,n)
    integer, kind :: k
    integer(k), len :: n
!CHECK: warning: INTEGER(1) addition overflowed
    integer :: c = n + 1_1
  end type
!CHECK: in the context: instantiation of parameterized derived type 't(k=1_4,n=127_1)'
  print *, t(1,127)()
end

!CHECK:  PRINT *, t(k=1_4,n=127_1)(c=-128_4)


