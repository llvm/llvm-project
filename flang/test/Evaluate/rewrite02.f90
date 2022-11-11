! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
! Tests handling of easily-misparsed substrings and substring
! type parameter inquiries.
subroutine foo(j)
  integer, intent(in) :: j
  character*4 sc, ac(1)
  type t
    character*4 sc, ac(1)
  end type
  type(t) st, at(1)
  !CHECK: PRINT *, sc(1_8:int(j,kind=8))
  print *, sc(1:j)
  !CHECK: PRINT *, ac(1_8)(1_8:int(j,kind=8))
  print *, ac(1)(1:j)
  !CHECK: PRINT *, st%sc(1_8:int(j,kind=8))
  print *, st%sc(1:j)
  !CHECK: PRINT *, st%ac(1_8)(1_8:int(j,kind=8))
  print *, st%ac(1)(1:j)
  !CHECK: PRINT *, at(1_8)%sc(1_8:int(j,kind=8))
  print *, at(1)%sc(1:j)
  !CHECK: PRINT *, at(1_8)%ac(1_8)(1_8:int(j,kind=8))
  print *, at(1)%ac(1)(1:j)
  !CHECK: PRINT *, 1_4
  print *, sc(1:j)%kind
  !CHECK: PRINT *, 1_4
  print *, ac(1)(1:j)%kind
  !CHECK: PRINT *, 1_4
  print *, st%sc(1:j)%kind
  !CHECK: PRINT *, 1_4
  print *, st%ac(1)(1:j)%kind
  !CHECK: PRINT *, 1_4
  print *, at(1)%sc(1:j)%kind
  !CHECK: PRINT *, 1_4
  print *, at(1)%ac(1)(1:j)%kind
  !CHECK: PRINT *, int(max(0_8,int(j,kind=8)-1_8+1_8),kind=4)
  print *, sc(1:j)%len
  !CHECK: PRINT *, int(max(0_8,int(j,kind=8)-1_8+1_8),kind=4)
  print *, ac(1)(1:j)%len
  !CHECK: PRINT *, int(max(0_8,int(j,kind=8)-1_8+1_8),kind=4)
  print *, st%sc(1:j)%len
  !CHECK: PRINT *, int(max(0_8,int(j,kind=8)-1_8+1_8),kind=4)
  print *, st%ac(1)(1:j)%len
  !CHECK: PRINT *, int(max(0_8,int(j,kind=8)-1_8+1_8),kind=4)
  print *, at(1)%sc(1:j)%len
  !CHECK: PRINT *, int(max(0_8,int(j,kind=8)-1_8+1_8),kind=4)
  print *, at(1)%ac(1)(1:j)%len
end
