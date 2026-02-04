!RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
type child
  integer, pointer :: id
end type
integer, parameter :: n = 5
integer, save, target :: t1(n)
type(child) :: t2(n) = [(child(t1(i)), i=1,n)]
!CHECK:  TYPE(child) :: t2(5_4) = [child::child(id=t1(1_8)),child(id=t1(2_8)),child(id=t1(3_8)),child(id=t1(4_8)),child(id=t1(5_8))]
end
