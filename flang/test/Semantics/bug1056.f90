! RUN: %flang_fc1 -fdebug-unparse %s | FileCheck %s
program bug
  integer, target :: ita(2) = [1,2], itb(2) = [3,4], itc(2) = [5,6]
  type t1
    integer, pointer :: p1(:) => ita, p2(:) => itb
  end type
  type t2
    !CHECK: TYPE(t1) :: comp = t1(p1=itc,p2=itb)
    type(t1) :: comp = t1(itc)
  end type
  integer, pointer :: p3(:) => itd
  integer, target :: itd(2) = [7,8]
end
