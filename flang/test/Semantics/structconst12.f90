!RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
type t1(k1a,k1b)
  integer, kind :: k1a, k1b
  integer(k1a) :: j = -666
  integer(k1b) :: c1 = k1a
end type
type t2(k2a,k2b)
  integer, kind:: k2a, k2b
  type(t1(k2a+1,k2b*2)) :: c2 = t1(k2a+1,k2b*2)(j=777)
end type
type (t2(3,4)), parameter :: x = t2(3,4)()
!CHECK: TYPE(t2(3_4,4_4)), PARAMETER :: x = t2(k2a=3_4,k2b=4_4)(c2=t1(k1a=4_4,k1b=8_4)(j=777_4,c1=4_8))
END
