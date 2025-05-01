!RUN: not %flang_fc1 %s 2>&1 | FileCheck %s
!CHECK: error: KIND parameter expression (int(1_4/0_4,kind=8)) of intrinsic type CHARACTER did not resolve to a constant value
!CHECK: in the context: instantiation of parameterized derived type 'ty(j=1_4,k=0_4)'
!CHECK: warning: INTEGER(4) division by zero
program main
  type ty(j,k)
    integer, kind :: j, k
    character(kind=j/k) a
  end type
  type(ty(1,0)) x
end
