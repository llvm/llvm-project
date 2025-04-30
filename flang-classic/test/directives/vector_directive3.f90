!! check for pragma support for !dir$ vector always
!RUN: %flang -S -O2 -emit-llvm %s -o - | FileCheck %s
!CHECK: define void @sumsimd_{{.*$}}
!CHECK: !llvm.access.group ![[ACCGRP:[0-9]+]]
!CHECK: vector.ph:
!CHECK: vector.body:
!CHECK: add nsw <{{[0-9]+}} x i32>
!CHECK: "llvm.loop.parallel_accesses", ![[ACCGRP]]}
!CHECK: "llvm.loop.isvectorized", i32 1
!CHECK: "llvm.loop.unroll.runtime.disable"

SUBROUTINE sumsimd(myarr1,myarr2,ub)
  INTEGER, POINTER :: myarr1(:)
  INTEGER, POINTER :: myarr2(:)
  INTEGER :: ub

  !DIR$ VECTOR ALWAYS
  DO i=1,ub
      myarr1(i) = myarr1(i)+myarr2(i)
  END DO
END SUBROUTINE
