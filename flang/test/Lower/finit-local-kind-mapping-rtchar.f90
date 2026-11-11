! Tests that the runtime-length CHARACTER stride computation in genInitLocal
! uses the code unit's allocation stride rather than charBits / 8.  This path
! is separate from the fixed-length path in genInitLocalStore and requires its
! own compilation: the first TODO in a run aborts lowering, so a file that
! contains both a fixed-length and a runtime-length subroutine would stop at
! the fixed-length one and never reach the runtime-length guard.
!
! A runtime-length CHARACTER local is declared as character(kind=1, len=n)
! where n is a dummy argument, so the length is not known at compile time.
!
! RUN: bbc -emit-hlfir --kind-mapping=a1:24 -finit-local=0xAA %s -o - 2>&1 | \
! RUN:     FileCheck --check-prefix=RT-24BIT %s
! RUN: bbc -emit-hlfir --kind-mapping=a1:12 -finit-local=0xAA %s -o - 2>&1 | \
! RUN:     FileCheck --check-prefix=RT-12BIT %s
! RUN: bbc -emit-hlfir --kind-mapping=a1:1  -finit-local=0xAA %s -o - 2>&1 | \
! RUN:     FileCheck --check-prefix=RT-1BIT  %s

! RT-24BIT-NOT: not yet implemented
! RT-24BIT: %[[FACTOR24:.*]] = arith.constant 4 : index
! RT-24BIT: %[[NBYTES24:.*]] = arith.muli %{{.*}}, %[[FACTOR24]] : index
! RT-24BIT: %[[LAST24:.*]] = arith.subi %[[NBYTES24]], %{{.*}} : index
! RT-24BIT: fir.do_loop %{{.*}} = %{{.*}} to %[[LAST24]] step %{{.*}}

! RT-12BIT-NOT: not yet implemented
! RT-12BIT: %[[FACTOR12:.*]] = arith.constant 2 : index
! RT-12BIT: %[[NBYTES12:.*]] = arith.muli %{{.*}}, %[[FACTOR12]] : index
! RT-12BIT: %[[LAST12:.*]] = arith.subi %[[NBYTES12]], %{{.*}} : index
! RT-12BIT: fir.do_loop %{{.*}} = %{{.*}} to %[[LAST12]] step %{{.*}}

! RT-1BIT-NOT: not yet implemented
! RT-1BIT-NOT: arith.muli
! RT-1BIT: %[[LAST1:.*]] = arith.subi %{{.*}}, %{{.*}} : index
! RT-1BIT: fir.do_loop %{{.*}} = %{{.*}} to %[[LAST1]] step %{{.*}}

subroutine test_rt_char(n, res)
  integer, intent(in) :: n
  character(kind=1, len=n) :: c
  integer :: res
  res = ichar(c(1:1))
end subroutine
