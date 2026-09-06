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
! RT-24BIT: fir.do_loop

! RT-12BIT-NOT: not yet implemented
! RT-12BIT: fir.do_loop

! RT-1BIT-NOT: not yet implemented
! RT-1BIT: fir.do_loop

subroutine test_rt_char(n, res)
  integer, intent(in) :: n
  character(kind=1, len=n) :: c
  integer :: res
  res = ichar(c(1:1))
end subroutine
