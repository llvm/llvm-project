! Test that GoTo exiting an acc.loop seq region generates acc.yield
! instead of an illegal cross-region cf.br that would crash the compiler.
! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPfoo
subroutine foo(N, A, B)
  implicit real*8 (a-h, o-z)
  !$acc routine gang
  dimension A(*), B(*)
  !$acc loop gang vector
  do 100 i = 1, N
  !$acc loop seq
    do 10 j = 1, 1000
      if (A(i) .gt. B(i)) goto 20
10  continue
20  B(i) = A(i)
100 continue
end subroutine

! Verify the inner acc.loop (seq) contains acc.yield for the GoTo exit.
! The GoTo target is outside the acc.loop region, so it must yield
! instead of generating an illegal cross-region cf.br.

! CHECK: acc.loop gang vector
! CHECK: acc.loop
! The GoTo comparison and branch:
! CHECK: arith.cmpf ogt
! CHECK-NEXT: cf.cond_br %{{.*}}, ^[[EXIT:bb[0-9]+]], ^
! CHECK-NEXT: ^[[EXIT]]:
! CHECK-NEXT: acc.yield
! Normal loop end yield and closing:
! CHECK: acc.yield
! CHECK-NEXT: } attributes {seq = [#acc.device_type<none>], unstructured}
