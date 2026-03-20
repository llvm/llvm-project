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

! Verify the inner acc.loop has seq and unstructured attributes,
! and that it contains acc.yield (from the GoTo cross-region exit).
! CHECK: acc.loop gang vector
! CHECK: acc.loop
! CHECK: acc.yield
! CHECK: acc.yield
! CHECK: } attributes {seq = [#acc.device_type<none>], unstructured}
