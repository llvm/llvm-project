! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! Exercise an array-constructor implied-do index after the OpenACC semantics
! pass has analyzed an enclosing array subscript.  The enclosing declaration
! with the same name must not turn the implied-do index into a SymbolRef.
subroutine implied_do_index_host(a, x, n)
  integer :: n
  integer :: a(n, n), x(n), i

  x = [(sum(a(:, i)), i = 1, n)]
end subroutine

! CHECK-LABEL: func.func @_QPimplied_do_index_host
! CHECK: hlfir.elemental
! CHECK: %[[HOST_INDEX:.*]] = fir.convert %{{.*}} : (index) -> i64
! CHECK-NEXT: %[[HOST_KIND:.*]] = fir.convert %[[HOST_INDEX]] : (i64) -> i32
! CHECK-NEXT: %[[HOST_SUBSCRIPT:.*]] = fir.convert %[[HOST_KIND]] : (i32) -> i64
! CHECK: hlfir.designate {{.*}}%[[HOST_SUBSCRIPT]]

! Preserve the explicit kind of an implied-do index when its resolved symbol,
! rather than an active ExpressionAnalyzer entry, identifies the index.
subroutine implied_do_index_kind8(a, x, n)
  integer :: n
  integer :: a(n, n), x(n), i

  x = [(sum(a(:, i)), integer(8) :: i = 1, n)]
end subroutine

! CHECK-LABEL: func.func @_QPimplied_do_index_kind8
! CHECK: hlfir.elemental
! CHECK: %[[KIND8_INDEX:.*]] = fir.convert %{{.*}} : (index) -> i64
! CHECK-NOT: fir.convert %[[KIND8_INDEX]]
! CHECK: hlfir.designate {{.*}}%[[KIND8_INDEX]]
