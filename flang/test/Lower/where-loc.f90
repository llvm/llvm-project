! Test line location lowering of WHERE statements.
! RUN: %flang -fc1 -emit-hlfir -mmlir -mlir-print-debuginfo -mmlir --mlir-print-local-scope -o - %s | FileCheck %s

! CHECK-LABEL: func.func @_QPtest_where_construct
subroutine test_where_construct(mask, m2, x, y)
  logical :: mask(10), m2(10)
  real :: x(10), y(10)
  x = 0.0
  ! Closing braces of hlfir.where / hlfir.elsewhere appear in reverse
  ! nesting order in the printed IR. The first closing brace is the
  ! innermost ELSEWHERE, then ELSEWHERE(m2), then WHERE.

  ! CHECK: hlfir.elsewhere do
  ! CHECK: hlfir.yield
  ! CHECK: hlfir.yield
  where (mask)
    x = y
  elsewhere (m2)
    x = -y
  elsewhere
    x = 1.0
  end where
  ! CHECK: } loc("{{.*}}where-loc.f90":[[# @LINE-3]]:
  ! CHECK: } loc("{{.*}}where-loc.f90":[[# @LINE-6]]:
  ! CHECK: } loc("{{.*}}where-loc.f90":[[# @LINE-9]]:
end subroutine

! CHECK-LABEL: func.func @_QPtest_where_stmt
subroutine test_where_stmt(mask, x, y)
  logical :: mask(10)
  real :: x(10), y(10)
  ! Prior statement.
  x = 0.0
  ! CHECK: hlfir.yield
  ! CHECK: hlfir.yield
  where (mask) x = y
  ! CHECK: } loc("{{.*}}where-loc.f90":[[# @LINE-1]]:
end subroutine
