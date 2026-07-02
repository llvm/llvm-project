! Test that lastprivate(conditional:) correctly identifies the enclosing
! omp.parallel even when the worksharing construct is inside a Fortran IF
! block (which lowers to fir.if).  The struct alloca must be placed before
! the omp.parallel — NOT treated as orphaned.

! RUN: bbc -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s

subroutine test_nested_if(n, a, x, flag)
  implicit none
  integer, intent(in) :: n, flag
  integer, intent(in) :: a(n)
  integer, intent(inout) :: x
  integer :: k

  !$omp parallel do lastprivate(conditional: x)
  do k = 1, n
    if (a(k) < 150) then
      x = k + 1
    end if
  end do
  !$omp end parallel do
end subroutine

! -- The struct is stack-allocated (fir.alloca), not a global -----------------
! CHECK-LABEL: func.func @_QPtest_nested_if
! CHECK:         fir.alloca !fir.type<_lp_cond_t.{{l[0-9]+\.[0-9]+}}{x:i32,$x:i64}>
! CHECK-NOT:     fir.address_of(@_lp_cond_global

! -- No nesting guard emitted (this is not orphaned) -------------------------
! CHECK-NOT:     fir.call @omp_get_level_

! -- omp.parallel with the struct as reduction --------------------------------
! CHECK:         omp.parallel
! CHECK:           omp.wsloop
! CHECK-SAME:        reduction(byref @lp_cond_byref_rec__lp_cond_t
