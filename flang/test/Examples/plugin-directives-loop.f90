! Check that a directive of a plugin loaded with `flang -fc1 -load` for the
! loop that follows it becomes a marker call at the start of the loop body.

! REQUIRES: plugins, examples
! XFAIL: system-aix

! RUN: rm -rf %t && mkdir -p %t
! RUN: %flang_fc1 -load %llvmshlibdir/flangDirectivesPlugin%pluginext \
! RUN:   -emit-hlfir -module-dir %t -o - %s | FileCheck %s

module m_loop
  real :: u(10)
  common /blk/ w
  real :: w
contains
  subroutine iterate(n, v)
    integer :: n, i
    real :: v(n)
    real, allocatable :: a(:)
    logical :: converged
    allocate(a(n))
    !dir$ example converge(u, v, tol=1.e-6) max_iters(100)
    do i = 1, 100
      u = u * 0.5
    end do
    !$example converge(a, /blk/, tol=-0.5)
    do while (.not. converged)
      converged = .true.
    end do
  end
end

! CHECK-LABEL: func.func @_QMm_loopPiterate(
! CHECK:         fir.do_loop
! CHECK:           fir.call @__flang_directive.example.converge({{.*}}) fastmath<contract> {fir.directive = {args = {max_iters = 100 : i64, tol = {{.*}} : f32}, keyword = "converge", prefix = "example"}}
! CHECK:         fir.call @__flang_directive.example.converge{{(\.[0-9]+)?}}({{.*}}) fastmath<contract> {fir.directive = {args = {tol = -5.000000e-01 : f32}, keyword = "converge", prefix = "example"}}
