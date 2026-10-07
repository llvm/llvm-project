! Check that a directive of a plugin loaded with `flang -fc1 -load` for the
! loop that follows it is attached to the loop: to its fir.do_loop or
! scf.while, or, for an unstructured loop, to the branch back to its header.

! REQUIRES: plugins, examples
! XFAIL: system-aix

! RUN: rm -rf %t && mkdir -p %t
! RUN: %flang_fc1 -load %llvmshlibdir/flangDirectivesPlugin%pluginext \
! RUN:   -emit-hlfir -module-dir %t -o - %s | FileCheck %s
! RUN: %flang_fc1 -load %llvmshlibdir/flangDirectivesPlugin%pluginext \
! RUN:   -mllvm -lower-do-while-to-scf-while -emit-hlfir -module-dir %t -o - %s \
! RUN:   | FileCheck %s --check-prefix=SCF

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
    !dir$ example converge(w)
    do i = 1, n
      if (u(i) < 0) exit
      u(i) = u(i) - 1
    end do
    !dir$ example converge(v) max_iters(3)
    do
      v = v * 0.5
      if (v(1) < 1) exit
    end do
  end
end

! The structured DO: on its fir.do_loop. A module variable is a reference to
! its global, another variable the unique name of its declaration.
! CHECK-LABEL: func.func @_QMm_loopPiterate(
! CHECK:         fir.do_loop {{.*}} attributes {fir.directives = [{args = {max_iters = 100 : i64, tol = {{.*}} : f32}, keyword = "converge", prefix = "example", variables = [@_QMm_loopEu, "_QMm_loopFiterateEv"]}]}
! An unstructured DO WHILE: on the branch back to its header.
! CHECK:       ^[[WHILE:bb[0-9]+]]:
! CHECK:         cf.cond_br %{{.*}}, ^[[WBODY:bb[0-9]+]], ^
! CHECK:       ^[[WBODY]]:
! CHECK:         cf.br ^[[WHILE]] {fir.directives = [{args = {tol = -5.000000e-01 : f32}, keyword = "converge", prefix = "example", variables = ["_QMm_loopFiterateEa", @blk_]}]}
! A DO with an EXIT, unstructured: the same, and a COMMON member is named by
! its declaration.
! CHECK:         cf.br ^{{.*}} {fir.directives = [{args = {}, keyword = "converge", prefix = "example", variables = ["_QMm_loopEw"]}]}
! A DO without loop control.
! CHECK:         cf.br ^{{.*}} {fir.directives = [{args = {max_iters = 3 : i64}, keyword = "converge", prefix = "example", variables = ["_QMm_loopFiterateEv"]}]}
! CHECK-NOT:     fir.call @__flang_directive

! A structured DO WHILE lowered to scf.while: on the scf.while.
! SCF-LABEL: func.func @_QMm_loopPiterate(
! SCF:         scf.while : () -> () {
! SCF:         } attributes {fir.directives = [{args = {tol = -5.000000e-01 : f32}, keyword = "converge", prefix = "example", variables = ["_QMm_loopFiterateEa", @blk_]}]}
