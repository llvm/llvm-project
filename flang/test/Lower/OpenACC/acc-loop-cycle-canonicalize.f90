! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! Plan B (lorado-2856 follow-up): rewriteIfGotos canonicalises
! `if (cond) then; <stmts>; cycle; end if; <after>` into a structured
! `if (cond) then; <stmts>; else; <after>; end if`. This means an
! `acc parallel loop` over such a body, with or without `collapse(N)`,
! lowers to a STRUCTURED acc.loop (with `control(...)` and bounds on the
! op, no `unstructured` attribute), so the GPU backend can parallelise it
! the same way nvfortran does.

! Reproducer from the lorado #2856 GitLab issue.
subroutine acc_collapse_cycle_lorado(a)
  integer :: i, j, jdiag
  real(8) :: a(:,:)
  jdiag = 4
  !$acc parallel loop collapse(2) copy(a)
  do j = 1, 8
    do i = 1, 8
      if (i == jdiag) then
        a(i, j) = 0.0d0
        cycle
      end if
      a(i, j) = real(i + j, 8)
    end do
  end do
  !$acc end parallel loop
end subroutine

! CHECK-LABEL: func.func @_QPacc_collapse_cycle_lorado
! CHECK: acc.parallel combined(loop)
! Both IVs are privatised:
! CHECK: acc.private varPtr(%{{.*}} : !fir.ref<i32>) recipe(@privatization_ref_i32) -> !fir.ref<i32> {implicit = true, name = "j"}
! CHECK: acc.private varPtr(%{{.*}} : !fir.ref<i32>) recipe(@privatization_ref_i32) -> !fir.ref<i32> {implicit = true, name = "i"}
! Structured acc.loop: bounds are on the op (control(...) clause) for both IVs.
! CHECK: acc.loop combined(parallel) private(%{{.*}}, %{{.*}} : !fir.ref<i32>, !fir.ref<i32>) control(%{{.*}} : i32, %{{.*}} : i32) = (%{{.*}}, %{{.*}} : i32, i32) to (%{{.*}}, %{{.*}} : i32, i32) step (%{{.*}}, %{{.*}} : i32, i32)
! Body uses fir.if/else for the canonicalised cycle (no cf.cond_br backedges).
! CHECK: fir.if %{{.*}} {
! CHECK:   hlfir.assign %{{cst.*}} to %{{.*}} : f64, !fir.ref<f64>
! CHECK: } else {
! CHECK:   hlfir.assign %{{.*}} to %{{.*}} : f64, !fir.ref<f64>
! CHECK: }
! Trailing attributes carry collapse and inclusiveUpperbound but NOT unstructured:
! CHECK: } attributes {collapse = [2], collapseDeviceType = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true, true>, independent = [#acc.device_type<none>]}
! CHECK-NOT: unstructured

! Same shape, single-level (no collapse): exercises the rewrite for an
! unadorned `acc parallel loop` body too.
subroutine acc_loop_cycle_canonicalize(a)
  integer :: i, jdiag
  real :: a(:)
  jdiag = 4
  !$acc parallel loop
  do i = 1, 8
    if (i == jdiag) then
      a(i) = 0.0
      cycle
    end if
    a(i) = real(i)
  end do
  !$acc end parallel loop
end subroutine

! CHECK-LABEL: func.func @_QPacc_loop_cycle_canonicalize
! CHECK: acc.loop combined(parallel) private(%{{.*}} : !fir.ref<i32>) control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32)
! CHECK: fir.if
! CHECK: } else {
! CHECK: } attributes {{{.*}}independent = [#acc.device_type<none>]{{.*}}}
! CHECK-NOT: unstructured

! Negative test: an existing ELSE branch blocks the rewrite. The loop
! stays unstructured (acc.loop with `unstructured` attribute, no `control`).
subroutine acc_loop_cycle_existing_else_unchanged(a)
  integer :: i, jdiag
  real :: a(:)
  jdiag = 4
  !$acc parallel loop
  do i = 1, 8
    if (i == jdiag) then
      a(i) = 0.0
    else
      a(i) = -1.0
      cycle
    end if
    a(i) = real(i)
  end do
  !$acc end parallel loop
end subroutine

! CHECK-LABEL: func.func @_QPacc_loop_cycle_existing_else_unchanged
! No control(...) on acc.loop; trailing attributes contain `unstructured`.
! CHECK: acc.loop combined(parallel) private(%{{.*}} : !fir.ref<i32>) {
! CHECK: } attributes {{{.*}}unstructured}
