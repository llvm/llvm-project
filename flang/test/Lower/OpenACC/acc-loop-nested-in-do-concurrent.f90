! Test lowering of an explicit `!$acc loop` construct nested inside a
! DO CONCURRENT that is itself associated with an `!$acc loop`, where the
! nested loop body references an index of the outer DO CONCURRENT.
!
! A DO CONCURRENT index-name is a construct entity in the construct's own
! scope, distinct from a like-named variable in the enclosing subprogram
! scope. Inside the nested acc loop, the reference to the outer index resolves
! to that construct entity; lowering must bind it to the same privatized
! storage as the loop control so the reference can be lowered.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPnested_acc_loop_in_do_concurrent
subroutine nested_acc_loop_in_do_concurrent(a, n)
  integer :: n, i, l
  real(8) :: a(n)
  !$acc parallel
  !$acc loop
  do concurrent(i = 1:n)
    !$acc loop seq
    do l = 1, n
      a(i) = a(i) + real(l, 8)
    end do
  end do
  !$acc end parallel
end subroutine

! The outer acc.loop privatizes the DO CONCURRENT index `i`.
! CHECK: acc.loop private({{.*}}) control(%[[IARG:[a-z0-9_]+]] : i32)
! CHECK: %[[IDECL:[0-9]+]]:2 = hlfir.declare %{{[a-z0-9_]+}} {uniq_name = "_QFnested_acc_loop_in_do_concurrentEi"}
! CHECK: fir.store %[[IARG]] to %[[IDECL]]#0 : !fir.ref<i32>

! The nested acc.loop body reads the outer index from that same storage.
! CHECK: acc.loop private({{.*}}) control(%{{[a-z0-9_]+}} : i32)
! CHECK: %[[ILOAD:[0-9]+]] = fir.load %[[IDECL]]#0 : !fir.ref<i32>
! CHECK: %[[ICONV:[0-9]+]] = fir.convert %[[ILOAD]] : (i32) -> i64
! CHECK: hlfir.designate %{{.*}} (%[[ICONV]])

! -----------------------------------------------------------------------------
! Multi-index DO CONCURRENT; the nested acc loop references both indices.
! -----------------------------------------------------------------------------
! CHECK-LABEL: func.func @_QPmulti_index
subroutine multi_index(a, n, m)
  integer :: n, m, i, j, l
  real(8) :: a(n, m)
  !$acc parallel
  !$acc loop
  do concurrent(i = 1:n, j = 1:m)
    !$acc loop seq
    do l = 1, n
      a(i, j) = a(i, j) + real(l, 8)
    end do
  end do
  !$acc end parallel
end subroutine

! The outer acc.loop privatizes both indices; the nested loop reads both.
! CHECK: acc.loop private({{.*}}) control(%{{[a-z0-9_]+}} : i32, %{{[a-z0-9_]+}} : i32)
! CHECK: %[[MI:[0-9]+]]:2 = hlfir.declare %{{[a-z0-9_]+}} {uniq_name = "_QFmulti_indexEi"}
! CHECK: %[[MJ:[0-9]+]]:2 = hlfir.declare %{{[a-z0-9_]+}} {uniq_name = "_QFmulti_indexEj"}
! CHECK: acc.loop private({{.*}}) control(%{{[a-z0-9_]+}} : i32)
! CHECK: fir.load %[[MI]]#0 : !fir.ref<i32>
! CHECK: fir.load %[[MJ]]#0 : !fir.ref<i32>

! -----------------------------------------------------------------------------
! The nested acc loop's BOUNDS (not just its body) reference the outer index.
! -----------------------------------------------------------------------------
! CHECK-LABEL: func.func @_QPouter_index_in_bounds
subroutine outer_index_in_bounds(a, n)
  integer :: n, i, l
  real(8) :: a(n)
  !$acc parallel
  !$acc loop
  do concurrent(i = 1:n)
    !$acc loop seq
    do l = i, n
      a(l) = a(l) + 1.0_8
    end do
  end do
  !$acc end parallel
end subroutine

! The nested loop's lower bound is loaded from the outer index's privatized
! storage (the declare inside the outer acc.loop, not the host-level one).
! CHECK: acc.loop private({{.*}}) control(%{{[a-z0-9_]+}} : i32)
! CHECK: %[[BI:[0-9]+]]:2 = hlfir.declare %{{[a-z0-9_]+}} {uniq_name = "_QFouter_index_in_boundsEi"}
! CHECK: %[[LB:[0-9]+]] = fir.load %[[BI]]#0 : !fir.ref<i32>
! CHECK: acc.loop private({{.*}}) control(%{{[a-z0-9_]+}} : i32) = (%[[LB]] : i32)

! -----------------------------------------------------------------------------
! Same nested shape under acc serial, acc kernels, and combined parallel loop.
! -----------------------------------------------------------------------------
! CHECK-LABEL: func.func @_QPnested_in_serial
subroutine nested_in_serial(a, n)
  integer :: n, i, l
  real(8) :: a(n)
  !$acc serial
  !$acc loop
  do concurrent(i = 1:n)
    !$acc loop seq
    do l = 1, n
      a(i) = a(i) + real(l, 8)
    end do
  end do
  !$acc end serial
end subroutine

! CHECK: acc.serial
! CHECK: %[[SI:[0-9]+]]:2 = hlfir.declare %{{[a-z0-9_]+}} {uniq_name = "_QFnested_in_serialEi"}
! CHECK: acc.loop
! CHECK: fir.load %[[SI]]#0 : !fir.ref<i32>

! CHECK-LABEL: func.func @_QPnested_in_kernels
subroutine nested_in_kernels(a, n)
  integer :: n, i, l
  real(8) :: a(n)
  !$acc kernels
  !$acc loop
  do concurrent(i = 1:n)
    !$acc loop seq
    do l = 1, n
      a(i) = a(i) + real(l, 8)
    end do
  end do
  !$acc end kernels
end subroutine

! CHECK: acc.kernels
! CHECK: %[[KI:[0-9]+]]:2 = hlfir.declare %{{[a-z0-9_]+}} {uniq_name = "_QFnested_in_kernelsEi"}
! CHECK: acc.loop
! CHECK: fir.load %[[KI]]#0 : !fir.ref<i32>

! CHECK-LABEL: func.func @_QPnested_in_combined
subroutine nested_in_combined(a, n)
  integer :: n, i, l
  real(8) :: a(n)
  !$acc parallel loop
  do concurrent(i = 1:n)
    !$acc loop seq
    do l = 1, n
      a(i) = a(i) + real(l, 8)
    end do
  end do
end subroutine

! CHECK: acc.parallel combined(loop)
! CHECK: %[[CI:[0-9]+]]:2 = hlfir.declare %{{[a-z0-9_]+}} {uniq_name = "_QFnested_in_combinedEi"}
! CHECK: acc.loop
! CHECK: fir.load %[[CI]]#0 : !fir.ref<i32>

! -----------------------------------------------------------------------------
! Two sibling nested acc loops inside one DO CONCURRENT. The first nested loop
! is a shadowing DO CONCURRENT over the same name `i`; the second references
! the OUTER index `i`. That reference must resolve to the outer loop's
! privatized storage (which dominates both siblings), not to anything created
! inside the first sibling's region.
! -----------------------------------------------------------------------------
! CHECK-LABEL: func.func @_QPtwo_sibling_nested
subroutine two_sibling_nested(a, b, n)
  integer :: n, i, l
  real(8) :: a(n, n), b(n)
  !$acc parallel
  !$acc loop
  do concurrent(i = 1:n)
    !$acc loop seq
    do concurrent(i = 1:n)
      a(i, i) = 1.0_8
    end do
    !$acc loop seq
    do l = 1, n
      b(i) = b(i) + real(l, 8)   ! outer i, referenced from 2nd nested region
    end do
  end do
  !$acc end parallel
end subroutine

! Outer acc.loop privatizes i; capture that (dominating) declare.
! CHECK: acc.loop private({{.*}}) control(%{{[a-z0-9_]+}} : i32)
! CHECK: %[[OUTERI:[0-9]+]]:2 = hlfir.declare %{{[a-z0-9_]+}} {uniq_name = "_QFtwo_sibling_nestedEi"}
! First sibling: the shadowing DO CONCURRENT declares its own i.
! CHECK: acc.loop
! CHECK: hlfir.declare %{{[a-z0-9_]+}} {uniq_name = "_QFtwo_sibling_nestedEi"}
! Second sibling: b(i) reads the OUTER index storage, not the shadow.
! CHECK: acc.loop
! CHECK: fir.load %[[OUTERI]]#0 : !fir.ref<i32>
