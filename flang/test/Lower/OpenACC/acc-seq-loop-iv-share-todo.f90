! Desired behavior that the current lowering does not produce yet.
!
! TODO:
! - Sibling seq loops write one private on the parent. After `do k = 1, n`
!   the value is n + 1, and both loops update that private.
! - A finished structured loop leaves one past the end. `do k = 1, 3` leaves
!   4. A zero-trip loop leaves the lower bound: after `k = 5` and
!   `do k = 1, 0`, the value is 1.
! - collapse(2) puts only the seq loop's own DO variable on the parent. The
!   collapsed index stays on the collapsed loop and is not visible after it.
! - device_type(nvidia) seq, device_type(default) seq, and device_type(*) seq
!   leave the induction variable on the parent. `do k = 1, 3` leaves 4.
! - When the parent is serial or parallel, the induction variable is private
!   on that construct. The read inside it sees the loop result, and the host
!   variable stays unchanged.
! - kernels copies the induction variable out, so the host variable becomes
!   the loop result.
! - An orphan seq loop is a host DO. The host variable is one past the end,
!   and there is no acc.loop.
! - reduction(+:s) on a seq loop is carried by the enclosing gang loop. An
!   index that is not used is not privatized.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s
! XFAIL: *

! -----------------------------------------------------------------------------
! Both seq loops write the gang loop's k. The value after `do k = 1, n` is
! n + 1.
! -----------------------------------------------------------------------------
subroutine siblings(a, n, kout)
  integer :: n, i, k, kout
  real :: a(n)
  !$acc parallel copy(kout)
  !$acc loop gang
  do i = 1, n
    !$acc loop seq
    do k = 1, 3
      a(i) = a(i) + k
    end do
    !$acc loop seq
    do k = 1, n
      a(i) = a(i) + k
    end do
    kout = k
  end do
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPsiblings
! CHECK: acc.private {{.*}} name("k")
! CHECK: acc.loop gang private({{.*}}, {{.*}} : !fir.ref<i32>, !fir.ref<i32>) control(%{{.*}} : i32) =
! CHECK: %[[KDECL:[0-9]+]]:2 = hlfir.declare %{{[0-9]+}} uniq_name("_QFsiblingsEk")
! CHECK: acc.loop control(%{{.*}} : i32) =
! CHECK: fir.store %{{.*}} to %[[KDECL]]#0 : !fir.ref<i32>
! CHECK: acc.loop control(%{{.*}} : i32) =
! CHECK: fir.store %{{.*}} to %[[KDECL]]#0 : !fir.ref<i32>
! CHECK: } inclusiveUpperbound(array<i1: true>) seq
! CHECK: %[[N:[0-9]+]] = fir.load %{{[0-9]+}}#0 : !fir.ref<i32>
! CHECK: %[[ONE:.*]] = arith.constant 1 : i32
! CHECK: %[[NEXT:.*]] = arith.addi %[[N]], %[[ONE]] : i32
! CHECK: fir.store %[[NEXT]] to %[[KDECL]]#0 : !fir.ref<i32>
! CHECK: fir.load %[[KDECL]]#0 : !fir.ref<i32>

! -----------------------------------------------------------------------------
! `do k = 1, 3` leaves 4. `do k = 1, 0` leaves 1, the lower bound.
! -----------------------------------------------------------------------------
subroutine final_value(a, n, kdone, kzero)
  integer :: n, i, k, kdone, kzero
  real :: a(n)
  !$acc parallel copy(kdone, kzero)
  !$acc loop gang
  do i = 1, n
    !$acc loop seq
    do k = 1, 3
      a(i) = a(i) + k
    end do
    kdone = k
    k = 5
    !$acc loop seq
    do k = 1, 0
      a(i) = a(i) + k
    end do
    kzero = k
  end do
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPfinal_value
! CHECK: %[[KDECL:[0-9]+]]:2 = hlfir.declare %{{[0-9]+}} uniq_name("_QFfinal_valueEk")
! CHECK: } inclusiveUpperbound(array<i1: true>) seq
! CHECK: %[[C4:.*]] = arith.constant 4 : i32
! CHECK: fir.store %[[C4]] to %[[KDECL]]#0 : !fir.ref<i32>
! CHECK: fir.load %[[KDECL]]#0 : !fir.ref<i32>
! CHECK: } inclusiveUpperbound(array<i1: true>) seq
! CHECK: %[[C1:.*]] = arith.constant 1 : i32
! CHECK: fir.store %[[C1]] to %[[KDECL]]#0 : !fir.ref<i32>
! CHECK: fir.load %[[KDECL]]#0 : !fir.ref<i32>

! -----------------------------------------------------------------------------
! Only j, the seq loop's own DO variable, is private on the parent. k stays
! on the collapsed loop.
! -----------------------------------------------------------------------------
subroutine collapse_seq(a, n, jout, kout)
  integer :: n, i, j, k, jout, kout
  real :: a(n)
  !$acc parallel copy(jout, kout)
  !$acc loop gang
  do i = 1, n
    !$acc loop seq collapse(2)
    do j = 1, 3
      do k = 1, 4
        a(i) = a(i) + j + k
      end do
    end do
    jout = j
    kout = k
  end do
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPcollapse_seq
! CHECK: acc.private {{.*}} name("j")
! CHECK-NOT: acc.private {{.*}} name("k")
! CHECK: acc.loop gang private({{.*}}, {{.*}} : !fir.ref<i32>, !fir.ref<i32>) control(%{{.*}} : i32) =
! CHECK: %[[JDECL:[0-9]+]]:2 = hlfir.declare %{{[0-9]+}} uniq_name("_QFcollapse_seqEj")
! CHECK: acc.private {{.*}} name("k")
! CHECK: acc.loop private({{.*}} : !fir.ref<i32>) control(%{{.*}} : i32, %{{.*}} : i32) =
! CHECK: fir.load %[[JDECL]]#0 : !fir.ref<i32>

! -----------------------------------------------------------------------------
! seq for a named device type is shared. `do k = 1, 3` leaves 4.
! -----------------------------------------------------------------------------
subroutine dtype_nvidia_seq(a, n, kout)
  integer :: n, i, k, kout
  real :: a(n)
  !$acc parallel copy(kout)
  !$acc loop gang
  do i = 1, n
    !$acc loop device_type(nvidia) seq
    do k = 1, 3
      a(i) = a(i) + k
    end do
    kout = k
  end do
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPdtype_nvidia_seq
! CHECK: acc.private {{.*}} name("k")
! CHECK: acc.loop gang private({{.*}}, {{.*}} : !fir.ref<i32>, !fir.ref<i32>) control(%{{.*}} : i32) =
! CHECK: %[[KDECL:[0-9]+]]:2 = hlfir.declare %{{[0-9]+}} uniq_name("_QFdtype_nvidia_seqEk")
! CHECK: acc.loop control(%{{.*}} : i32) =
! CHECK: fir.load %[[KDECL]]#0 : !fir.ref<i32>

subroutine dtype_default_seq(a, n, kout)
  integer :: n, i, k, kout
  real :: a(n)
  !$acc parallel copy(kout)
  !$acc loop gang
  do i = 1, n
    !$acc loop device_type(default) seq
    do k = 1, 3
      a(i) = a(i) + k
    end do
    kout = k
  end do
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPdtype_default_seq
! CHECK: acc.private {{.*}} name("k")
! CHECK: acc.loop gang private({{.*}}, {{.*}} : !fir.ref<i32>, !fir.ref<i32>) control(%{{.*}} : i32) =
! CHECK: hlfir.declare %{{[0-9]+}} uniq_name("_QFdtype_default_seqEk")
! CHECK: acc.loop control(%{{.*}} : i32) =

subroutine dtype_star_seq(a, n, kout)
  integer :: n, i, k, kout
  real :: a(n)
  !$acc parallel copy(kout)
  !$acc loop gang
  do i = 1, n
    !$acc loop device_type(*) seq
    do k = 1, 3
      a(i) = a(i) + k
    end do
    kout = k
  end do
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPdtype_star_seq
! CHECK: acc.private {{.*}} name("k")
! CHECK: acc.loop gang private({{.*}}, {{.*}} : !fir.ref<i32>, !fir.ref<i32>) control(%{{.*}} : i32) =
! CHECK: hlfir.declare %{{[0-9]+}} uniq_name("_QFdtype_star_seqEk")
! CHECK: acc.loop control(%{{.*}} : i32) =

! -----------------------------------------------------------------------------
! The construct, not only the loop, holds k. The read inside it sees the
! loop result. The host variable is unchanged, except under kernels, which
! copies k out.
! -----------------------------------------------------------------------------
subroutine parent_serial(a, n, kout)
  integer :: n, k, kout
  real :: a(n)
  !$acc serial copy(kout)
  !$acc loop seq
  do k = 1, 3
    a(k) = a(k) + 1.0
  end do
  kout = k
  !$acc end serial
end subroutine

! CHECK-LABEL: func.func @_QPparent_serial
! CHECK: acc.private {{.*}} name("k")
! CHECK: acc.serial {{.*}} private({{.*}} : !fir.ref<i32>)
! CHECK: %[[KDECL:[0-9]+]]:2 = hlfir.declare %{{[0-9]+}} uniq_name("_QFparent_serialEk")
! CHECK: acc.loop
! CHECK: } inclusiveUpperbound(array<i1: true>) seq
! CHECK: fir.load %[[KDECL]]#0 : !fir.ref<i32>

subroutine parent_parallel(a, n, kout)
  integer :: n, k, kout
  real :: a(n)
  !$acc parallel copy(kout)
  !$acc loop seq
  do k = 1, 3
    a(k) = a(k) + 1.0
  end do
  kout = k
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPparent_parallel
! CHECK: acc.private {{.*}} name("k")
! CHECK: acc.parallel {{.*}} private({{.*}} : !fir.ref<i32>)
! CHECK: %[[KDECL:[0-9]+]]:2 = hlfir.declare %{{[0-9]+}} uniq_name("_QFparent_parallelEk")
! CHECK: } inclusiveUpperbound(array<i1: true>) seq
! CHECK: fir.load %[[KDECL]]#0 : !fir.ref<i32>

subroutine parent_kernels(a, n, kout)
  integer :: n, k, kout
  real :: a(n)
  !$acc kernels copy(kout, k)
  !$acc loop seq
  do k = 1, 3
    a(k) = a(k) + 1.0
  end do
  kout = k
  !$acc end kernels
end subroutine

! CHECK-LABEL: func.func @_QPparent_kernels
! CHECK: acc.copyout {{.*}} name("k")
! CHECK: %[[KDECL:[0-9]+]]:2 = hlfir.declare %{{[0-9]+}} uniq_name("_QFparent_kernelsEk")
! CHECK: } inclusiveUpperbound(array<i1: true>) seq
! CHECK: fir.load %[[KDECL]]#0 : !fir.ref<i32>

! -----------------------------------------------------------------------------
! No accelerator loop. The host DO leaves k one past the end.
! -----------------------------------------------------------------------------
subroutine orphan_seq(a, n, kout)
  integer :: n, k, kout
  real :: a(n)
  !$acc loop seq
  do k = 1, 3
    a(k) = a(k) + 1.0
  end do
  kout = k
end subroutine

! CHECK-LABEL: func.func @_QPorphan_seq
! CHECK: fir.do_loop
! CHECK: %[[C4:.*]] = arith.constant 4 : i32
! CHECK: fir.store %[[C4]] to %{{[0-9]+}}#0 : !fir.ref<i32>
! CHECK-NOT: acc.loop

! -----------------------------------------------------------------------------
! The reduction is on the gang loop. The unused index is not privatized.
! -----------------------------------------------------------------------------
subroutine reduction_other(a, n, s)
  integer :: n, i, k
  real :: a(n), s
  !$acc parallel copy(s)
  !$acc loop gang
  do i = 1, n
    !$acc loop seq reduction(+:s)
    do k = 1, n
      s = s + a(i)
    end do
  end do
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPreduction_other
! CHECK: acc.loop gang {{.*}} reduction({{.*}} : !fir.ref<f32>) control(%{{.*}} : i32) =
! CHECK-NOT: acc.private {{.*}} name("k")
! CHECK: acc.loop
