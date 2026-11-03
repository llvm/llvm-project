! The induction variable of an explicit `!$acc loop seq` is privatized on the
! closest enclosing acc.loop. A seq loop whose parent is not an acc.loop keeps
! that private. `seq` counts only when it applies to the default device type
! (no device_type clause yet). `-openacc-share-seq-loop-iv=false` keeps the
! private on the seq loop.
!
! Desired behavior the current lowering does not produce yet is in
! acc-seq-loop-iv-share-todo.f90.
! DO CONCURRENT index nesting is covered by acc-loop-nested-in-do-concurrent.f90.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s
! RUN: bbc -fopenacc -openacc-share-seq-loop-iv=false -emit-hlfir %s -o - | FileCheck %s --check-prefix=CHECK-OFF

! -----------------------------------------------------------------------------
! Structured seq loop. The read after it uses the private on the parent.
! -----------------------------------------------------------------------------
subroutine structured_after(a, n, kout)
  integer :: n, i, k, kout
  real :: a(n)
  !$acc parallel
  !$acc loop gang
  do i = 1, n
    !$acc loop seq
    do k = 1, n
      a(i) = a(i) + k
    end do
    kout = k
  end do
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPstructured_after
! CHECK: acc.private {{.*}} name("k")
! CHECK: acc.loop gang private({{.*}}, {{.*}} : !fir.ref<i32>, !fir.ref<i32>) control(%{{.*}} : i32) =
! CHECK: %[[KDECL:[0-9]+]]:2 = hlfir.declare %{{[0-9]+}} uniq_name("_QFstructured_afterEk")
! CHECK: acc.loop control(%{{.*}} : i32) =
! CHECK: } inclusiveUpperbound(array<i1: true>) seq
! CHECK: fir.load %[[KDECL]]#0 : !fir.ref<i32>
! CHECK-OFF-LABEL: func.func @_QPstructured_after
! CHECK-OFF: acc.loop gang private(%{{[0-9]+}} : !fir.ref<i32>) control(%{{.*}} : i32) =
! CHECK-OFF: acc.loop private(%{{[0-9]+}} : !fir.ref<i32>) control(%{{.*}} : i32) =
! CHECK-OFF: } inclusiveUpperbound(array<i1: true>) seq

! -----------------------------------------------------------------------------
! A store of the same variable before the seq loop is retargeted at that
! private, and so is the read after the loop.
! -----------------------------------------------------------------------------
subroutine store_before(a, n, kout)
  integer :: n, i, k, kout
  real :: a(n)
  !$acc parallel
  !$acc loop gang
  do i = 1, n
    k = 1
    !$acc loop seq
    do k = 1, n
      a(i) = a(i) + k
    end do
    kout = k
  end do
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPstore_before
! CHECK: acc.private {{.*}} name("k")
! CHECK: acc.loop gang private({{.*}}, {{.*}} : !fir.ref<i32>, !fir.ref<i32>) control(%{{.*}} : i32) =
! CHECK: %[[KDECL:[0-9]+]]:2 = hlfir.declare %{{[0-9]+}} uniq_name("_QFstore_beforeEk")
! CHECK: hlfir.assign %c1_i32{{[_0-9]*}} to %[[KDECL]]#0 : i32, !fir.ref<i32>
! CHECK: acc.loop control(%{{.*}} : i32) =
! CHECK: fir.load %[[KDECL]]#0 : !fir.ref<i32>

! -----------------------------------------------------------------------------
! Goto leaves the seq loop early. The read after it still uses the parent
! private. The outer loop is unstructured, so the seq loop is not a direct
! child of the parent operation.
! -----------------------------------------------------------------------------
subroutine unstructured_parent(a, n, kout)
  integer :: n, i, k, kout
  real :: a(n)
  !$acc parallel
  !$acc loop gang
  do 100 i = 1, n
    if (a(i) < 0.0) goto 200
    !$acc loop seq
    do k = 1, n
      a(i) = a(i) + k
    end do
    kout = k
100 continue
200 continue
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPunstructured_parent
! CHECK: acc.private {{.*}} name("k")
! CHECK: acc.loop gang private({{.*}}, {{.*}} : !fir.ref<i32>, !fir.ref<i32>) {
! CHECK: %[[KDECL:[0-9]+]]:2 = hlfir.declare %{{[0-9]+}} uniq_name("_QFunstructured_parentEk")
! CHECK: acc.loop control(%{{.*}} : i32) =
! CHECK: } inclusiveUpperbound(array<i1: true>) seq
! CHECK: fir.load %[[KDECL]]#0 : !fir.ref<i32>

! -----------------------------------------------------------------------------
! The seq loop itself is unstructured. Its IV is still private on the parent,
! and the read after the loop uses that private.
! -----------------------------------------------------------------------------
subroutine unstructured_seq(a, n, kout)
  integer :: n, i, k, kout
  real :: a(n)
  !$acc parallel
  !$acc loop gang
  do i = 1, n
    !$acc loop seq
    do 10 k = 1, n
      if (a(i) > real(k)) goto 20
10  continue
20  kout = k
  end do
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPunstructured_seq
! CHECK: acc.private {{.*}} name("k")
! CHECK: acc.loop gang private({{.*}}, {{.*}} : !fir.ref<i32>, !fir.ref<i32>) control(%{{.*}} : i32) =
! CHECK: %[[KDECL:[0-9]+]]:2 = hlfir.declare %{{[0-9]+}} uniq_name("_QFunstructured_seqEk")
! CHECK: acc.loop {
! CHECK: } seq unstructured
! CHECK: fir.load %[[KDECL]]#0 : !fir.ref<i32>

! -----------------------------------------------------------------------------
! Closest parent, not the outermost loop.
! -----------------------------------------------------------------------------
subroutine closest_parent(a, n, kout)
  integer :: n, i, j, k, kout
  real :: a(n)
  !$acc parallel
  !$acc loop gang
  do i = 1, n
    !$acc loop vector
    do j = 1, n
      !$acc loop seq
      do k = 1, n
        a(i) = a(i) + k
      end do
      kout = k
    end do
  end do
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPclosest_parent
! CHECK: acc.loop gang private(%{{[0-9]+}} : !fir.ref<i32>) control(%{{.*}} : i32) =
! CHECK: acc.private {{.*}} name("k")
! CHECK: acc.loop vector private({{.*}}, {{.*}} : !fir.ref<i32>, !fir.ref<i32>) control(%{{.*}} : i32) =
! CHECK: %[[KDECL:[0-9]+]]:2 = hlfir.declare %{{[0-9]+}} uniq_name("_QFclosest_parentEk")
! CHECK: acc.loop control(%{{.*}} : i32) =
! CHECK: fir.load %[[KDECL]]#0 : !fir.ref<i32>

! -----------------------------------------------------------------------------
! A seq loop nested in another seq loop. Each IV goes one level up.
! -----------------------------------------------------------------------------
subroutine seq_in_seq(a, n)
  integer :: n, i, j, k
  real :: a(n)
  !$acc parallel
  !$acc loop gang
  do i = 1, n
    !$acc loop seq
    do j = 1, n
      !$acc loop seq
      do k = 1, n
        a(i) = a(i) + j + k
      end do
    end do
  end do
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPseq_in_seq
! CHECK: acc.private {{.*}} name("j")
! CHECK: acc.loop gang private({{.*}}, {{.*}} : !fir.ref<i32>, !fir.ref<i32>) control(%{{.*}} : i32) =
! CHECK: hlfir.declare %{{[0-9]+}} uniq_name("_QFseq_in_seqEj")
! CHECK: acc.private {{.*}} name("k")
! CHECK: acc.loop private({{.*}} : !fir.ref<i32>) control(%{{.*}} : i32) =
! CHECK: hlfir.declare %{{[0-9]+}} uniq_name("_QFseq_in_seqEk")
! CHECK: acc.loop control(%{{.*}} : i32) =

! -----------------------------------------------------------------------------
! Two seq loops in one parent, different induction variables.
! -----------------------------------------------------------------------------
subroutine two_ivs(a, n, jout, kout)
  integer :: n, i, j, k, jout, kout
  real :: a(n)
  !$acc parallel
  !$acc loop gang
  do i = 1, n
    !$acc loop seq
    do j = 1, n
      a(i) = a(i) + j
    end do
    jout = j
    !$acc loop seq
    do k = 1, n
      a(i) = a(i) + k
    end do
    kout = k
  end do
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtwo_ivs
! CHECK: acc.private {{.*}} name("j")
! CHECK: acc.private {{.*}} name("k")
! CHECK: acc.loop gang private({{.*}}, {{.*}}, {{.*}} : !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>) control(%{{.*}} : i32) =
! CHECK: %[[KDECL:[0-9]+]]:2 = hlfir.declare %{{[0-9]+}} uniq_name("_QFtwo_ivsEk")
! CHECK: %[[JDECL:[0-9]+]]:2 = hlfir.declare %{{[0-9]+}} uniq_name("_QFtwo_ivsEj")
! CHECK: acc.loop control(%{{.*}} : i32) =
! CHECK: fir.load %[[JDECL]]#0 : !fir.ref<i32>
! CHECK: acc.loop control(%{{.*}} : i32) =
! CHECK: fir.load %[[KDECL]]#0 : !fir.ref<i32>

! -----------------------------------------------------------------------------
! private(k) is already on the parent. The seq loop writes that copy.
! -----------------------------------------------------------------------------
subroutine private_on_parent(a, n, kout)
  integer :: n, i, k, kout
  real :: a(n)
  !$acc parallel
  !$acc loop gang private(k)
  do i = 1, n
    !$acc loop seq
    do k = 1, 3
      a(i) = a(i) + k
    end do
    kout = k
  end do
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPprivate_on_parent
! CHECK: acc.private {{.*}} name("k")
! CHECK: acc.loop gang private({{.*}}, {{.*}} : !fir.ref<i32>, !fir.ref<i32>) control(%{{.*}} : i32) =
! CHECK: %[[KDECL:[0-9]+]]:2 = hlfir.declare %{{[0-9]+}} uniq_name("_QFprivate_on_parentEk")
! CHECK: acc.loop control(%{{.*}} : i32) =
! CHECK: fir.store %{{.*}} to %[[KDECL]]#0 : !fir.ref<i32>
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
! integer(8) induction variable.
! -----------------------------------------------------------------------------
subroutine i64_iv(a, n, kout)
  integer :: n, i
  integer(8) :: k, kout
  real :: a(n)
  !$acc parallel
  !$acc loop gang
  do i = 1, n
    !$acc loop seq
    do k = 1_8, 3_8
      a(i) = a(i) + real(k)
    end do
    kout = k
  end do
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPi64_iv
! CHECK: acc.private {{.*}} name("k") -> !fir.ref<i64>
! CHECK: acc.loop gang private({{.*}}, {{.*}} : !fir.ref<i32>, !fir.ref<i64>) control(%{{.*}} : i32) =
! CHECK: %[[KDECL:[0-9]+]]:2 = hlfir.declare %{{[0-9]+}} uniq_name("_QFi64_ivEk")
! CHECK: acc.loop control(%{{.*}} : i64) =
! CHECK: fir.load %[[KDECL]]#0 : !fir.ref<i64>

! -----------------------------------------------------------------------------
! A combined seq loop is not moved onto a parent.
! -----------------------------------------------------------------------------
subroutine combined_seq(a, n)
  integer :: n, k
  real :: a(n)
  !$acc parallel loop seq
  do k = 1, n
    a(k) = a(k) + 1.0
  end do
end subroutine

! CHECK-LABEL: func.func @_QPcombined_seq
! CHECK: acc.parallel combined(loop)
! CHECK: acc.loop combined(parallel) private(%{{[0-9]+}} : !fir.ref<i32>) control(%{{.*}} : i32) =
! CHECK: } inclusiveUpperbound(array<i1: true>) seq

! -----------------------------------------------------------------------------
! seq before a device_type clause applies to the default device type and is
! moved. seq that applies only to another device type is in
! acc-seq-loop-iv-share-todo.f90.
! -----------------------------------------------------------------------------
subroutine seq_then_dtype(a, n)
  integer :: n, i, k
  real :: a(n)
  !$acc parallel
  !$acc loop gang
  do i = 1, n
    !$acc loop seq device_type(nvidia) collapse(1)
    do k = 1, n
      a(i) = a(i) + k
    end do
  end do
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPseq_then_dtype
! CHECK: acc.private {{.*}} name("k")
! CHECK: acc.loop gang private({{.*}}, {{.*}} : !fir.ref<i32>, !fir.ref<i32>) control(%{{.*}} : i32) =
! CHECK: hlfir.declare %{{[0-9]+}} uniq_name("_QFseq_then_dtypeEk")
! CHECK: acc.loop control(%{{.*}} : i32) =
! CHECK: collapse([1]) collapseDeviceType([#acc.device_type<nvidia>]) seq

! -----------------------------------------------------------------------------
! The same move inside an acc routine seq.
! -----------------------------------------------------------------------------
subroutine in_routine(a, n, kout)
  integer :: n, i, k, kout
  real :: a(n)
  !$acc routine seq
  !$acc loop gang
  do i = 1, n
    !$acc loop seq
    do k = 1, n
      a(i) = a(i) + k
    end do
    kout = k
  end do
end subroutine

! CHECK-LABEL: func.func @_QPin_routine
! CHECK: acc.private {{.*}} name("k")
! CHECK: acc.loop gang private({{.*}}, {{.*}} : !fir.ref<i32>, !fir.ref<i32>) control(%{{.*}} : i32) =
! CHECK: %[[KDECL:[0-9]+]]:2 = hlfir.declare %{{[0-9]+}} uniq_name("_QFin_routineEk")
! CHECK: acc.loop control(%{{.*}} : i32) =
! CHECK: fir.load %[[KDECL]]#0 : !fir.ref<i32>
