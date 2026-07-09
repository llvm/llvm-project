! Test lowering of DO CONCURRENT with locality specs inside ACC constructs.
! Per OpenACC 2.17.2:
!   - DO CONCURRENT without a loop construct in a kernels construct is
!     treated as if annotated with loop auto.
!   - DO CONCURRENT in a parallel construct or accelerator routine is
!     treated as if annotated with loop independent.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! ---------------------------------------------------------------------------
! REDUCE locality spec
! ---------------------------------------------------------------------------

! Scalar reduction in kernels region (no explicit loop → auto)
! CHECK-LABEL: func.func @_QPreduce_kernels_region
subroutine reduce_kernels_region()
  real :: a(10), s
  integer :: i
  s = 0.
  !$acc kernels
  do concurrent(i=1:10) reduce(+:s)
    s = s + a(i)
  end do
  !$acc end kernels
end subroutine

! CHECK: acc.kernels {
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%{{.*}} : !fir.ref<f32>) recipe(@reduction_add{{.*}}) -> !fir.ref<f32> {name = "s"}
! CHECK: acc.loop {{.*}}reduction(%[[RED]] : !fir.ref<f32>)
! CHECK: } attributes {auto_ = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}

! Scalar reduction in parallel region (no explicit loop → independent)
! CHECK-LABEL: func.func @_QPreduce_parallel_region
subroutine reduce_parallel_region()
  real :: a(10), s
  integer :: i
  s = 0.
  !$acc parallel
  do concurrent(i=1:10) reduce(+:s)
    s = s + a(i)
  end do
  !$acc end parallel
end subroutine

! CHECK: acc.parallel {
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%{{.*}} : !fir.ref<f32>) recipe(@reduction_add{{.*}}) -> !fir.ref<f32> {name = "s"}
! CHECK: acc.loop {{.*}}reduction(%[[RED]] : !fir.ref<f32>)
! CHECK: } attributes {inclusiveUpperbound = array<i1: true>, independent = [#acc.device_type<none>]}

! Combined kernels loop with reduce (auto)
! CHECK-LABEL: func.func @_QPreduce_kernels_loop
subroutine reduce_kernels_loop()
  real :: a(16,16), b(16,16), s
  integer :: i, j
  s = 0.
  !$acc kernels loop
  do concurrent(i=1:16, j=1:16) reduce(+:s)
    b(i,j) = a(i,j)**2
    s = s + b(i,j)
  end do
end subroutine

! CHECK: acc.kernels combined(loop)
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%{{.*}} : !fir.ref<f32>) recipe(@reduction_add{{.*}}) -> !fir.ref<f32> {name = "s"}
! CHECK: acc.loop combined(kernels) {{.*}}reduction(%[[RED]] : !fir.ref<f32>)
! CHECK: } attributes {auto_ = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true, true>}

! Combined parallel loop with reduce (independent)
! CHECK-LABEL: func.func @_QPreduce_parallel_loop
subroutine reduce_parallel_loop()
  real :: a(10), s
  integer :: i
  s = 0.
  !$acc parallel loop
  do concurrent(i=1:10) reduce(+:s)
    s = s + a(i)
  end do
end subroutine

! CHECK: acc.parallel combined(loop)
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%{{.*}} : !fir.ref<f32>) recipe(@reduction_add{{.*}}) -> !fir.ref<f32> {name = "s"}
! CHECK: acc.loop combined(parallel) {{.*}}reduction(%[[RED]] : !fir.ref<f32>)
! CHECK: } attributes {inclusiveUpperbound = array<i1: true>, independent = [#acc.device_type<none>]}

! Multiple reductions (add + multiply)
! CHECK-LABEL: func.func @_QPmulti_reduce
subroutine multi_reduce()
  real :: a(10), s, p
  integer :: i
  s = 0.
  p = 1.
  !$acc parallel loop
  do concurrent(i=1:10) reduce(+:s) reduce(*:p)
    s = s + a(i)
    p = p * a(i)
  end do
end subroutine

! CHECK: acc.parallel combined(loop)
! CHECK-DAG: acc.reduction varPtr(%{{.*}} : !fir.ref<f32>) recipe(@reduction_add{{.*}}) -> !fir.ref<f32> {name = "s"}
! CHECK-DAG: acc.reduction varPtr(%{{.*}} : !fir.ref<f32>) recipe(@reduction_mul{{.*}}) -> !fir.ref<f32> {name = "p"}
! CHECK: acc.loop {{.*}}reduction(

! Max/min reductions
! CHECK-LABEL: func.func @_QPreduce_max_min
subroutine reduce_max_min()
  real :: a(10), mx, mn
  integer :: i
  mx = -huge(mx)
  mn = huge(mn)
  !$acc kernels loop
  do concurrent(i=1:10) reduce(max:mx) reduce(min:mn)
    mx = max(mx, a(i))
    mn = min(mn, a(i))
  end do
end subroutine

! CHECK: acc.kernels combined(loop)
! CHECK-DAG: acc.reduction varPtr(%{{.*}} : !fir.ref<f32>) recipe(@reduction_max{{.*}}) -> !fir.ref<f32> {name = "mx"}
! CHECK-DAG: acc.reduction varPtr(%{{.*}} : !fir.ref<f32>) recipe(@reduction_min{{.*}}) -> !fir.ref<f32> {name = "mn"}
! CHECK: acc.loop {{.*}}reduction(

! Integer multiply reduction
! CHECK-LABEL: func.func @_QPint_reduce
subroutine int_reduce()
  integer :: a(10), prod, i
  prod = 1
  !$acc kernels loop
  do concurrent(i=1:10) reduce(*:prod)
    prod = prod * a(i)
  end do
end subroutine

! CHECK: acc.kernels combined(loop)
! CHECK: acc.reduction varPtr(%{{.*}} : !fir.ref<i32>) recipe(@reduction_mul{{.*}}) -> !fir.ref<i32> {name = "prod"}

! ---------------------------------------------------------------------------
! LOCAL locality spec → acc.private
! ---------------------------------------------------------------------------

! LOCAL in kernels region (auto)
! CHECK-LABEL: func.func @_QPlocal_kernels_region
subroutine local_kernels_region()
  real :: a(10), tmp
  integer :: i
  !$acc kernels
  do concurrent(i=1:10) local(tmp)
    tmp = a(i) * 2.0
    a(i) = tmp + 1.0
  end do
  !$acc end kernels
end subroutine

! CHECK: acc.kernels {
! CHECK: %[[PRIV:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<f32>) recipe(@privatization_ref_f32) -> !fir.ref<f32> {name = "tmp"}
! CHECK: acc.loop private(%[[PRIV]],
! CHECK: } attributes {auto_ = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}

! LOCAL in parallel region (independent)
! CHECK-LABEL: func.func @_QPlocal_parallel_region
subroutine local_parallel_region()
  real :: a(10), tmp
  integer :: i
  !$acc parallel
  do concurrent(i=1:10) local(tmp)
    tmp = a(i) * 2.0
    a(i) = tmp + 1.0
  end do
  !$acc end parallel
end subroutine

! CHECK: acc.parallel {
! CHECK: %[[PRIV:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<f32>) recipe(@privatization_ref_f32) -> !fir.ref<f32> {name = "tmp"}
! CHECK: acc.loop private(%[[PRIV]],
! CHECK: } attributes {inclusiveUpperbound = array<i1: true>, independent = [#acc.device_type<none>]}

! ---------------------------------------------------------------------------
! LOCAL_INIT locality spec → acc.firstprivate
! ---------------------------------------------------------------------------

! LOCAL_INIT in kernels region (auto)
! CHECK-LABEL: func.func @_QPlocal_init_kernels_region
subroutine local_init_kernels_region()
  real :: a(10), scale
  integer :: i
  scale = 2.0
  !$acc kernels
  do concurrent(i=1:10) local_init(scale)
    a(i) = a(i) * scale
  end do
  !$acc end kernels
end subroutine

! CHECK: acc.kernels {
! CHECK: %[[FP:.*]] = acc.firstprivate varPtr(%{{.*}} : !fir.ref<f32>) recipe(@firstprivatization_ref_f32) -> !fir.ref<f32> {name = "scale"}
! CHECK: acc.loop {{.*}}firstprivate(%[[FP]] : !fir.ref<f32>)
! CHECK: } attributes {auto_ = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}

! ---------------------------------------------------------------------------
! Mixed locality specs: REDUCE + LOCAL
! ---------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPmixed_locality
subroutine mixed_locality()
  real :: a(10), s, tmp
  integer :: i
  s = 0.
  !$acc parallel loop
  do concurrent(i=1:10) reduce(+:s) local(tmp)
    tmp = a(i) * a(i)
    s = s + tmp
  end do
end subroutine

! CHECK: acc.parallel combined(loop)
! CHECK-DAG: %[[RED:.*]] = acc.reduction varPtr(%{{.*}} : !fir.ref<f32>) recipe(@reduction_add{{.*}}) -> !fir.ref<f32> {name = "s"}
! CHECK-DAG: %[[PRIV:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<f32>) recipe(@privatization_ref_f32) -> !fir.ref<f32> {name = "tmp"}
! CHECK: acc.loop {{.*}}reduction(%[[RED]] : !fir.ref<f32>)
! CHECK: } attributes {inclusiveUpperbound = array<i1: true>, independent = [#acc.device_type<none>]}

! ---------------------------------------------------------------------------
! Reduce combined with explicit ACC reduction clause
! ---------------------------------------------------------------------------

! CHECK-LABEL: func.func @_QPreduce_with_acc_clause
subroutine reduce_with_acc_clause()
  real :: a(10), s1, s2
  integer :: i
  s1 = 0.
  s2 = 0.
  !$acc parallel loop reduction(+:s1)
  do concurrent(i=1:10) reduce(+:s2)
    s1 = s1 + a(i)
    s2 = s2 + a(i) * 2.0
  end do
end subroutine

! CHECK: acc.parallel combined(loop)
! CHECK-DAG: acc.reduction varPtr(%{{.*}} : !fir.ref<f32>) recipe(@reduction_add{{.*}}) -> !fir.ref<f32> {name = "s1"}
! CHECK-DAG: acc.reduction varPtr(%{{.*}} : !fir.ref<f32>) recipe(@reduction_add{{.*}}) -> !fir.ref<f32> {name = "s2"}
! CHECK: acc.loop {{.*}}reduction(

! ---------------------------------------------------------------------------
! Explicit loop par mode clauses: auto, seq, independent
! ---------------------------------------------------------------------------

! kernels loop auto with reduce
! CHECK-LABEL: func.func @_QPreduce_kernels_loop_auto
subroutine reduce_kernels_loop_auto()
  real :: a(10), s
  integer :: i
  s = 0.
  !$acc kernels loop auto
  do concurrent(i=1:10) reduce(+:s)
    s = s + a(i)
  end do
end subroutine

! CHECK: acc.kernels combined(loop)
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%{{.*}} : !fir.ref<f32>) recipe(@reduction_add{{.*}}) -> !fir.ref<f32> {name = "s"}
! CHECK: acc.loop combined(kernels) {{.*}}reduction(%[[RED]] : !fir.ref<f32>)
! CHECK: } attributes {auto_ = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}

! kernels loop seq with reduce
! CHECK-LABEL: func.func @_QPreduce_kernels_loop_seq
subroutine reduce_kernels_loop_seq()
  real :: a(10), s
  integer :: i
  s = 0.
  !$acc kernels loop seq
  do concurrent(i=1:10) reduce(+:s)
    s = s + a(i)
  end do
end subroutine

! CHECK: acc.kernels combined(loop)
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%{{.*}} : !fir.ref<f32>) recipe(@reduction_add{{.*}}) -> !fir.ref<f32> {name = "s"}
! CHECK: acc.loop combined(kernels) {{.*}}reduction(%[[RED]] : !fir.ref<f32>)
! CHECK: } attributes {inclusiveUpperbound = array<i1: true>, seq = [#acc.device_type<none>]}

! kernels loop independent with reduce
! CHECK-LABEL: func.func @_QPreduce_kernels_loop_independent
subroutine reduce_kernels_loop_independent()
  real :: a(10), s
  integer :: i
  s = 0.
  !$acc kernels loop independent
  do concurrent(i=1:10) reduce(+:s)
    s = s + a(i)
  end do
end subroutine

! CHECK: acc.kernels combined(loop)
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%{{.*}} : !fir.ref<f32>) recipe(@reduction_add{{.*}}) -> !fir.ref<f32> {name = "s"}
! CHECK: acc.loop combined(kernels) {{.*}}reduction(%[[RED]] : !fir.ref<f32>)
! CHECK: } attributes {inclusiveUpperbound = array<i1: true>, independent = [#acc.device_type<none>]}
