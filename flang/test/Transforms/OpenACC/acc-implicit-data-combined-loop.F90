! RUN: bbc -fopenacc -emit-hlfir %s -o - | \
! RUN:   fir-opt --pass-pipeline="builtin.module(acc-initialize-fir-analyses,acc-implicit-data)" | \
! RUN:   FileCheck %s

! RUN: bbc -fopenacc -emit-hlfir %s -o - | \
! RUN:   fir-opt --pass-pipeline="builtin.module(acc-initialize-fir-analyses,acc-implicit-data{enable-combined-loop-implicit-firstprivate=false})" | \
! RUN:   FileCheck %s --check-prefix=FLAG-OFF

subroutine combined_parallel_loop_implicit_fp
  integer :: i, n, t
  real :: a(10)
  n = 10
  t = 1
  !$acc parallel loop
  do i = 1, n
    t = t + 1
    a(i) = t
  end do
end subroutine

! CHECK-LABEL: func.func @_QPcombined_parallel_loop_implicit_fp
! CHECK: %[[FP_T:.*]] = acc.firstprivate varPtr({{.*}} : !fir.ref<i32>) recipe({{.*}}) implicit(true) name("t") -> !fir.ref<i32>
! CHECK-NOT: acc.parallel combined(loop){{.*}}firstprivate(%[[FP_T]]
! CHECK: acc.loop combined(parallel) {{.*}}firstprivate({{.*}}%[[FP_T]]{{.*}})

! FLAG-OFF-LABEL: func.func @_QPcombined_parallel_loop_implicit_fp
! FLAG-OFF: %[[FP_T:.*]] = acc.firstprivate varPtr({{.*}} : !fir.ref<i32>) recipe({{.*}}) implicit(true) name("t") -> !fir.ref<i32>
! FLAG-OFF: acc.parallel combined(loop) {{.*}}firstprivate({{.*}}%[[FP_T]]{{.*}})
! FLAG-OFF-NOT: acc.loop combined(parallel){{.*}}firstprivate({{.*}}%[[FP_T]]

subroutine combined_serial_loop_implicit_fp
  integer :: i, n, t
  real :: a(10)
  n = 10
  t = 1
  !$acc serial loop
  do i = 1, n
    t = t + 1
    a(i) = t
  end do
end subroutine

! CHECK-LABEL: func.func @_QPcombined_serial_loop_implicit_fp
! CHECK: %[[FP_T:.*]] = acc.firstprivate varPtr({{.*}} : !fir.ref<i32>) recipe({{.*}}) implicit(true) name("t") -> !fir.ref<i32>
! CHECK-NOT: acc.serial combined(loop){{.*}}firstprivate(%[[FP_T]]
! CHECK: acc.loop combined(serial) {{.*}}firstprivate({{.*}}%[[FP_T]]{{.*}})

! FLAG-OFF-LABEL: func.func @_QPcombined_serial_loop_implicit_fp
! FLAG-OFF: %[[FP_T:.*]] = acc.firstprivate varPtr({{.*}} : !fir.ref<i32>) recipe({{.*}}) implicit(true) name("t") -> !fir.ref<i32>
! FLAG-OFF: acc.serial combined(loop) {{.*}}firstprivate({{.*}}%[[FP_T]]{{.*}})
! FLAG-OFF-NOT: acc.loop combined(serial){{.*}}firstprivate({{.*}}%[[FP_T]]

subroutine noncombined_parallel_nested_loop_implicit_fp
  integer :: i, n, t
  real :: a(10)
  n = 10
  t = 1
  !$acc parallel
  !$acc loop
  do i = 1, n
    t = t + 1
    a(i) = t
  end do
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPnoncombined_parallel_nested_loop_implicit_fp
! CHECK: %[[FP_T:.*]] = acc.firstprivate varPtr({{.*}} : !fir.ref<i32>) recipe({{.*}}) implicit(true) name("t") -> !fir.ref<i32>
! CHECK: acc.parallel {{.*}}firstprivate({{.*}}%[[FP_T]]{{.*}})
! CHECK-NOT: acc.loop{{.*}}firstprivate({{.*}}%[[FP_T]]

! FLAG-OFF-LABEL: func.func @_QPnoncombined_parallel_nested_loop_implicit_fp
! FLAG-OFF: %[[FP_T:.*]] = acc.firstprivate varPtr({{.*}} : !fir.ref<i32>) recipe({{.*}}) implicit(true) name("t") -> !fir.ref<i32>
! FLAG-OFF: acc.parallel {{.*}}firstprivate({{.*}}%[[FP_T]]{{.*}})
! FLAG-OFF-NOT: acc.loop{{.*}}firstprivate({{.*}}%[[FP_T]]
