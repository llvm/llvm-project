! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s | FileCheck %s

! Keep each modifier aligned with its variables, including implicit defaults.

! CHECK-LABEL: func.func @_QPmixed_task_byref
subroutine mixed_task_byref(x, y)
  integer :: x(4), y
  ! CHECK: omp.parallel {{.*}}reduction(
  ! CHECK-SAME: task byref @{{[^ ]+}} %{{[^ ]+}} -> %{{[^ ]+}},
  ! CHECK-SAME: @{{[^ ]+}} %{{[^ ]+}} -> %{{[^ ]+}}
  ! CHECK-SAME: : !fir.ref<!fir.box<!fir.array<4xi32>>>, !fir.ref<i32>)
  !$omp parallel reduction(task, +: x) reduction(+: y)
  x = x + 1
  y = y + 1
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPmixed_parallel
subroutine mixed_parallel(x, y, z)
  integer :: x, y, z, i
  ! CHECK: omp.parallel {{.*}}reduction(
  ! CHECK-SAME: task @{{[^ ]+}} %{{[^ ]+}} -> %{{[^ ]+}},
  ! CHECK-SAME: @{{[^ ]+}} %{{[^ ]+}} -> %{{[^ ]+}}
  ! CHECK-SAME: : !fir.ref<i32>, !fir.ref<i32>)
  !$omp parallel reduction(task, +: x) reduction(default, +: y)
  ! CHECK: omp.task in_reduction(
  !$omp task in_reduction(+: x)
  x = x + 1
  !$omp end task
  y = y + 1
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPmultiple_inscan
subroutine multiple_inscan(x, y, a)
  integer :: x, y, a(4), i
  ! CHECK: omp.wsloop {{.*}}reduction(
  ! CHECK-SAME: inscan @{{[^ ]+}} %{{[^ ]+}} -> %{{[^ ]+}},
  ! CHECK-SAME: inscan @{{[^ ]+}} %{{[^ ]+}} -> %{{[^ ]+}}
  ! CHECK-SAME: : !fir.ref<i32>, !fir.ref<i32>)
  !$omp parallel do reduction(inscan, +: x, y)
  do i = 1, 4
    x = x + i
    y = y + 1
    ! CHECK: omp.scan inclusive(
    !$omp scan inclusive(x, y)
    a(i) = x + y
  end do
  !$omp end parallel do
end subroutine

! CHECK-LABEL: func.func @_QPmixed_do
subroutine mixed_do(x, y, z)
  integer :: x, y, z, i
  ! CHECK: omp.wsloop {{.*}}reduction(
  ! CHECK-SAME: @{{[^ ]+}} %{{[^ ]+}} -> %{{[^ ]+}},
  ! CHECK-SAME: task @{{[^ ]+}} %{{[^ ]+}} -> %{{[^ ]+}}
  ! CHECK-SAME: : !fir.ref<i32>, !fir.ref<i32>)
  !$omp do reduction(default, +: y) reduction(task, +: x)
  do i = 1, 4
  ! CHECK: omp.task in_reduction(
  !$omp task in_reduction(+: x)
  x = x + 1
  !$omp end task
  y = y + 1
  end do
  !$omp end do
end subroutine

! CHECK-LABEL: func.func @_QPmixed_sections
subroutine mixed_sections(x, y, z)
  integer :: x, y, z, i
  ! CHECK: omp.sections {{.*}}reduction(
  ! CHECK-SAME: task @{{[^ ]+}} %{{[^ ]+}} -> %{{[^ ]+}},
  ! CHECK-SAME: @{{[^ ]+}} %{{[^ ]+}} -> %{{[^ ]+}}
  ! CHECK-SAME: : !fir.ref<i32>, !fir.ref<i32>)
  !$omp sections reduction(task, +: x) reduction(+: y)
  !$omp section
  ! CHECK: omp.task in_reduction(
  !$omp task in_reduction(+: x)
  x = x + 1
  !$omp end task
  y = y + 1
  !$omp end sections
end subroutine

! CHECK-LABEL: func.func @_QPmixed_combined
subroutine mixed_combined(x, y, z)
  integer :: x, y, z, i
  ! CHECK: omp.wsloop {{.*}}reduction(
  ! CHECK-SAME: @{{[^ ]+}} %{{[^ ]+}} -> %{{[^ ]+}},
  ! CHECK-SAME: task @{{[^ ]+}} %{{[^ ]+}} -> %{{[^ ]+}}
  ! CHECK-SAME: : !fir.ref<i32>, !fir.ref<i32>)
  !$omp parallel do reduction(+: y) reduction(task, +: x)
  do i = 1, 4
  ! CHECK: omp.task in_reduction(
  !$omp task in_reduction(+: x)
  x = x + 1
  !$omp end task
  y = y + 1
  end do
  !$omp end parallel do
end subroutine

! CHECK-LABEL: func.func @_QPmixed_multiple
subroutine mixed_multiple(x, y, z)
  integer :: x, y, z, i
  ! CHECK: omp.wsloop {{.*}}reduction(
  ! CHECK-SAME: task @{{[^ ]+}} %{{[^ ]+}} -> %{{[^ ]+}},
  ! CHECK-SAME: task @{{[^ ]+}} %{{[^ ]+}} -> %{{[^ ]+}},
  ! CHECK-SAME: @{{[^ ]+}} %{{[^ ]+}} -> %{{[^ ]+}}
  ! CHECK-SAME: : !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>)
  !$omp parallel do reduction(task, +: x, z) reduction(default, +: y)
  do i = 1, 4
  ! CHECK: omp.task in_reduction(
  !$omp task in_reduction(+: x)
  x = x + 1
  !$omp end task
  y = y + 1
  z = z + 1
  end do
  !$omp end parallel do
end subroutine
