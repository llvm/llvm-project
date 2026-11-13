! By-reference (box) reductions get no atomic region, so volatile and
! non-volatile allocatables or pointers share one declaration with no
! _volatile suffix, whichever operand comes first.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s | FileCheck %s

! CHECK-NOT:   add_reduction_byref_box_heap_f32_volatile
! CHECK-NOT:   add_reduction_byref_box_ptr_f32_volatile

! CHECK-LABEL: func.func @_QPred_byref_vol
! CHECK:         omp.wsloop {{.*}}reduction(byref @add_reduction_byref_box_heap_f32 %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.heap<f32>>>)
! CHECK:         omp.wsloop {{.*}}reduction(byref @add_reduction_byref_box_heap_f32 %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.heap<f32>, volatile>, volatile>)

subroutine red_byref_vol(a, n, res, vres)
  integer :: n, i
  real :: a(n)
  real, allocatable :: res
  real, allocatable, volatile :: vres
  !$omp parallel do reduction(+:res)
  do i = 1, n
    res = res + a(i)
  end do
  !$omp parallel do reduction(+:vres)
  do i = 1, n
    vres = vres + a(i)
  end do
end subroutine

! Pointers behave the same way.
! CHECK-LABEL: func.func @_QPred_byref_ptr
! CHECK:         omp.wsloop {{.*}}reduction(byref @add_reduction_byref_box_ptr_f32 %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.ptr<f32>>>)
! CHECK:         omp.wsloop {{.*}}reduction(byref @add_reduction_byref_box_ptr_f32 %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.ptr<f32>, volatile>, volatile>)

subroutine red_byref_ptr(a, n, res, vres)
  integer :: n, i
  real :: a(n)
  real, pointer :: res
  real, pointer, volatile :: vres
  !$omp parallel do reduction(+:res)
  do i = 1, n
    res = res + a(i)
  end do
  !$omp parallel do reduction(+:vres)
  do i = 1, n
    vres = vres + a(i)
  end do
end subroutine

! Volatile operand first: the declaration is still built with the non-volatile
! type, so both orderings share it.
! CHECK-LABEL: func.func @_QPred_byref_vol_first
! CHECK:         omp.wsloop {{.*}}reduction(byref @add_reduction_byref_box_heap_f32 %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.heap<f32>, volatile>, volatile>)
! CHECK:         omp.wsloop {{.*}}reduction(byref @add_reduction_byref_box_heap_f32 %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.heap<f32>>>)

subroutine red_byref_vol_first(a, n, res, vres)
  integer :: n, i
  real :: a(n)
  real, allocatable :: res
  real, allocatable, volatile :: vres
  !$omp parallel do reduction(+:vres)
  do i = 1, n
    vres = vres + a(i)
  end do
  !$omp parallel do reduction(+:res)
  do i = 1, n
    res = res + a(i)
  end do
end subroutine

! A standalone volatile allocatable uses the same non-volatile declaration.
! CHECK-LABEL: func.func @_QPred_byref_vol_alone
! CHECK:         omp.wsloop {{.*}}reduction(byref @add_reduction_byref_box_heap_f32 %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.heap<f32>, volatile>, volatile>)

subroutine red_byref_vol_alone(a, n, vres)
  integer :: n, i
  real :: a(n)
  real, allocatable, volatile :: vres
  !$omp parallel do reduction(+:vres)
  do i = 1, n
    vres = vres + a(i)
  end do
end subroutine

! Volatile-first pointer reduction: same canonicalization, shared declaration.
! CHECK-LABEL: func.func @_QPred_byref_ptr_vol_first
! CHECK:         omp.wsloop {{.*}}reduction(byref @add_reduction_byref_box_ptr_f32 %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.ptr<f32>, volatile>, volatile>)
! CHECK:         omp.wsloop {{.*}}reduction(byref @add_reduction_byref_box_ptr_f32 %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.ptr<f32>>>)

subroutine red_byref_ptr_vol_first(a, n, res, vres)
  integer :: n, i
  real :: a(n)
  real, pointer :: res
  real, pointer, volatile :: vres
  !$omp parallel do reduction(+:vres)
  do i = 1, n
    vres = vres + a(i)
  end do
  !$omp parallel do reduction(+:res)
  do i = 1, n
    res = res + a(i)
  end do
end subroutine

! Standalone volatile pointer reduction.
! CHECK-LABEL: func.func @_QPred_byref_ptr_vol_alone
! CHECK:         omp.wsloop {{.*}}reduction(byref @add_reduction_byref_box_ptr_f32 %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.ptr<f32>, volatile>, volatile>)

subroutine red_byref_ptr_vol_alone(a, n, vres)
  integer :: n, i
  real :: a(n)
  real, pointer, volatile :: vres
  !$omp parallel do reduction(+:vres)
  do i = 1, n
    vres = vres + a(i)
  end do
end subroutine
