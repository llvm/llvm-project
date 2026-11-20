! By-reference (box) reductions get no atomic region, so volatile and
! non-volatile allocatables or pointers share one declaration with no
! _volatile suffix, whichever operand comes first. Each case is compiled on its
! own so that the declaration is created by the operand under test.

! RUN: split-file %s %t
! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --strict-fir-volatile-verifier -o - %t/heap.f90 | FileCheck %s --check-prefix=HEAP
! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --strict-fir-volatile-verifier -o - %t/ptr.f90 | FileCheck %s --check-prefix=PTR
! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --strict-fir-volatile-verifier -o - %t/heap_vol_first.f90 | FileCheck %s --check-prefix=HEAPVF
! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --strict-fir-volatile-verifier -o - %t/heap_vol_alone.f90 | FileCheck %s --check-prefix=HEAPVA
! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --strict-fir-volatile-verifier -o - %t/ptr_vol_first.f90 | FileCheck %s --check-prefix=PTRVF
! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --strict-fir-volatile-verifier -o - %t/ptr_vol_alone.f90 | FileCheck %s --check-prefix=PTRVA

!--- heap.f90
! HEAP-NOT:   add_reduction_volatile_byref_box_heap_f32
! HEAP:       omp.declare_reduction @add_reduction_byref_box_heap_f32 {{.*}} : !fir.ref<!fir.box<!fir.heap<f32>>>
! HEAP-NOT:   add_reduction_volatile_byref_box_heap_f32
! HEAP-LABEL: func.func @_QPred_byref_vol
! HEAP:         omp.wsloop {{.*}}reduction(byref @add_reduction_byref_box_heap_f32 %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.heap<f32>>>)
! HEAP:         omp.wsloop {{.*}}reduction(byref @add_reduction_byref_box_heap_f32 %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.heap<f32>, volatile>, volatile>)
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

!--- ptr.f90
! PTR-NOT:   add_reduction_volatile_byref_box_ptr_f32
! PTR:       omp.declare_reduction @add_reduction_byref_box_ptr_f32 {{.*}} : !fir.ref<!fir.box<!fir.ptr<f32>>>
! PTR-NOT:   add_reduction_volatile_byref_box_ptr_f32
! PTR-LABEL: func.func @_QPred_byref_ptr
! PTR:         omp.wsloop {{.*}}reduction(byref @add_reduction_byref_box_ptr_f32 %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.ptr<f32>>>)
! PTR:         omp.wsloop {{.*}}reduction(byref @add_reduction_byref_box_ptr_f32 %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.ptr<f32>, volatile>, volatile>)
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

!--- heap_vol_first.f90
! The volatile operand creates the declaration; it is still built from the
! non-volatile type, so the later non-volatile operand shares it.
! HEAPVF-NOT:   add_reduction_volatile_byref_box_heap_f32
! HEAPVF:       omp.declare_reduction @add_reduction_byref_box_heap_f32 {{.*}} : !fir.ref<!fir.box<!fir.heap<f32>>>
! HEAPVF-NOT:   add_reduction_volatile_byref_box_heap_f32
! HEAPVF-LABEL: func.func @_QPred_byref_vol_first
! HEAPVF:         omp.wsloop {{.*}}reduction(byref @add_reduction_byref_box_heap_f32 %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.heap<f32>, volatile>, volatile>)
! HEAPVF:         omp.wsloop {{.*}}reduction(byref @add_reduction_byref_box_heap_f32 %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.heap<f32>>>)
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

!--- heap_vol_alone.f90
! HEAPVA-NOT:   add_reduction_volatile_byref_box_heap_f32
! HEAPVA:       omp.declare_reduction @add_reduction_byref_box_heap_f32 {{.*}} : !fir.ref<!fir.box<!fir.heap<f32>>>
! HEAPVA-NOT:   add_reduction_volatile_byref_box_heap_f32
! HEAPVA-LABEL: func.func @_QPred_byref_vol_alone
! HEAPVA:         omp.wsloop {{.*}}reduction(byref @add_reduction_byref_box_heap_f32 %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.heap<f32>, volatile>, volatile>)
subroutine red_byref_vol_alone(a, n, vres)
  integer :: n, i
  real :: a(n)
  real, allocatable, volatile :: vres
  !$omp parallel do reduction(+:vres)
  do i = 1, n
    vres = vres + a(i)
  end do
end subroutine

!--- ptr_vol_first.f90
! PTRVF-NOT:   add_reduction_volatile_byref_box_ptr_f32
! PTRVF:       omp.declare_reduction @add_reduction_byref_box_ptr_f32 {{.*}} : !fir.ref<!fir.box<!fir.ptr<f32>>>
! PTRVF-NOT:   add_reduction_volatile_byref_box_ptr_f32
! PTRVF-LABEL: func.func @_QPred_byref_ptr_vol_first
! PTRVF:         omp.wsloop {{.*}}reduction(byref @add_reduction_byref_box_ptr_f32 %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.ptr<f32>, volatile>, volatile>)
! PTRVF:         omp.wsloop {{.*}}reduction(byref @add_reduction_byref_box_ptr_f32 %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.ptr<f32>>>)
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

!--- ptr_vol_alone.f90
! PTRVA-NOT:   add_reduction_volatile_byref_box_ptr_f32
! PTRVA:       omp.declare_reduction @add_reduction_byref_box_ptr_f32 {{.*}} : !fir.ref<!fir.box<!fir.ptr<f32>>>
! PTRVA-NOT:   add_reduction_volatile_byref_box_ptr_f32
! PTRVA-LABEL: func.func @_QPred_byref_ptr_vol_alone
! PTRVA:         omp.wsloop {{.*}}reduction(byref @add_reduction_byref_box_ptr_f32 %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.ptr<f32>, volatile>, volatile>)
subroutine red_byref_ptr_vol_alone(a, n, vres)
  integer :: n, i
  real :: a(n)
  real, pointer, volatile :: vres
  !$omp parallel do reduction(+:vres)
  do i = 1, n
    vres = vres + a(i)
  end do
end subroutine
