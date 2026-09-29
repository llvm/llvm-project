! RUN: bbc -emit-hlfir -gpu=unified %s -o - | FileCheck %s --check-prefixes=CHECK,UNIFIED
! RUN: bbc -emit-hlfir -gpu=managed %s -o - | FileCheck %s --check-prefixes=CHECK,MANAGED
! RUN: bbc -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK,NOFLAG

! Under -gpu=mem:unified|managed, dynamic automatic arrays are later moved to
! the heap and allocated with malloc_unified / malloc_managed. Lowering only
! records the mode on the module; symbols stay unmarked (no cudaDataAttr).

! UNIFIED: module attributes {{{.*}}fir.cuda_heap_alloc = "unified"
! MANAGED: module attributes {{{.*}}fir.cuda_heap_alloc = "managed"
! NOFLAG-NOT: fir.cuda_heap_alloc

module m_adj
  integer :: nx = 32
end module

! CHECK-LABEL: func.func @_QPvla(
! CHECK-NOT: cuf.alloc
! CHECK-NOT: data_attr = #cuf.cuda
! CHECK: fir.alloca !fir.array<?xf32>
subroutine vla(n)
  integer :: n
  real :: a(n)
  a(1) = 1.0
end subroutine

! CHECK-LABEL: func.func @_QPadjustable(
! CHECK-NOT: cuf.alloc
! CHECK-NOT: data_attr = #cuf.cuda
! CHECK: fir.alloca !fir.array<?xf32>
subroutine adjustable
  use m_adj
  real :: a(0:(nx+1)/2)
  a(0) = 0.0
end subroutine

! Fixed-size automatic arrays remain ordinary stack allocations.
! CHECK-LABEL: func.func @_QPfixed(
! CHECK-NOT: cuf.alloc
! CHECK: fir.alloca !fir.array<128xf32>
subroutine fixed
  real :: a(128)
  a(1) = 1.0
end subroutine

! Dummy adjustable arrays are caller-allocated.
! CHECK-LABEL: func.func @_QPdummy_adj(
! CHECK-NOT: cuf.alloc
! CHECK-NOT: fir.alloca !fir.array<?xf32>
subroutine dummy_adj(a, n)
  integer :: n
  real :: a(n)
  a(1) = 1.0
end subroutine
