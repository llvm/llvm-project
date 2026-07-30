! RUN: bbc -emit-hlfir -gpu=unified %s -o - | FileCheck %s
! RUN: bbc -emit-hlfir -gpu=managed %s -o - | FileCheck %s --check-prefix=MANAGED

! Under -gpu=mem:unified|managed, allocate adjustable / VLA automatic arrays
! in CUDA unified/managed memory. Fixed-size automatic arrays stay on the
! stack so host pointers remain shared under unified memory (e.g. OpenACC).

module m_adj
  integer :: nx = 32
end module

! CHECK-LABEL: func.func @_QPvla(
! CHECK: %[[ALLOC:.*]] = cuf.alloc !fir.array<?xf32>, %{{.*}} : index {bindc_name = "a", data_attr = #cuf.cuda<unified>, uniq_name = "_QFvlaEa"}
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ALLOC]](%{{.*}}) {data_attr = #cuf.cuda<unified>, uniq_name = "_QFvlaEa"}
! CHECK: cuf.free %[[DECL]]#1 : !fir.ref<!fir.array<?xf32>> {data_attr = #cuf.cuda<unified>}
! MANAGED-LABEL: func.func @_QPvla(
! MANAGED: cuf.alloc !fir.array<?xf32>, %{{.*}} : index {{{.*}}data_attr = #cuf.cuda<managed>
! MANAGED: cuf.free %{{.*}} : !fir.ref<!fir.array<?xf32>> {data_attr = #cuf.cuda<managed>}
subroutine vla(n)
  integer :: n
  real :: a(n)
  a(1) = 1.0
end subroutine

! CHECK-LABEL: func.func @_QPadjustable(
! CHECK: cuf.alloc !fir.array<?xf32>, %{{.*}} : index {{{.*}}data_attr = #cuf.cuda<unified>
! CHECK: cuf.free %{{.*}} {data_attr = #cuf.cuda<unified>}
! MANAGED-LABEL: func.func @_QPadjustable(
! MANAGED: cuf.alloc !fir.array<?xf32>, %{{.*}} : index {{{.*}}data_attr = #cuf.cuda<managed>
subroutine adjustable
  use m_adj
  real :: a(0:(nx+1)/2)
  a(0) = 0.0
end subroutine

! Fixed-size automatic arrays must remain ordinary stack allocations.
! CHECK-LABEL: func.func @_QPfixed(
! CHECK-NOT: cuf.alloc
! CHECK: fir.alloca !fir.array<128xf32>
! CHECK-NOT: cuf.free
! MANAGED-LABEL: func.func @_QPfixed(
! MANAGED-NOT: cuf.alloc
! MANAGED: fir.alloca !fir.array<128xf32>
subroutine fixed
  real :: a(128)
  a(1) = 1.0
end subroutine

! Dummy adjustable arrays are caller-allocated; do not retag them.
! CHECK-LABEL: func.func @_QPdummy_adj(
! CHECK-NOT: cuf.alloc
! MANAGED-LABEL: func.func @_QPdummy_adj(
! MANAGED-NOT: cuf.alloc
subroutine dummy_adj(a, n)
  integer :: n
  real :: a(n)
  a(1) = 1.0
end subroutine
