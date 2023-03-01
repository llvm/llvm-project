! RUN: bbc -emit-fir %s -o - | FileCheck %s

! Test lowering of ALLOCATE statement with a MOLD argument for scalars

subroutine scalar_mold_allocation()
  integer, allocatable :: a
  allocate(a, mold=9)
end subroutine

! CHECK-LABEL: func.func @_QPscalar_mold_allocation() {
! CHECK: %[[A:.*]] = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "a", uniq_name = "_QFscalar_mold_allocationEa"}
! CHECK: %[[HEAP_A:.*]] = fir.alloca !fir.heap<i32> {uniq_name = "_QFscalar_mold_allocationEa.addr"}
! CHECK: %[[ADDR_A:.*]] = fir.load %[[HEAP_A]] : !fir.ref<!fir.heap<i32>>
! CHECK: %[[BOX_ADDR_A:.*]] = fir.embox %[[ADDR_A]] : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
! CHECK: fir.store %[[BOX_ADDR_A]] to %[[A]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK: %[[A_REF_BOX_NONE1:.*]] = fir.convert %[[A]] : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableApplyMold(%[[A_REF_BOX_NONE1]], %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32) -> none
! CHECK: %[[A_REF_BOX_NONE2:.*]] = fir.convert %[[A]] : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableAllocate(%[[A_REF_BOX_NONE2]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
