! RUN: bbc --use-desc-for-alloc=false -emit-fir -hlfir=false %s -o - | FileCheck %s

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
! CHECK: fir.call @_FortranAAllocatableApplyMold(%[[A_REF_BOX_NONE1]], %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32) -> ()
! CHECK: %[[A_REF_BOX_NONE2:.*]] = fir.convert %[[A]] : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableAllocate(%[[A_REF_BOX_NONE2]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

subroutine array_scalar_mold_allocation()
  real, allocatable :: a(:)

  allocate (a(10), mold=3.0)
end subroutine array_scalar_mold_allocation

! CHECK-LABEL: func.func @_QParray_scalar_mold_allocation() {
! CHECK: %[[A:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", uniq_name = "_QFarray_scalar_mold_allocationEa"}
! CHECK: %[[HEAP_A:.*]] = fir.alloca !fir.heap<!fir.array<?xf32>> {uniq_name = "_QFarray_scalar_mold_allocationEa.addr"}
! CHECK: %[[EXT0:.*]] = fir.alloca index {uniq_name = "_QFarray_scalar_mold_allocationEa.ext0"}
! CHECK: %[[ZERO:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf32>>
! CHECK: fir.store %[[ZERO]] to %[[HEAP_A]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
! CHECK: %[[LOADED_A:.*]] = fir.load %[[HEAP_A]] : !fir.ref<!fir.heap<!fir.array<?xf32>>>
! CHECK: %[[SHAPESHIFT:.*]] = fir.shape_shift {{.*}}, {{.*}} : (index, index) -> !fir.shapeshift<1>
! CHECK: %[[BOX_SHAPESHIFT:.*]] = fir.embox %[[LOADED_A]](%[[SHAPESHIFT]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK: fir.store %[[BOX_SHAPESHIFT]] to %[[A]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK: %[[REF_BOX_A0:.*]] = fir.convert %1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: fir.call @_FortranAAllocatableApplyMold(%[[REF_BOX_A0]], {{.*}}, {{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32) -> ()
! CHECK: %[[C10:.*]] = arith.constant 10 : i32
! CHECK: %[[REF_BOX_A1:.*]] = fir.convert %1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: fir.call @_FortranAAllocatableSetBounds(%[[REF_BOX_A1]], {{.*}},{{.*}}, {{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK: %[[REF_BOX_A2:.*]] = fir.convert %[[A]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableAllocate(%[[REF_BOX_A2]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
