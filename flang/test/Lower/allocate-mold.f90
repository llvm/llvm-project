! RUN: %flang_fc1 -emit-hlfir -O2 %s -o - | FileCheck %s

! Test lowering of ALLOCATE statement with a MOLD argument for scalars

subroutine scalar_mold_allocation()
  integer, allocatable :: a
  allocate(a, mold=9)
end subroutine

! CHECK-LABEL: func.func @_QPscalar_mold_allocation() {
! CHECK: %[[A:.*]] = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "a", uniq_name = "_QFscalar_mold_allocationEa"}
! CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[A]] {{.*}}
! CHECK: %[[A_BOX_NONE:.*]] = fir.convert %[[A_DECL]]#0 : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<none>>
! CHECK: fir.call @_FortranAAllocatableApplyMold(%[[A_BOX_NONE]], %{{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32) -> ()
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableAllocate(%{{.*}}, %{{.*}} : (!fir.ref<!fir.box<none>>, !fir.ref<i64>, i1, !fir.box<none>, !fir.ref<i8>, i32, {{.*}}) -> i32

subroutine array_scalar_mold_allocation()
  real, allocatable :: a(:)

  allocate (a(10), mold=3.0)
end subroutine array_scalar_mold_allocation

! CHECK-LABEL: func.func @_QParray_scalar_mold_allocation() {
! CHECK: %[[A:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "a", uniq_name = "_QFarray_scalar_mold_allocationEa"}
! CHECK: %[[A_DECL:.*]]:2 = hlfir.declare %[[A]] {{.*}}
! CHECK: %[[REF_BOX_A:.*]] = fir.convert %[[A_DECL]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK: fir.call @_FortranAAllocatableApplyMold(%[[REF_BOX_A]], {{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32) -> ()
! CHECK: fir.call @_FortranAAllocatableSetBounds(%{{.*}}, {{.*}}) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK: %{{.*}} = fir.call @_FortranAAllocatableAllocate(%{{.*}}, %{{.*}}) {{.*}} : (!fir.ref<!fir.box<none>>, !fir.ref<i64>, i1, !fir.box<none>, !fir.ref<i8>, i32, {{.*}}) -> i32
