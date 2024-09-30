! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

!CHECK: fir.global common @[[CB_C:.*]](dense<0> : vector<8xi8>) {alignment = 4 : i64} : !fir.array<8xi8>
!CHECK-LABEL: func.func @_QPlastprivate_common
!CHECK:      %[[CB_C_REF:.*]] = fir.address_of(@[[CB_C]]) : !fir.ref<!fir.array<8xi8>>
!CHECK:      %[[CB_C_REF_CVT:.*]] = fir.convert %[[CB_C_REF]] : (!fir.ref<!fir.array<8xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK:      %[[CB_C_X_COOR:.*]] = fir.coordinate_of %[[CB_C_REF_CVT]], %{{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK:      %[[CB_C_X_ADDR:.*]] = fir.convert %[[CB_C_X_COOR]] : (!fir.ref<i8>) -> !fir.ref<f32>
!CHECK:      %[[X_DECL:.*]]:2 = hlfir.declare %[[CB_C_X_ADDR]] {uniq_name = "_QFlastprivate_commonEx"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK:      %[[CB_C_REF_CVT:.*]] = fir.convert %[[CB_C_REF]] : (!fir.ref<!fir.array<8xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK:      %[[CB_C_Y_COOR:.*]] = fir.coordinate_of %[[CB_C_REF_CVT]], %{{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK:      %[[CB_C_Y_ADDR:.*]] = fir.convert %[[CB_C_Y_COOR]] : (!fir.ref<i8>) -> !fir.ref<f32>
!CHECK:      %[[Y_DECL:.*]]:2 = hlfir.declare %[[CB_C_Y_ADDR]] {uniq_name = "_QFlastprivate_commonEy"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK:      %[[PRIVATE_X_REF:.*]] = fir.alloca f32 {bindc_name = "x", pinned, uniq_name = "_QFlastprivate_commonEx"}
!CHECK:      %[[PRIVATE_X_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_X_REF]] {uniq_name = "_QFlastprivate_commonEx"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK:      %[[PRIVATE_Y_REF:.*]] = fir.alloca f32 {bindc_name = "y", pinned, uniq_name = "_QFlastprivate_commonEy"}
!CHECK:      %[[PRIVATE_Y_DECL:.*]]:2 = hlfir.declare %[[PRIVATE_Y_REF]] {uniq_name = "_QFlastprivate_commonEy"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!CHECK:      omp.wsloop {
!CHECK-NEXT:   omp.loop_nest (%[[I:.*]]) : i32 = (%{{.*}}) to (%{{.*}}) inclusive step (%{{.*}}) {
!CHECK:          %[[V:.*]] = arith.addi %[[I]], %{{.*}} : i32
!CHECK:          %[[C0:.*]] = arith.constant 0 : i32
!CHECK:          %[[NEG_STEP:.*]] = arith.cmpi slt, %{{.*}}, %[[C0]] : i32
!CHECK:          %[[V_LT:.*]] = arith.cmpi slt, %[[V]], %{{.*}} : i32
!CHECK:          %[[V_GT:.*]] = arith.cmpi sgt, %[[V]], %{{.*}} : i32
!CHECK:          %[[LAST_ITER:.*]] = arith.select %[[NEG_STEP]], %[[V_LT]], %[[V_GT]] : i1
!CHECK:          fir.if %[[LAST_ITER]] {
!CHECK:            fir.store %[[V]] to %{{.*}} : !fir.ref<i32>
!CHECK:            %[[PRIVATE_X_VAL:.*]] = fir.load %[[PRIVATE_X_DECL]]#0 : !fir.ref<f32>
!CHECK:            hlfir.assign %[[PRIVATE_X_VAL]] to %[[X_DECL]]#0 : f32, !fir.ref<f32>
!CHECK:            %[[PRIVATE_Y_VAL:.*]] = fir.load %[[PRIVATE_Y_DECL]]#0 : !fir.ref<f32>
!CHECK:            hlfir.assign %[[PRIVATE_Y_VAL]] to %[[Y_DECL]]#0 : f32, !fir.ref<f32>
!CHECK:          }
!CHECK:          omp.yield
!CHECK:        }
!CHECK:        omp.terminator
!CHECK:      }
subroutine lastprivate_common
  common /c/ x, y
  real x, y
  !$omp do lastprivate(/c/)
  do i=1,100
  end do
  !$omp end do
end subroutine
