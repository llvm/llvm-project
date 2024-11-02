! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! Test matmul intrinsic

! CHECK-LABEL: matmul_test
! CHECK-SAME: (%[[X:.*]]: !fir.ref<!fir.array<3x1xf32>>{{.*}}, %[[Y:.*]]: !fir.ref<!fir.array<1x3xf32>>{{.*}}, %[[Z:.*]]: !fir.ref<!fir.array<3x3xf32>>{{.*}})
! CHECK:  %[[RESULT_BOX_ADDR:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xf32>>>
! CHECK:  %[[C3:.*]] = arith.constant 3 : index
! CHECK:  %[[C1:.*]] = arith.constant 1 : index
! CHECK:  %[[C1_0:.*]] = arith.constant 1 : index
! CHECK:  %[[C3_1:.*]] = arith.constant 3 : index
! CHECK:  %[[Z_BOX:.*]] = fir.array_load %[[Z]]({{.*}}) : (!fir.ref<!fir.array<3x3xf32>>, !fir.shape<2>) -> !fir.array<3x3xf32>
! CHECK:  %[[X_SHAPE:.*]] = fir.shape %[[C3]], %[[C1]] : (index, index) -> !fir.shape<2>
! CHECK:  %[[X_BOX:.*]] = fir.embox %[[X]](%[[X_SHAPE]]) : (!fir.ref<!fir.array<3x1xf32>>, !fir.shape<2>) -> !fir.box<!fir.array<3x1xf32>>
! CHECK:  %[[Y_SHAPE:.*]] = fir.shape %[[C1_0]], %[[C3_1]] : (index, index) -> !fir.shape<2>
! CHECK:  %[[Y_BOX:.*]] = fir.embox %[[Y]](%[[Y_SHAPE]]) : (!fir.ref<!fir.array<1x3xf32>>, !fir.shape<2>) -> !fir.box<!fir.array<1x3xf32>>
! CHECK:  %[[ZERO_INIT:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?xf32>>
! CHECK:  %[[C0:.*]] = arith.constant 0 : index
! CHECK:  %[[RESULT_SHAPE:.*]] = fir.shape %[[C0]], %[[C0]] : (index, index) -> !fir.shape<2>
! CHECK:  %[[RESULT_BOX_VAL:.*]] = fir.embox %[[ZERO_INIT]](%[[RESULT_SHAPE]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xf32>>>
! CHECK:  fir.store %[[RESULT_BOX_VAL]] to %[[RESULT_BOX_ADDR]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
! CHECK:  %[[RESULT_BOX_ADDR_RUNTIME:.*]] = fir.convert %[[RESULT_BOX_ADDR]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:  %[[X_BOX_RUNTIME:.*]] = fir.convert %[[X_BOX]] : (!fir.box<!fir.array<3x1xf32>>) -> !fir.box<none>
! CHECK:  %[[Y_BOX_RUNTIME:.*]] = fir.convert %[[Y_BOX]] : (!fir.box<!fir.array<1x3xf32>>) -> !fir.box<none>
! CHECK:  {{.*}}fir.call @_FortranAMatmul(%[[RESULT_BOX_ADDR_RUNTIME]], %[[X_BOX_RUNTIME]], %[[Y_BOX_RUNTIME]], {{.*}}, {{.*}} : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:  %[[RESULT_BOX:.*]] = fir.load %[[RESULT_BOX_ADDR]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
! CHECK:  %[[RESULT_TMP:.*]] = fir.box_addr %[[RESULT_BOX]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>) -> !fir.heap<!fir.array<?x?xf32>>
! CHECK:  %[[Z_COPY_FROM_RESULT:.*]] = fir.do_loop
! CHECK:    {{.*}}fir.array_fetch
! CHECK:    {{.*}}fir.array_update
! CHECK:    fir.result
! CHECK:  }
! CHECK:  fir.array_merge_store %[[Z_BOX]], %[[Z_COPY_FROM_RESULT]] to %[[Z]] : !fir.array<3x3xf32>, !fir.array<3x3xf32>, !fir.ref<!fir.array<3x3xf32>>
! CHECK:  fir.freemem %[[RESULT_TMP]] : !fir.heap<!fir.array<?x?xf32>>
subroutine matmul_test(x,y,z)
  real :: x(3,1), y(1,3), z(3,3)
  z = matmul(x,y)
end subroutine

! CHECK-LABEL: matmul_test2
! CHECK-SAME: (%[[X_BOX:.*]]: !fir.box<!fir.array<?x?x!fir.logical<4>>>{{.*}}, %[[Y_BOX:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>{{.*}}, %[[Z_BOX:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>{{.*}})
!CHECK:  %[[RESULT_BOX_ADDR:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>
!CHECK:  %[[Z:.*]] = fir.array_load %[[Z_BOX]] : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.array<?x!fir.logical<4>>
!CHECK:  %[[ZERO_INIT:.*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.logical<4>>>
!CHECK:  %[[C0:.*]] = arith.constant 0 : index
!CHECK:  %[[RESULT_SHAPE:.*]] = fir.shape %[[C0]] : (index) -> !fir.shape<1>
!CHECK:  %[[RESULT_BOX:.*]] = fir.embox %[[ZERO_INIT]](%[[RESULT_SHAPE]]) : (!fir.heap<!fir.array<?x!fir.logical<4>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>
!CHECK:  fir.store %[[RESULT_BOX]] to %[[RESULT_BOX_ADDR]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>
!CHECK:  %[[RESULT_BOX_RUNTIME:.*]] = fir.convert %[[RESULT_BOX_ADDR]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>) -> !fir.ref<!fir.box<none>>
!CHECK:  %[[X_BOX_RUNTIME:.*]] = fir.convert %[[X_BOX]] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>) -> !fir.box<none>
!CHECK:  %[[Y_BOX_RUNTIME:.*]] = fir.convert %[[Y_BOX]] : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<none>
!CHECK:  {{.*}}fir.call @_FortranAMatmul(%[[RESULT_BOX_RUNTIME]], %[[X_BOX_RUNTIME]], %[[Y_BOX_RUNTIME]], {{.*}}, {{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> none
!CHECK:  %[[RESULT_BOX:.*]] = fir.load %[[RESULT_BOX_ADDR]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>
!CHECK:  %[[RESULT_TMP:.*]] = fir.box_addr %[[RESULT_BOX]] : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>) -> !fir.heap<!fir.array<?x!fir.logical<4>>>
!CHECK:  %[[Z_COPY_FROM_RESULT:.*]] = fir.do_loop
!CHECK:    {{.*}}fir.array_fetch
!CHECK:    {{.*}}fir.array_update
!CHECK:    fir.result
!CHECK:  }
!CHECK:  fir.array_merge_store %[[Z]], %[[Z_COPY_FROM_RESULT]] to %[[Z_BOX]] : !fir.array<?x!fir.logical<4>>, !fir.array<?x!fir.logical<4>>, !fir.box<!fir.array<?x!fir.logical<4>>>
!CHECK:  fir.freemem %[[RESULT_TMP]] : !fir.heap<!fir.array<?x!fir.logical<4>>>
subroutine matmul_test2(X, Y, Z)
  logical :: X(:,:)
  logical :: Y(:)
  logical :: Z(:)
  Z = matmul(X, Y)
end subroutine
