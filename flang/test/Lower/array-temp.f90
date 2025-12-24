! RUN: bbc -hlfir=false -fwrapv %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPss1()
subroutine ss1
  ! CHECK: %[[aa:[0-9]+]] = fir.alloca !fir.array<2650000xf32> {bindc_name = "aa", uniq_name = "_QFss1Eaa"}
  ! CHECK: %[[shape:[0-9]+]] = fir.shape {{.*}} : (index) -> !fir.shape<1>
  integer, parameter :: N = 2650000
  real aa(N)
  ! CHECK: fir.array_coor %[[aa]](%[[shape]]) {{.*}} : (!fir.ref<!fir.array<2650000xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  aa = -2
  ! CHECK: %[[temp:[0-9]+]] = fir.allocmem !fir.array<2650000xf32>
  ! CHECK: fir.array_coor %[[aa]](%[[shape]]) {{.*}} : (!fir.ref<!fir.array<2650000xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  ! CHECK: fir.array_coor %[[temp]](%[[shape]]) {{.*}} : (!fir.heap<!fir.array<2650000xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  ! CHECK: fir.array_coor %[[aa]](%[[shape]]) [{{.*}}] {{.*}} : (!fir.ref<!fir.array<2650000xf32>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<f32>
  ! CHECK: fir.array_coor %[[temp]](%[[shape]]) [{{.*}}] {{.*}} : (!fir.heap<!fir.array<2650000xf32>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<f32>
  ! CHECK: fir.array_coor %[[temp]](%[[shape]]) {{.*}} : (!fir.heap<!fir.array<2650000xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  ! CHECK: fir.array_coor %[[aa]](%[[shape]]) {{.*}} : (!fir.ref<!fir.array<2650000xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  ! CHECK: fir.freemem %[[temp]] : !fir.heap<!fir.array<2650000xf32>>
  aa(2:N) = aa(1:N-1) + 7.0
! print*, aa(1:2), aa(N-1:N)
end

subroutine ss2(N)
  real aa(N)
  aa = -2
  aa(2:N) = aa(1:N-1) + 7.0
  print*, aa(1:2), aa(N-1:N)
end

subroutine ss3(N)
  real aa(2,N)
  aa = -2
  aa(:,2:N) = aa(:,1:N-1) + 7.0
  print*, aa(:,1:2), aa(:,N-1:N)
end

subroutine ss4(N)
  real aa(N,2)
  aa = -2
  aa(2:N,:) = aa(1:N-1,:) + 7.0
  print*, aa(1:2,:), aa(N-1:N,:)
end

! CHECK-LABEL: func @_QPss2(
! CHECK-SAME:               %arg0: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK-DAG: %[[C_m1:[-0-9a-z_]+]] = arith.constant -1 : index
! CHECK-DAG: %[[C_2:[-0-9a-z_]+]] = arith.constant 2 : index
! CHECK-DAG: %[[C_1:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK-DAG: %[[C_27_i32:[-0-9a-z_]+]] = arith.constant 27 : i32
! CHECK-DAG: %[[C_6_i32:[-0-9a-z_]+]] = arith.constant 6 : i32
! CHECK-DAG: %[[C_st:[-0-9a-z_]+]] = arith.constant 7.000000e+00 : f32
! CHECK-DAG: %[[C_1_i32:[-0-9a-z_]+]] = arith.constant 1 : i32
! CHECK-DAG: %[[C_st_0:[-0-9a-z_]+]] = arith.constant -2.000000e+00 : f32
! CHECK-DAG: %[[C_0:[-0-9a-z_]+]] = arith.constant 0 : index
! CHECK:   %[[V_0:[0-9]+]] = fir.load %arg0 : !fir.ref<i32>
! CHECK:   %[[V_1:[0-9]+]] = fir.convert %[[V_0:[0-9]+]] : (i32) -> index
! CHECK:   %[[V_2:[0-9]+]] = arith.cmpi sgt, %[[V_1]], %[[C_0]] : index
! CHECK:   %[[V_3:[0-9]+]] = arith.select %[[V_2]], %[[V_1]], %[[C_0]] : index
! CHECK:   %[[V_4:[0-9]+]] = fir.alloca !fir.array<?xf32>, %[[V_3]] {bindc_name = "aa", uniq_name = "_QFss2Eaa"}
! CHECK:   %[[V_5:[0-9]+]] = fir.shape %[[V_3:[0-9]+]] : (index) -> !fir.shape<1>
! CHECK:   cf.br ^bb1(%[[C_0]], %[[V_3:[0-9]+]] : index, index)
! CHECK: ^bb1(%[[V_6:[0-9]+]]: index, %[[V_7:[0-9]+]]: index):  // 2 preds: ^bb0, ^bb2
! CHECK:   %[[V_8:[0-9]+]] = arith.cmpi sgt, %[[V_7]], %[[C_0]] : index
! CHECK:   cf.cond_br %[[V_8]], ^bb2, ^bb3
! CHECK: ^bb2:  // pred: ^bb1
! CHECK:   %[[V_9:[0-9]+]] = arith.addi %[[V_6]], %[[C_1]] : index
! CHECK:   %[[V_10:[0-9]+]] = fir.array_coor %[[V_4]](%[[V_5]]) %[[V_9:[0-9]+]] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
! CHECK:   fir.store %[[C_st_0]] to %[[V_10:[0-9]+]] : !fir.ref<f32>
! CHECK:   %[[V_11:[0-9]+]] = arith.subi %[[V_7]], %[[C_1]] : index
! CHECK:   cf.br ^bb1(%[[V_9]], %[[V_11:[0-9]+]] : index, index)
! CHECK: ^bb3:  // pred: ^bb1
! CHECK:   %[[V_12:[0-9]+]] = fir.load %arg0 : !fir.ref<i32>
! CHECK:   %[[V_13:[0-9]+]] = fir.convert %[[V_12:[0-9]+]] : (i32) -> index
! CHECK:   %[[V_14:[0-9]+]] = arith.addi %[[V_13]], %[[C_m1]] : index
! CHECK:   %[[V_15:[0-9]+]] = arith.cmpi sgt, %[[V_14]], %[[C_0]] : index
! CHECK:   %[[V_16:[0-9]+]] = arith.select %[[V_15]], %[[V_14]], %[[C_0]] : index
! CHECK:   %[[V_17:[0-9]+]] = fir.slice %[[C_2]], %[[V_13]], %[[C_1]] : (index, index, index) -> !fir.slice<1>
! CHECK:   %[[V_18:[0-9]+]] = fir.allocmem !fir.array<?xf32>, %[[V_3]]
! CHECK:   cf.br ^bb4(%[[C_0]], %[[V_3:[0-9]+]] : index, index)
! CHECK: ^bb4(%[[V_19:[0-9]+]]: index, %[[V_20:[0-9]+]]: index):  // 2 preds: ^bb3, ^bb5
! CHECK:   %[[V_21:[0-9]+]] = arith.cmpi sgt, %[[V_20]], %[[C_0]] : index
! CHECK:   cf.cond_br %[[V_21]], ^bb5, ^bb6
! CHECK: ^bb5:  // pred: ^bb4
! CHECK:   %[[V_22:[0-9]+]] = arith.addi %[[V_19]], %[[C_1]] : index
! CHECK:   %[[V_23:[0-9]+]] = fir.array_coor %[[V_4]](%[[V_5]]) %[[V_22:[0-9]+]] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
! CHECK:   %[[V_24:[0-9]+]] = fir.array_coor %[[V_18]](%[[V_5]]) %[[V_22:[0-9]+]] : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
! CHECK:   %[[V_25:[0-9]+]] = fir.load %[[V_23:[0-9]+]] : !fir.ref<f32>
! CHECK:   fir.store %[[V_25]] to %[[V_24:[0-9]+]] : !fir.ref<f32>
! CHECK:   %[[V_26:[0-9]+]] = arith.subi %[[V_20]], %[[C_1]] : index
! CHECK:   cf.br ^bb4(%[[V_22]], %[[V_26:[0-9]+]] : index, index)
! CHECK: ^bb6:  // pred: ^bb4
! CHECK:   %[[V_27:[0-9]+]] = arith.subi %[[V_12]], %[[C_1_i32]] : i32
! CHECK:   %[[V_28:[0-9]+]] = fir.convert %[[V_27:[0-9]+]] : (i32) -> index
! CHECK:   %[[V_29:[0-9]+]] = fir.slice %[[C_1]], %[[V_28]], %[[C_1]] : (index, index, index) -> !fir.slice<1>
! CHECK:   cf.br ^bb7(%[[C_0]], %[[V_16:[0-9]+]] : index, index)
! CHECK: ^bb7(%[[V_30:[0-9]+]]: index, %[[V_31:[0-9]+]]: index):  // 2 preds: ^bb6, ^bb8
! CHECK:   %[[V_32:[0-9]+]] = arith.cmpi sgt, %[[V_31]], %[[C_0]] : index
! CHECK:   cf.cond_br %[[V_32]], ^bb8, ^bb9(%[[C_0]], %[[V_3:[0-9]+]] : index, index)
! CHECK: ^bb8:  // pred: ^bb7
! CHECK:   %[[V_33:[0-9]+]] = arith.addi %[[V_30]], %[[C_1]] : index
! CHECK:   %[[V_34:[0-9]+]] = fir.array_coor %[[V_4]](%[[V_5]]) [%[[V_29]]] %[[V_33:[0-9]+]] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<f32>
! CHECK:   %[[V_35:[0-9]+]] = fir.load %[[V_34:[0-9]+]] : !fir.ref<f32>
! CHECK:   %[[V_36:[0-9]+]] = arith.addf %[[V_35]], %[[C_st]] fastmath<contract> : f32
! CHECK:   %[[V_37:[0-9]+]] = fir.array_coor %[[V_18]](%[[V_5]]) [%[[V_17]]] %[[V_33:[0-9]+]] : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<f32>
! CHECK:   fir.store %[[V_36]] to %[[V_37:[0-9]+]] : !fir.ref<f32>
! CHECK:   %[[V_38:[0-9]+]] = arith.subi %[[V_31]], %[[C_1]] : index
! CHECK:   cf.br ^bb7(%[[V_33]], %[[V_38:[0-9]+]] : index, index)
! CHECK: ^bb9(%[[V_39:[0-9]+]]: index, %[[V_40:[0-9]+]]: index):  // 2 preds: ^bb7, ^bb10
! CHECK:   %[[V_41:[0-9]+]] = arith.cmpi sgt, %[[V_40]], %[[C_0]] : index
! CHECK:   cf.cond_br %[[V_41]], ^bb10, ^bb11
! CHECK: ^bb10:  // pred: ^bb9
! CHECK:   %[[V_42:[0-9]+]] = arith.addi %[[V_39]], %[[C_1]] : index
! CHECK:   %[[V_43:[0-9]+]] = fir.array_coor %[[V_18]](%[[V_5]]) %[[V_42:[0-9]+]] : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
! CHECK:   %[[V_44:[0-9]+]] = fir.array_coor %[[V_4]](%[[V_5]]) %[[V_42:[0-9]+]] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
! CHECK:   %[[V_45:[0-9]+]] = fir.load %[[V_43:[0-9]+]] : !fir.ref<f32>
! CHECK:   fir.store %[[V_45]] to %[[V_44:[0-9]+]] : !fir.ref<f32>
! CHECK:   %[[V_46:[0-9]+]] = arith.subi %[[V_40]], %[[C_1]] : index
! CHECK:   cf.br ^bb9(%[[V_42]], %[[V_46:[0-9]+]] : index, index)
! CHECK: ^bb11:  // pred: ^bb9
! CHECK:   fir.freemem %[[V_18:[0-9]+]] : !fir.heap<!fir.array<?xf32>>
! CHECK:   %[[V_49:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput(%[[C_6_i32]], %{{.*}}, %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:   %[[V_50:[0-9]+]] = fir.slice %[[C_1]], %[[C_2]], %[[C_1]] : (index, index, index) -> !fir.slice<1>
! CHECK:   %[[V_51:[0-9]+]] = fir.embox %[[V_4]](%[[V_5]]) [%[[V_50]]] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<2xf32>>
! CHECK:   %[[V_52:[0-9]+]] = fir.convert %[[V_51:[0-9]+]] : (!fir.box<!fir.array<2xf32>>) -> !fir.box<none>
! CHECK:   %[[V_53:[0-9]+]] = fir.call @_FortranAioOutputDescriptor(%[[V_49]], %[[V_52]]) fastmath<contract> {{.*}}: (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:   %[[V_54:[0-9]+]] = fir.load %arg0 : !fir.ref<i32>
! CHECK:   %[[V_55:[0-9]+]] = arith.subi %[[V_54]], %[[C_1_i32]] : i32
! CHECK:   %[[V_56:[0-9]+]] = fir.convert %[[V_55:[0-9]+]] : (i32) -> index
! CHECK:   %[[V_57:[0-9]+]] = fir.convert %[[V_54:[0-9]+]] : (i32) -> index
! CHECK:   %[[V_58:[0-9]+]] = fir.slice %[[V_56]], %[[V_57]], %[[C_1]] : (index, index, index) -> !fir.slice<1>
! CHECK:   %[[V_59:[0-9]+]] = fir.embox %[[V_4]](%[[V_5]]) [%[[V_58]]] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK:   %[[V_60:[0-9]+]] = fir.convert %[[V_59:[0-9]+]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
! CHECK:   %[[V_61:[0-9]+]] = fir.call @_FortranAioOutputDescriptor(%[[V_49]], %[[V_60]]) fastmath<contract> {{.*}}: (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:   %[[V_62:[0-9]+]] = fir.call @_FortranAioEndIoStatement(%[[V_49]]) fastmath<contract> {{.*}}: (!fir.ref<i8>) -> i32
! CHECK:   return
! CHECK:   }

! CHECK-LABEL:   func.func @_QPss3(
! CHECK-SAME:                      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant -1 : index
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 6 : i32
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 7.000000e+00 : f32
! CHECK:           %[[CONSTANT_4:.*]] = arith.constant 1 : i32
! CHECK:           %[[CONSTANT_5:.*]] = arith.constant 1 : index
! CHECK:           %[[CONSTANT_6:.*]] = arith.constant -2.000000e+00 : f32
! CHECK:           %[[CONSTANT_7:.*]] = arith.constant 0 : index
! CHECK:           %[[CONSTANT_8:.*]] = arith.constant 2 : index
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[ARG0]] : !fir.ref<i32>
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[LOAD_0]] : (i32) -> index
! CHECK:           %[[CMPI_0:.*]] = arith.cmpi sgt, %[[CONVERT_0]], %[[CONSTANT_7]] : index
! CHECK:           %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[CONVERT_0]], %[[CONSTANT_7]] : index
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.array<2x?xf32>, %[[SELECT_0]] {bindc_name = "aa", uniq_name = "_QFss3Eaa"}
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_8]], %[[SELECT_0]] : (index, index) -> !fir.shape<2>
! CHECK:           cf.br ^bb1(%[[CONSTANT_7]], %[[SELECT_0]] : index, index)
! CHECK:         ^bb1(%[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index):
! CHECK:           %[[CMPI_1:.*]] = arith.cmpi sgt, %[[VAL_1]], %[[CONSTANT_7]] : index
! CHECK:           cf.cond_br %[[CMPI_1]], ^bb2, ^bb6
! CHECK:         ^bb2:
! CHECK:           %[[ADDI_0:.*]] = arith.addi %[[VAL_0]], %[[CONSTANT_5]] : index
! CHECK:           cf.br ^bb3(%[[CONSTANT_7]], %[[CONSTANT_8]] : index, index)
! CHECK:         ^bb3(%[[VAL_2:.*]]: index, %[[VAL_3:.*]]: index):
! CHECK:           %[[CMPI_2:.*]] = arith.cmpi sgt, %[[VAL_3]], %[[CONSTANT_7]] : index
! CHECK:           cf.cond_br %[[CMPI_2]], ^bb4, ^bb5
! CHECK:         ^bb4:
! CHECK:           %[[ADDI_1:.*]] = arith.addi %[[VAL_2]], %[[CONSTANT_5]] : index
! CHECK:           %[[ARRAY_COOR_0:.*]] = fir.array_coor %[[ALLOCA_0]](%[[SHAPE_0]]) %[[ADDI_1]], %[[ADDI_0]] : (!fir.ref<!fir.array<2x?xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:           fir.store %[[CONSTANT_6]] to %[[ARRAY_COOR_0]] : !fir.ref<f32>
! CHECK:           %[[SUBI_0:.*]] = arith.subi %[[VAL_3]], %[[CONSTANT_5]] : index
! CHECK:           cf.br ^bb3(%[[ADDI_1]], %[[SUBI_0]] : index, index)
! CHECK:         ^bb5:
! CHECK:           %[[SUBI_1:.*]] = arith.subi %[[VAL_1]], %[[CONSTANT_5]] : index
! CHECK:           cf.br ^bb1(%[[ADDI_0]], %[[SUBI_1]] : index, index)
! CHECK:         ^bb6:
! CHECK:           %[[LOAD_1:.*]] = fir.load %[[ARG0]] : !fir.ref<i32>
! CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[LOAD_1]] : (i32) -> index
! CHECK:           %[[ADDI_2:.*]] = arith.addi %[[CONVERT_1]], %[[CONSTANT_0]] : index
! CHECK:           %[[CMPI_3:.*]] = arith.cmpi sgt, %[[ADDI_2]], %[[CONSTANT_7]] : index
! CHECK:           %[[SELECT_1:.*]] = arith.select %[[CMPI_3]], %[[ADDI_2]], %[[CONSTANT_7]] : index
! CHECK:           %[[SLICE_0:.*]] = fir.slice %[[CONSTANT_5]], %[[CONSTANT_8]], %[[CONSTANT_5]], %[[CONSTANT_8]], %[[CONVERT_1]], %[[CONSTANT_5]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:           %[[ALLOCMEM_0:.*]] = fir.allocmem !fir.array<2x?xf32>, %[[SELECT_0]]
! CHECK:           cf.br ^bb7(%[[CONSTANT_7]], %[[SELECT_0]] : index, index)
! CHECK:         ^bb7(%[[VAL_4:.*]]: index, %[[VAL_5:.*]]: index):
! CHECK:           %[[CMPI_4:.*]] = arith.cmpi sgt, %[[VAL_5]], %[[CONSTANT_7]] : index
! CHECK:           cf.cond_br %[[CMPI_4]], ^bb8, ^bb12
! CHECK:         ^bb8:
! CHECK:           %[[ADDI_3:.*]] = arith.addi %[[VAL_4]], %[[CONSTANT_5]] : index
! CHECK:           cf.br ^bb9(%[[CONSTANT_7]], %[[CONSTANT_8]] : index, index)
! CHECK:         ^bb9(%[[VAL_6:.*]]: index, %[[VAL_7:.*]]: index):
! CHECK:           %[[CMPI_5:.*]] = arith.cmpi sgt, %[[VAL_7]], %[[CONSTANT_7]] : index
! CHECK:           cf.cond_br %[[CMPI_5]], ^bb10, ^bb11
! CHECK:         ^bb10:
! CHECK:           %[[ADDI_4:.*]] = arith.addi %[[VAL_6]], %[[CONSTANT_5]] : index
! CHECK:           %[[ARRAY_COOR_1:.*]] = fir.array_coor %[[ALLOCA_0]](%[[SHAPE_0]]) %[[ADDI_4]], %[[ADDI_3]] : (!fir.ref<!fir.array<2x?xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:           %[[ARRAY_COOR_2:.*]] = fir.array_coor %[[ALLOCMEM_0]](%[[SHAPE_0]]) %[[ADDI_4]], %[[ADDI_3]] : (!fir.heap<!fir.array<2x?xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:           %[[LOAD_2:.*]] = fir.load %[[ARRAY_COOR_1]] : !fir.ref<f32>
! CHECK:           fir.store %[[LOAD_2]] to %[[ARRAY_COOR_2]] : !fir.ref<f32>
! CHECK:           %[[SUBI_2:.*]] = arith.subi %[[VAL_7]], %[[CONSTANT_5]] : index
! CHECK:           cf.br ^bb9(%[[ADDI_4]], %[[SUBI_2]] : index, index)
! CHECK:         ^bb11:
! CHECK:           %[[SUBI_3:.*]] = arith.subi %[[VAL_5]], %[[CONSTANT_5]] : index
! CHECK:           cf.br ^bb7(%[[ADDI_3]], %[[SUBI_3]] : index, index)
! CHECK:         ^bb12:
! CHECK:           %[[SUBI_4:.*]] = arith.subi %[[LOAD_1]], %[[CONSTANT_4]] : i32
! CHECK:           %[[CONVERT_2:.*]] = fir.convert %[[SUBI_4]] : (i32) -> index
! CHECK:           %[[SLICE_1:.*]] = fir.slice %[[CONSTANT_5]], %[[CONSTANT_8]], %[[CONSTANT_5]], %[[CONSTANT_5]], %[[CONVERT_2]], %[[CONSTANT_5]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:           cf.br ^bb13(%[[CONSTANT_7]], %[[SELECT_1]] : index, index)
! CHECK:         ^bb13(%[[VAL_8:.*]]: index, %[[VAL_9:.*]]: index):
! CHECK:           %[[CMPI_6:.*]] = arith.cmpi sgt, %[[VAL_9]], %[[CONSTANT_7]] : index
! CHECK:           cf.cond_br %[[CMPI_6]], ^bb14, ^bb18(%[[CONSTANT_7]], %[[SELECT_0]] : index, index)
! CHECK:         ^bb14:
! CHECK:           %[[ADDI_5:.*]] = arith.addi %[[VAL_8]], %[[CONSTANT_5]] : index
! CHECK:           cf.br ^bb15(%[[CONSTANT_7]], %[[CONSTANT_8]] : index, index)
! CHECK:         ^bb15(%[[VAL_10:.*]]: index, %[[VAL_11:.*]]: index):
! CHECK:           %[[CMPI_7:.*]] = arith.cmpi sgt, %[[VAL_11]], %[[CONSTANT_7]] : index
! CHECK:           cf.cond_br %[[CMPI_7]], ^bb16, ^bb17
! CHECK:         ^bb16:
! CHECK:           %[[ADDI_6:.*]] = arith.addi %[[VAL_10]], %[[CONSTANT_5]] : index
! CHECK:           %[[ARRAY_COOR_3:.*]] = fir.array_coor %[[ALLOCA_0]](%[[SHAPE_0]]) {{\[}}%[[SLICE_1]]] %[[ADDI_6]], %[[ADDI_5]] : (!fir.ref<!fir.array<2x?xf32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f32>
! CHECK:           %[[LOAD_3:.*]] = fir.load %[[ARRAY_COOR_3]] : !fir.ref<f32>
! CHECK:           %[[ADDF_0:.*]] = arith.addf %[[LOAD_3]], %[[CONSTANT_3]] fastmath<contract> : f32
! CHECK:           %[[ARRAY_COOR_4:.*]] = fir.array_coor %[[ALLOCMEM_0]](%[[SHAPE_0]]) {{\[}}%[[SLICE_0]]] %[[ADDI_6]], %[[ADDI_5]] : (!fir.heap<!fir.array<2x?xf32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f32>
! CHECK:           fir.store %[[ADDF_0]] to %[[ARRAY_COOR_4]] : !fir.ref<f32>
! CHECK:           %[[SUBI_5:.*]] = arith.subi %[[VAL_11]], %[[CONSTANT_5]] : index
! CHECK:           cf.br ^bb15(%[[ADDI_6]], %[[SUBI_5]] : index, index)
! CHECK:         ^bb17:
! CHECK:           %[[SUBI_6:.*]] = arith.subi %[[VAL_9]], %[[CONSTANT_5]] : index
! CHECK:           cf.br ^bb13(%[[ADDI_5]], %[[SUBI_6]] : index, index)
! CHECK:         ^bb18(%[[VAL_12:.*]]: index, %[[VAL_13:.*]]: index):
! CHECK:           %[[CMPI_8:.*]] = arith.cmpi sgt, %[[VAL_13]], %[[CONSTANT_7]] : index
! CHECK:           cf.cond_br %[[CMPI_8]], ^bb19, ^bb23
! CHECK:         ^bb19:
! CHECK:           %[[ADDI_7:.*]] = arith.addi %[[VAL_12]], %[[CONSTANT_5]] : index
! CHECK:           cf.br ^bb20(%[[CONSTANT_7]], %[[CONSTANT_8]] : index, index)
! CHECK:         ^bb20(%[[VAL_14:.*]]: index, %[[VAL_15:.*]]: index):
! CHECK:           %[[CMPI_9:.*]] = arith.cmpi sgt, %[[VAL_15]], %[[CONSTANT_7]] : index
! CHECK:           cf.cond_br %[[CMPI_9]], ^bb21, ^bb22
! CHECK:         ^bb21:
! CHECK:           %[[ADDI_8:.*]] = arith.addi %[[VAL_14]], %[[CONSTANT_5]] : index
! CHECK:           %[[ARRAY_COOR_5:.*]] = fir.array_coor %[[ALLOCMEM_0]](%[[SHAPE_0]]) %[[ADDI_8]], %[[ADDI_7]] : (!fir.heap<!fir.array<2x?xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:           %[[ARRAY_COOR_6:.*]] = fir.array_coor %[[ALLOCA_0]](%[[SHAPE_0]]) %[[ADDI_8]], %[[ADDI_7]] : (!fir.ref<!fir.array<2x?xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:           %[[LOAD_4:.*]] = fir.load %[[ARRAY_COOR_5]] : !fir.ref<f32>
! CHECK:           fir.store %[[LOAD_4]] to %[[ARRAY_COOR_6]] : !fir.ref<f32>
! CHECK:           %[[SUBI_7:.*]] = arith.subi %[[VAL_15]], %[[CONSTANT_5]] : index
! CHECK:           cf.br ^bb20(%[[ADDI_8]], %[[SUBI_7]] : index, index)
! CHECK:         ^bb22:
! CHECK:           %[[SUBI_8:.*]] = arith.subi %[[VAL_13]], %[[CONSTANT_5]] : index
! CHECK:           cf.br ^bb18(%[[ADDI_7]], %[[SUBI_8]] : index, index)
! CHECK:         ^bb23:
! CHECK:           fir.freemem %[[ALLOCMEM_0]] : !fir.heap<!fir.array<2x?xf32>>
! CHECK:           %[[ADDRESS_OF_0:.*]] = fir.address_of(@_QQcl{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:           %[[CONVERT_3:.*]] = fir.convert %[[ADDRESS_OF_0]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:           %[[CALL_0:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[CONSTANT_2]], %[[CONVERT_3]], %[[CONSTANT_1]]) fastmath<contract> {fir.llvm_memory = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = readwrite, errnoMem = none, targetMem0 = none, targetMem1 = none>, llvm.nocallback, llvm.nosync} : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:           %[[SLICE_2:.*]] = fir.slice %[[CONSTANT_5]], %[[CONSTANT_8]], %[[CONSTANT_5]], %[[CONSTANT_5]], %[[CONSTANT_8]], %[[CONSTANT_5]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:           %[[EMBOX_0:.*]] = fir.embox %[[ALLOCA_0]](%[[SHAPE_0]]) {{\[}}%[[SLICE_2]]] : (!fir.ref<!fir.array<2x?xf32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<?x2xf32>>
! CHECK:           %[[CONVERT_4:.*]] = fir.convert %[[EMBOX_0]] : (!fir.box<!fir.array<?x2xf32>>) -> !fir.box<none>
! CHECK:           %[[CALL_1:.*]] = fir.call @_FortranAioOutputDescriptor(%[[CALL_0]], %[[CONVERT_4]]) fastmath<contract> {llvm.nocallback, llvm.nosync} : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:           %[[LOAD_5:.*]] = fir.load %[[ARG0]] : !fir.ref<i32>
! CHECK:           %[[SUBI_9:.*]] = arith.subi %[[LOAD_5]], %[[CONSTANT_4]] : i32
! CHECK:           %[[CONVERT_5:.*]] = fir.convert %[[SUBI_9]] : (i32) -> index
! CHECK:           %[[CONVERT_6:.*]] = fir.convert %[[LOAD_5]] : (i32) -> index
! CHECK:           %[[SLICE_3:.*]] = fir.slice %[[CONSTANT_5]], %[[CONSTANT_8]], %[[CONSTANT_5]], %[[CONVERT_5]], %[[CONVERT_6]], %[[CONSTANT_5]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:           %[[EMBOX_1:.*]] = fir.embox %[[ALLOCA_0]](%[[SHAPE_0]]) {{\[}}%[[SLICE_3]]] : (!fir.ref<!fir.array<2x?xf32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<?x?xf32>>
! CHECK:           %[[CONVERT_7:.*]] = fir.convert %[[EMBOX_1]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
! CHECK:           %[[CALL_2:.*]] = fir.call @_FortranAioOutputDescriptor(%[[CALL_0]], %[[CONVERT_7]]) fastmath<contract> {llvm.nocallback, llvm.nosync} : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:           %[[CALL_3:.*]] = fir.call @_FortranAioEndIoStatement(%[[CALL_0]]) fastmath<contract> {fir.llvm_memory = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = readwrite, errnoMem = none, targetMem0 = none, targetMem1 = none>, llvm.nocallback, llvm.nosync} : (!fir.ref<i8>) -> i32
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPss4(
! CHECK-SAME:                      %[[ARG0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant -1 : index
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 2 : index
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant {{.*}} : i32
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 6 : i32
! CHECK:           %[[CONSTANT_4:.*]] = arith.constant 7.000000e+00 : f32
! CHECK:           %[[CONSTANT_5:.*]] = arith.constant 1 : i32
! CHECK:           %[[CONSTANT_6:.*]] = arith.constant 1 : index
! CHECK:           %[[CONSTANT_7:.*]] = arith.constant -2.000000e+00 : f32
! CHECK:           %[[CONSTANT_8:.*]] = arith.constant 0 : index
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[ARG0]] : !fir.ref<i32>
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[LOAD_0]] : (i32) -> index
! CHECK:           %[[CMPI_0:.*]] = arith.cmpi sgt, %[[CONVERT_0]], %[[CONSTANT_8]] : index
! CHECK:           %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[CONVERT_0]], %[[CONSTANT_8]] : index
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.array<?x2xf32>, %[[SELECT_0]] {bindc_name = "aa", uniq_name = "_QFss4Eaa"}
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[SELECT_0]], %[[CONSTANT_1]] : (index, index) -> !fir.shape<2>
! CHECK:           cf.br ^bb1(%[[CONSTANT_8]], %[[CONSTANT_1]] : index, index)
! CHECK:         ^bb1(%[[VAL_0:.*]]: index, %[[VAL_1:.*]]: index):
! CHECK:           %[[CMPI_1:.*]] = arith.cmpi sgt, %[[VAL_1]], %[[CONSTANT_8]] : index
! CHECK:           cf.cond_br %[[CMPI_1]], ^bb2, ^bb6
! CHECK:         ^bb2:
! CHECK:           %[[ADDI_0:.*]] = arith.addi %[[VAL_0]], %[[CONSTANT_6]] : index
! CHECK:           cf.br ^bb3(%[[CONSTANT_8]], %[[SELECT_0]] : index, index)
! CHECK:         ^bb3(%[[VAL_2:.*]]: index, %[[VAL_3:.*]]: index):
! CHECK:           %[[CMPI_2:.*]] = arith.cmpi sgt, %[[VAL_3]], %[[CONSTANT_8]] : index
! CHECK:           cf.cond_br %[[CMPI_2]], ^bb4, ^bb5
! CHECK:         ^bb4:
! CHECK:           %[[ADDI_1:.*]] = arith.addi %[[VAL_2]], %[[CONSTANT_6]] : index
! CHECK:           %[[ARRAY_COOR_0:.*]] = fir.array_coor %[[ALLOCA_0]](%[[SHAPE_0]]) %[[ADDI_1]], %[[ADDI_0]] : (!fir.ref<!fir.array<?x2xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:           fir.store %[[CONSTANT_7]] to %[[ARRAY_COOR_0]] : !fir.ref<f32>
! CHECK:           %[[SUBI_0:.*]] = arith.subi %[[VAL_3]], %[[CONSTANT_6]] : index
! CHECK:           cf.br ^bb3(%[[ADDI_1]], %[[SUBI_0]] : index, index)
! CHECK:         ^bb5:
! CHECK:           %[[SUBI_1:.*]] = arith.subi %[[VAL_1]], %[[CONSTANT_6]] : index
! CHECK:           cf.br ^bb1(%[[ADDI_0]], %[[SUBI_1]] : index, index)
! CHECK:         ^bb6:
! CHECK:           %[[LOAD_1:.*]] = fir.load %[[ARG0]] : !fir.ref<i32>
! CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[LOAD_1]] : (i32) -> index
! CHECK:           %[[ADDI_2:.*]] = arith.addi %[[CONVERT_1]], %[[CONSTANT_0]] : index
! CHECK:           %[[CMPI_3:.*]] = arith.cmpi sgt, %[[ADDI_2]], %[[CONSTANT_8]] : index
! CHECK:           %[[SELECT_1:.*]] = arith.select %[[CMPI_3]], %[[ADDI_2]], %[[CONSTANT_8]] : index
! CHECK:           %[[SLICE_0:.*]] = fir.slice %[[CONSTANT_1]], %[[CONVERT_1]], %[[CONSTANT_6]], %[[CONSTANT_6]], %[[CONSTANT_1]], %[[CONSTANT_6]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:           %[[ALLOCMEM_0:.*]] = fir.allocmem !fir.array<?x2xf32>, %[[SELECT_0]]
! CHECK:           cf.br ^bb7(%[[CONSTANT_8]], %[[CONSTANT_1]] : index, index)
! CHECK:         ^bb7(%[[VAL_4:.*]]: index, %[[VAL_5:.*]]: index):
! CHECK:           %[[CMPI_4:.*]] = arith.cmpi sgt, %[[VAL_5]], %[[CONSTANT_8]] : index
! CHECK:           cf.cond_br %[[CMPI_4]], ^bb8, ^bb12
! CHECK:         ^bb8:
! CHECK:           %[[ADDI_3:.*]] = arith.addi %[[VAL_4]], %[[CONSTANT_6]] : index
! CHECK:           cf.br ^bb9(%[[CONSTANT_8]], %[[SELECT_0]] : index, index)
! CHECK:         ^bb9(%[[VAL_6:.*]]: index, %[[VAL_7:.*]]: index):
! CHECK:           %[[CMPI_5:.*]] = arith.cmpi sgt, %[[VAL_7]], %[[CONSTANT_8]] : index
! CHECK:           cf.cond_br %[[CMPI_5]], ^bb10, ^bb11
! CHECK:         ^bb10:
! CHECK:           %[[ADDI_4:.*]] = arith.addi %[[VAL_6]], %[[CONSTANT_6]] : index
! CHECK:           %[[ARRAY_COOR_1:.*]] = fir.array_coor %[[ALLOCA_0]](%[[SHAPE_0]]) %[[ADDI_4]], %[[ADDI_3]] : (!fir.ref<!fir.array<?x2xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:           %[[ARRAY_COOR_2:.*]] = fir.array_coor %[[ALLOCMEM_0]](%[[SHAPE_0]]) %[[ADDI_4]], %[[ADDI_3]] : (!fir.heap<!fir.array<?x2xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:           %[[LOAD_2:.*]] = fir.load %[[ARRAY_COOR_1]] : !fir.ref<f32>
! CHECK:           fir.store %[[LOAD_2]] to %[[ARRAY_COOR_2]] : !fir.ref<f32>
! CHECK:           %[[SUBI_2:.*]] = arith.subi %[[VAL_7]], %[[CONSTANT_6]] : index
! CHECK:           cf.br ^bb9(%[[ADDI_4]], %[[SUBI_2]] : index, index)
! CHECK:         ^bb11:
! CHECK:           %[[SUBI_3:.*]] = arith.subi %[[VAL_5]], %[[CONSTANT_6]] : index
! CHECK:           cf.br ^bb7(%[[ADDI_3]], %[[SUBI_3]] : index, index)
! CHECK:         ^bb12:
! CHECK:           %[[SUBI_4:.*]] = arith.subi %[[LOAD_1]], %[[CONSTANT_5]] : i32
! CHECK:           %[[CONVERT_2:.*]] = fir.convert %[[SUBI_4]] : (i32) -> index
! CHECK:           %[[SLICE_1:.*]] = fir.slice %[[CONSTANT_6]], %[[CONVERT_2]], %[[CONSTANT_6]], %[[CONSTANT_6]], %[[CONSTANT_1]], %[[CONSTANT_6]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:           cf.br ^bb13(%[[CONSTANT_8]], %[[CONSTANT_1]] : index, index)
! CHECK:         ^bb13(%[[VAL_8:.*]]: index, %[[VAL_9:.*]]: index):
! CHECK:           %[[CMPI_6:.*]] = arith.cmpi sgt, %[[VAL_9]], %[[CONSTANT_8]] : index
! CHECK:           cf.cond_br %[[CMPI_6]], ^bb14, ^bb18(%[[CONSTANT_8]], %[[CONSTANT_1]] : index, index)
! CHECK:         ^bb14:
! CHECK:           %[[ADDI_5:.*]] = arith.addi %[[VAL_8]], %[[CONSTANT_6]] : index
! CHECK:           cf.br ^bb15(%[[CONSTANT_8]], %[[SELECT_1]] : index, index)
! CHECK:         ^bb15(%[[VAL_10:.*]]: index, %[[VAL_11:.*]]: index):
! CHECK:           %[[CMPI_7:.*]] = arith.cmpi sgt, %[[VAL_11]], %[[CONSTANT_8]] : index
! CHECK:           cf.cond_br %[[CMPI_7]], ^bb16, ^bb17
! CHECK:         ^bb16:
! CHECK:           %[[ADDI_6:.*]] = arith.addi %[[VAL_10]], %[[CONSTANT_6]] : index
! CHECK:           %[[ARRAY_COOR_3:.*]] = fir.array_coor %[[ALLOCA_0]](%[[SHAPE_0]]) {{\[}}%[[SLICE_1]]] %[[ADDI_6]], %[[ADDI_5]] : (!fir.ref<!fir.array<?x2xf32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f32>
! CHECK:           %[[LOAD_3:.*]] = fir.load %[[ARRAY_COOR_3]] : !fir.ref<f32>
! CHECK:           %[[ADDF_0:.*]] = arith.addf %[[LOAD_3]], %[[CONSTANT_4]] fastmath<contract> : f32
! CHECK:           %[[ARRAY_COOR_4:.*]] = fir.array_coor %[[ALLOCMEM_0]](%[[SHAPE_0]]) {{\[}}%[[SLICE_0]]] %[[ADDI_6]], %[[ADDI_5]] : (!fir.heap<!fir.array<?x2xf32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f32>
! CHECK:           fir.store %[[ADDF_0]] to %[[ARRAY_COOR_4]] : !fir.ref<f32>
! CHECK:           %[[SUBI_5:.*]] = arith.subi %[[VAL_11]], %[[CONSTANT_6]] : index
! CHECK:           cf.br ^bb15(%[[ADDI_6]], %[[SUBI_5]] : index, index)
! CHECK:         ^bb17:
! CHECK:           %[[SUBI_6:.*]] = arith.subi %[[VAL_9]], %[[CONSTANT_6]] : index
! CHECK:           cf.br ^bb13(%[[ADDI_5]], %[[SUBI_6]] : index, index)
! CHECK:         ^bb18(%[[VAL_12:.*]]: index, %[[VAL_13:.*]]: index):
! CHECK:           %[[CMPI_8:.*]] = arith.cmpi sgt, %[[VAL_13]], %[[CONSTANT_8]] : index
! CHECK:           cf.cond_br %[[CMPI_8]], ^bb19, ^bb23
! CHECK:         ^bb19:
! CHECK:           %[[ADDI_7:.*]] = arith.addi %[[VAL_12]], %[[CONSTANT_6]] : index
! CHECK:           cf.br ^bb20(%[[CONSTANT_8]], %[[SELECT_0]] : index, index)
! CHECK:         ^bb20(%[[VAL_14:.*]]: index, %[[VAL_15:.*]]: index):
! CHECK:           %[[CMPI_9:.*]] = arith.cmpi sgt, %[[VAL_15]], %[[CONSTANT_8]] : index
! CHECK:           cf.cond_br %[[CMPI_9]], ^bb21, ^bb22
! CHECK:         ^bb21:
! CHECK:           %[[ADDI_8:.*]] = arith.addi %[[VAL_14]], %[[CONSTANT_6]] : index
! CHECK:           %[[ARRAY_COOR_5:.*]] = fir.array_coor %[[ALLOCMEM_0]](%[[SHAPE_0]]) %[[ADDI_8]], %[[ADDI_7]] : (!fir.heap<!fir.array<?x2xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:           %[[ARRAY_COOR_6:.*]] = fir.array_coor %[[ALLOCA_0]](%[[SHAPE_0]]) %[[ADDI_8]], %[[ADDI_7]] : (!fir.ref<!fir.array<?x2xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:           %[[LOAD_4:.*]] = fir.load %[[ARRAY_COOR_5]] : !fir.ref<f32>
! CHECK:           fir.store %[[LOAD_4]] to %[[ARRAY_COOR_6]] : !fir.ref<f32>
! CHECK:           %[[SUBI_7:.*]] = arith.subi %[[VAL_15]], %[[CONSTANT_6]] : index
! CHECK:           cf.br ^bb20(%[[ADDI_8]], %[[SUBI_7]] : index, index)
! CHECK:         ^bb22:
! CHECK:           %[[SUBI_8:.*]] = arith.subi %[[VAL_13]], %[[CONSTANT_6]] : index
! CHECK:           cf.br ^bb18(%[[ADDI_7]], %[[SUBI_8]] : index, index)
! CHECK:         ^bb23:
! CHECK:           fir.freemem %[[ALLOCMEM_0]] : !fir.heap<!fir.array<?x2xf32>>
! CHECK:           %[[ADDRESS_OF_0:.*]] = fir.address_of(@_QQcl{{.*}}) : !fir.ref<!fir.char<1,{{.*}}>>
! CHECK:           %[[CONVERT_3:.*]] = fir.convert %[[ADDRESS_OF_0]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
! CHECK:           %[[CALL_0:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[CONSTANT_3]], %[[CONVERT_3]], %[[CONSTANT_2]]) fastmath<contract> {fir.llvm_memory = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = readwrite, errnoMem = none, targetMem0 = none, targetMem1 = none>, llvm.nocallback, llvm.nosync} : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:           %[[SLICE_2:.*]] = fir.slice %[[CONSTANT_6]], %[[CONSTANT_1]], %[[CONSTANT_6]], %[[CONSTANT_6]], %[[CONSTANT_1]], %[[CONSTANT_6]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:           %[[EMBOX_0:.*]] = fir.embox %[[ALLOCA_0]](%[[SHAPE_0]]) {{\[}}%[[SLICE_2]]] : (!fir.ref<!fir.array<?x2xf32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<2x?xf32>>
! CHECK:           %[[CONVERT_4:.*]] = fir.convert %[[EMBOX_0]] : (!fir.box<!fir.array<2x?xf32>>) -> !fir.box<none>
! CHECK:           %[[CALL_1:.*]] = fir.call @_FortranAioOutputDescriptor(%[[CALL_0]], %[[CONVERT_4]]) fastmath<contract> {llvm.nocallback, llvm.nosync} : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:           %[[LOAD_5:.*]] = fir.load %[[ARG0]] : !fir.ref<i32>
! CHECK:           %[[SUBI_9:.*]] = arith.subi %[[LOAD_5]], %[[CONSTANT_5]] : i32
! CHECK:           %[[CONVERT_5:.*]] = fir.convert %[[SUBI_9]] : (i32) -> index
! CHECK:           %[[CONVERT_6:.*]] = fir.convert %[[LOAD_5]] : (i32) -> index
! CHECK:           %[[SLICE_3:.*]] = fir.slice %[[CONVERT_5]], %[[CONVERT_6]], %[[CONSTANT_6]], %[[CONSTANT_6]], %[[CONSTANT_1]], %[[CONSTANT_6]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:           %[[EMBOX_1:.*]] = fir.embox %[[ALLOCA_0]](%[[SHAPE_0]]) {{\[}}%[[SLICE_3]]] : (!fir.ref<!fir.array<?x2xf32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<?x?xf32>>
! CHECK:           %[[CONVERT_7:.*]] = fir.convert %[[EMBOX_1]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
! CHECK:           %[[CALL_2:.*]] = fir.call @_FortranAioOutputDescriptor(%[[CALL_0]], %[[CONVERT_7]]) fastmath<contract> {llvm.nocallback, llvm.nosync} : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:           %[[CALL_3:.*]] = fir.call @_FortranAioEndIoStatement(%[[CALL_0]]) fastmath<contract> {fir.llvm_memory = #llvm.memory_effects<other = none, argMem = readwrite, inaccessibleMem = readwrite, errnoMem = none, targetMem0 = none, targetMem1 = none>, llvm.nocallback, llvm.nosync} : (!fir.ref<i8>) -> i32
! CHECK:           return
! CHECK:         }

! CHECK-LABEL: func @_QPtt1
subroutine tt1
  ! CHECK: fir.call @_FortranAioBeginExternalListOutput
  ! CHECK: %[[temp3:[0-9]+]] = fir.allocmem !fir.array<3xf32>
  ! CHECK: br ^bb1(%[[temp3]]
  ! CHECK-NEXT: ^bb1(%[[temp3arg:[0-9]+]]: !fir.heap<!fir.array<3xf32>>
  ! CHECK: %[[temp1:[0-9]+]] = fir.allocmem !fir.array<1xf32>
  ! CHECK: fir.call @_QFtt1Pr
  ! CHECK: fir.call @realloc
  ! CHECK: fir.freemem %[[temp1]] : !fir.heap<!fir.array<1xf32>>
  ! CHECK: %[[temp3x:[0-9]+]] = fir.allocmem !fir.array<3xf32>
  ! CHECK: fir.call @_FortranAioOutputDescriptor
  ! CHECK-NEXT: fir.freemem %[[temp3x]] : !fir.heap<!fir.array<3xf32>>
  ! CHECK-NEXT: fir.freemem %[[temp3arg]] : !fir.heap<!fir.array<3xf32>>
  ! CHECK-NEXT: fir.call @_FortranAioEndIoStatement
  print*, [(r([7.0]),i=1,3)]
contains
  ! CHECK-LABEL: func private @_QFtt1Pr
  function r(x)
    real x(:)
    r = x(1)
  end
end
