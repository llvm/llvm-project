! RUN: bbc -hlfir=false %s -o - | FileCheck %s

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
! CHECK:   %[[V_53:[0-9]+]] = fir.call @_FortranAioOutputDescriptor(%[[V_49]], %[[V_52]]) fastmath<contract> : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:   %[[V_54:[0-9]+]] = fir.load %arg0 : !fir.ref<i32>
! CHECK:   %[[V_55:[0-9]+]] = arith.subi %[[V_54]], %[[C_1_i32]] : i32
! CHECK:   %[[V_56:[0-9]+]] = fir.convert %[[V_55:[0-9]+]] : (i32) -> index
! CHECK:   %[[V_57:[0-9]+]] = fir.convert %[[V_54:[0-9]+]] : (i32) -> index
! CHECK:   %[[V_58:[0-9]+]] = fir.slice %[[V_56]], %[[V_57]], %[[C_1]] : (index, index, index) -> !fir.slice<1>
! CHECK:   %[[V_59:[0-9]+]] = fir.embox %[[V_4]](%[[V_5]]) [%[[V_58]]] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK:   %[[V_60:[0-9]+]] = fir.convert %[[V_59:[0-9]+]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
! CHECK:   %[[V_61:[0-9]+]] = fir.call @_FortranAioOutputDescriptor(%[[V_49]], %[[V_60]]) fastmath<contract> : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:   %[[V_62:[0-9]+]] = fir.call @_FortranAioEndIoStatement(%[[V_49]]) fastmath<contract> : (!fir.ref<i8>) -> i32
! CHECK:   return
! CHECK:   }

! CHECK-LABEL: func @_QPss3(
! CHECK-SAME:               %arg0: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK-DAG: %[[C_m1:[-0-9a-z_]+]] = arith.constant -1 : index
! CHECK-DAG: %[[C_2:[-0-9a-z_]+]] = arith.constant 2 : index
! CHECK-DAG: %[[C_1:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK-DAG: %[[C_34_i32:[-0-9a-z_]+]] = arith.constant 34 : i32
! CHECK-DAG: %[[C_6_i32:[-0-9a-z_]+]] = arith.constant 6 : i32
! CHECK-DAG: %[[C_st:[-0-9a-z_]+]] = arith.constant 7.000000e+00 : f32
! CHECK-DAG: %[[C_1_i32:[-0-9a-z_]+]] = arith.constant 1 : i32
! CHECK-DAG: %[[C_st_0:[-0-9a-z_]+]] = arith.constant -2.000000e+00 : f32
! CHECK-DAG: %[[C_0:[-0-9a-z_]+]] = arith.constant 0 : index
! CHECK:   %[[V_0:[0-9]+]] = fir.load %arg0 : !fir.ref<i32>
! CHECK:   %[[V_1:[0-9]+]] = fir.convert %[[V_0:[0-9]+]] : (i32) -> index
! CHECK:   %[[V_2:[0-9]+]] = arith.cmpi sgt, %[[V_1]], %[[C_0]] : index
! CHECK:   %[[V_3:[0-9]+]] = arith.select %[[V_2]], %[[V_1]], %[[C_0]] : index
! CHECK:   %[[V_4:[0-9]+]] = fir.alloca !fir.array<2x?xf32>, %[[V_3]] {bindc_name = "aa", uniq_name = "_QFss3Eaa"}
! CHECK:   %[[V_5:[0-9]+]] = fir.shape %[[C_2]], %[[V_3:[0-9]+]] : (index, index) -> !fir.shape<2>
! CHECK:   cf.br ^bb1(%[[C_0]], %[[V_3:[0-9]+]] : index, index)
! CHECK: ^bb1(%[[V_6:[0-9]+]]: index, %[[V_7:[0-9]+]]: index):  // 2 preds: ^bb0, ^bb4
! CHECK:   %[[V_8:[0-9]+]] = arith.cmpi sgt, %[[V_7]], %[[C_0]] : index
! CHECK:   cf.cond_br %[[V_8]], ^bb2(%[[C_0]], %[[C_2]] : index, index), ^bb5
! CHECK: ^bb2(%[[V_9:[0-9]+]]: index, %[[V_10:[0-9]+]]: index):  // 2 preds: ^bb1, ^bb3
! CHECK:   %[[V_11:[0-9]+]] = arith.cmpi sgt, %[[V_10]], %[[C_0]] : index
! CHECK:   cf.cond_br %[[V_11]], ^bb3, ^bb4
! CHECK: ^bb3:  // pred: ^bb2
! CHECK:   %[[V_12:[0-9]+]] = arith.addi %[[V_9]], %[[C_1]] : index
! CHECK:   %[[V_13:[0-9]+]] = arith.addi %[[V_6]], %[[C_1]] : index
! CHECK:   %[[V_14:[0-9]+]] = fir.array_coor %[[V_4]](%[[V_5]]) %[[V_12]], %[[V_13:[0-9]+]] : (!fir.ref<!fir.array<2x?xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:   fir.store %[[C_st_0]] to %[[V_14:[0-9]+]] : !fir.ref<f32>
! CHECK:   %[[V_15:[0-9]+]] = arith.subi %[[V_10]], %[[C_1]] : index
! CHECK:   cf.br ^bb2(%[[V_12]], %[[V_15:[0-9]+]] : index, index)
! CHECK: ^bb4:  // pred: ^bb2
! CHECK:   %[[V_16:[0-9]+]] = arith.addi %[[V_6]], %[[C_1]] : index
! CHECK:   %[[V_17:[0-9]+]] = arith.subi %[[V_7]], %[[C_1]] : index
! CHECK:   cf.br ^bb1(%[[V_16]], %[[V_17:[0-9]+]] : index, index)
! CHECK: ^bb5:  // pred: ^bb1
! CHECK:   %[[V_18:[0-9]+]] = fir.load %arg0 : !fir.ref<i32>
! CHECK:   %[[V_19:[0-9]+]] = fir.convert %[[V_18:[0-9]+]] : (i32) -> index
! CHECK:   %[[V_20:[0-9]+]] = arith.addi %[[V_19]], %[[C_m1]] : index
! CHECK:   %[[V_21:[0-9]+]] = arith.cmpi sgt, %[[V_20]], %[[C_0]] : index
! CHECK:   %[[V_22:[0-9]+]] = arith.select %[[V_21]], %[[V_20]], %[[C_0]] : index
! CHECK:   %[[V_23:[0-9]+]] = fir.slice %[[C_1]], %[[C_2]], %[[C_1]], %[[C_2]], %[[V_19]], %[[C_1]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:   %[[V_24:[0-9]+]] = fir.allocmem !fir.array<2x?xf32>, %[[V_3]]
! CHECK:   cf.br ^bb6(%[[C_0]], %[[V_3:[0-9]+]] : index, index)
! CHECK: ^bb6(%[[V_25:[0-9]+]]: index, %[[V_26:[0-9]+]]: index):  // 2 preds: ^bb5, ^bb9
! CHECK:   %[[V_27:[0-9]+]] = arith.cmpi sgt, %[[V_26]], %[[C_0]] : index
! CHECK:   cf.cond_br %[[V_27]], ^bb7(%[[C_0]], %[[C_2]] : index, index), ^bb10
! CHECK: ^bb7(%[[V_28:[0-9]+]]: index, %[[V_29:[0-9]+]]: index):  // 2 preds: ^bb6, ^bb8
! CHECK:   %[[V_30:[0-9]+]] = arith.cmpi sgt, %[[V_29]], %[[C_0]] : index
! CHECK:   cf.cond_br %[[V_30]], ^bb8, ^bb9
! CHECK: ^bb8:  // pred: ^bb7
! CHECK:   %[[V_31:[0-9]+]] = arith.addi %[[V_28]], %[[C_1]] : index
! CHECK:   %[[V_32:[0-9]+]] = arith.addi %[[V_25]], %[[C_1]] : index
! CHECK:   %[[V_33:[0-9]+]] = fir.array_coor %[[V_4]](%[[V_5]]) %[[V_31]], %[[V_32:[0-9]+]] : (!fir.ref<!fir.array<2x?xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:   %[[V_34:[0-9]+]] = fir.array_coor %[[V_24]](%[[V_5]]) %[[V_31]], %[[V_32:[0-9]+]] : (!fir.heap<!fir.array<2x?xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:   %[[V_35:[0-9]+]] = fir.load %[[V_33:[0-9]+]] : !fir.ref<f32>
! CHECK:   fir.store %[[V_35]] to %[[V_34:[0-9]+]] : !fir.ref<f32>
! CHECK:   %[[V_36:[0-9]+]] = arith.subi %[[V_29]], %[[C_1]] : index
! CHECK:   cf.br ^bb7(%[[V_31]], %[[V_36:[0-9]+]] : index, index)
! CHECK: ^bb9:  // pred: ^bb7
! CHECK:   %[[V_37:[0-9]+]] = arith.addi %[[V_25]], %[[C_1]] : index
! CHECK:   %[[V_38:[0-9]+]] = arith.subi %[[V_26]], %[[C_1]] : index
! CHECK:   cf.br ^bb6(%[[V_37]], %[[V_38:[0-9]+]] : index, index)
! CHECK: ^bb10:  // pred: ^bb6
! CHECK:   %[[V_39:[0-9]+]] = arith.subi %[[V_18]], %[[C_1_i32]] : i32
! CHECK:   %[[V_40:[0-9]+]] = fir.convert %[[V_39:[0-9]+]] : (i32) -> index
! CHECK:   %[[V_41:[0-9]+]] = fir.slice %[[C_1]], %[[C_2]], %[[C_1]], %[[C_1]], %[[V_40]], %[[C_1]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:   cf.br ^bb11(%[[C_0]], %[[V_22:[0-9]+]] : index, index)
! CHECK: ^bb11(%[[V_42:[0-9]+]]: index, %[[V_43:[0-9]+]]: index):  // 2 preds: ^bb10, ^bb14
! CHECK:   %[[V_44:[0-9]+]] = arith.cmpi sgt, %[[V_43]], %[[C_0]] : index
! CHECK:   cf.cond_br %[[V_44]], ^bb12(%[[C_0]], %[[C_2]] : index, index), ^bb15(%[[C_0]], %[[V_3:[0-9]+]] : index, index)
! CHECK: ^bb12(%[[V_45:[0-9]+]]: index, %[[V_46:[0-9]+]]: index):  // 2 preds: ^bb11, ^bb13
! CHECK:   %[[V_47:[0-9]+]] = arith.cmpi sgt, %[[V_46]], %[[C_0]] : index
! CHECK:   cf.cond_br %[[V_47]], ^bb13, ^bb14
! CHECK: ^bb13:  // pred: ^bb12
! CHECK:   %[[V_48:[0-9]+]] = arith.addi %[[V_45]], %[[C_1]] : index
! CHECK:   %[[V_49:[0-9]+]] = arith.addi %[[V_42]], %[[C_1]] : index
! CHECK:   %[[V_50:[0-9]+]] = fir.array_coor %[[V_4]](%[[V_5]]) [%[[V_41]]] %[[V_48]], %[[V_49:[0-9]+]] : (!fir.ref<!fir.array<2x?xf32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f32>
! CHECK:   %[[V_51:[0-9]+]] = fir.load %[[V_50:[0-9]+]] : !fir.ref<f32>
! CHECK:   %[[V_52:[0-9]+]] = arith.addf %[[V_51]], %[[C_st]] fastmath<contract> : f32
! CHECK:   %[[V_53:[0-9]+]] = fir.array_coor %[[V_24]](%[[V_5]]) [%[[V_23]]] %[[V_48]], %[[V_49:[0-9]+]] : (!fir.heap<!fir.array<2x?xf32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f32>
! CHECK:   fir.store %[[V_52]] to %[[V_53:[0-9]+]] : !fir.ref<f32>
! CHECK:   %[[V_54:[0-9]+]] = arith.subi %[[V_46]], %[[C_1]] : index
! CHECK:   cf.br ^bb12(%[[V_48]], %[[V_54:[0-9]+]] : index, index)
! CHECK: ^bb14:  // pred: ^bb12
! CHECK:   %[[V_55:[0-9]+]] = arith.addi %[[V_42]], %[[C_1]] : index
! CHECK:   %[[V_56:[0-9]+]] = arith.subi %[[V_43]], %[[C_1]] : index
! CHECK:   cf.br ^bb11(%[[V_55]], %[[V_56:[0-9]+]] : index, index)
! CHECK: ^bb15(%[[V_57:[0-9]+]]: index, %[[V_58:[0-9]+]]: index):  // 2 preds: ^bb11, ^bb18
! CHECK:   %[[V_59:[0-9]+]] = arith.cmpi sgt, %[[V_58]], %[[C_0]] : index
! CHECK:   cf.cond_br %[[V_59]], ^bb16(%[[C_0]], %[[C_2]] : index, index), ^bb19
! CHECK: ^bb16(%[[V_60:[0-9]+]]: index, %[[V_61:[0-9]+]]: index):  // 2 preds: ^bb15, ^bb17
! CHECK:   %[[V_62:[0-9]+]] = arith.cmpi sgt, %[[V_61]], %[[C_0]] : index
! CHECK:   cf.cond_br %[[V_62]], ^bb17, ^bb18
! CHECK: ^bb17:  // pred: ^bb16
! CHECK:   %[[V_63:[0-9]+]] = arith.addi %[[V_60]], %[[C_1]] : index
! CHECK:   %[[V_64:[0-9]+]] = arith.addi %[[V_57]], %[[C_1]] : index
! CHECK:   %[[V_65:[0-9]+]] = fir.array_coor %[[V_24]](%[[V_5]]) %[[V_63]], %[[V_64:[0-9]+]] : (!fir.heap<!fir.array<2x?xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:   %[[V_66:[0-9]+]] = fir.array_coor %[[V_4]](%[[V_5]]) %[[V_63]], %[[V_64:[0-9]+]] : (!fir.ref<!fir.array<2x?xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:   %[[V_67:[0-9]+]] = fir.load %[[V_65:[0-9]+]] : !fir.ref<f32>
! CHECK:   fir.store %[[V_67]] to %[[V_66:[0-9]+]] : !fir.ref<f32>
! CHECK:   %[[V_68:[0-9]+]] = arith.subi %[[V_61]], %[[C_1]] : index
! CHECK:   cf.br ^bb16(%[[V_63]], %[[V_68:[0-9]+]] : index, index)
! CHECK: ^bb18:  // pred: ^bb16
! CHECK:   %[[V_69:[0-9]+]] = arith.addi %[[V_57]], %[[C_1]] : index
! CHECK:   %[[V_70:[0-9]+]] = arith.subi %[[V_58]], %[[C_1]] : index
! CHECK:   cf.br ^bb15(%[[V_69]], %[[V_70:[0-9]+]] : index, index)
! CHECK: ^bb19:  // pred: ^bb15
! CHECK:   fir.freemem %[[V_24:[0-9]+]] : !fir.heap<!fir.array<2x?xf32>>
! CHECK:   %[[V_73:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput(%[[C_6_i32]], %{{.*}}, %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:   %[[V_74:[0-9]+]] = fir.slice %[[C_1]], %[[C_2]], %[[C_1]], %[[C_1]], %[[C_2]], %[[C_1]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:   %[[V_75:[0-9]+]] = fir.embox %[[V_4]](%[[V_5]]) [%[[V_74]]] : (!fir.ref<!fir.array<2x?xf32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<?x2xf32>>
! CHECK:   %[[V_76:[0-9]+]] = fir.convert %[[V_75:[0-9]+]] : (!fir.box<!fir.array<?x2xf32>>) -> !fir.box<none>
! CHECK:   %[[V_77:[0-9]+]] = fir.call @_FortranAioOutputDescriptor(%[[V_73]], %[[V_76]]) fastmath<contract> : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:   %[[V_78:[0-9]+]] = fir.load %arg0 : !fir.ref<i32>
! CHECK:   %[[V_79:[0-9]+]] = arith.subi %[[V_78]], %[[C_1_i32]] : i32
! CHECK:   %[[V_80:[0-9]+]] = fir.convert %[[V_79:[0-9]+]] : (i32) -> index
! CHECK:   %[[V_81:[0-9]+]] = fir.convert %[[V_78:[0-9]+]] : (i32) -> index
! CHECK:   %[[V_82:[0-9]+]] = fir.slice %[[C_1]], %[[C_2]], %[[C_1]], %[[V_80]], %[[V_81]], %[[C_1]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:   %[[V_83:[0-9]+]] = fir.embox %[[V_4]](%[[V_5]]) [%[[V_82]]] : (!fir.ref<!fir.array<2x?xf32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<?x?xf32>>
! CHECK:   %[[V_84:[0-9]+]] = fir.convert %[[V_83:[0-9]+]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
! CHECK:   %[[V_85:[0-9]+]] = fir.call @_FortranAioOutputDescriptor(%[[V_73]], %[[V_84]]) fastmath<contract> : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:   %[[V_86:[0-9]+]] = fir.call @_FortranAioEndIoStatement(%[[V_73]]) fastmath<contract> : (!fir.ref<i8>) -> i32
! CHECK:   return
! CHECK:   }

! CHECK-LABEL: func @_QPss4(
! CHECK-SAME:               %arg0: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK-DAG: %[[C_2:[-0-9a-z_]+]] = arith.constant 2 : index
! CHECK-DAG: %[[C_m1:[-0-9a-z_]+]] = arith.constant -1 : index
! CHECK-DAG: %[[C_1:[-0-9a-z_]+]] = arith.constant 1 : index
! CHECK-DAG: %[[C_41_i32:[-0-9a-z_]+]] = arith.constant 41 : i32
! CHECK-DAG: %[[C_6_i32:[-0-9a-z_]+]] = arith.constant 6 : i32
! CHECK-DAG: %[[C_st:[-0-9a-z_]+]] = arith.constant 7.000000e+00 : f32
! CHECK-DAG: %[[C_1_i32:[-0-9a-z_]+]] = arith.constant 1 : i32
! CHECK-DAG: %[[C_st_0:[-0-9a-z_]+]] = arith.constant -2.000000e+00 : f32
! CHECK-DAG: %[[C_0:[-0-9a-z_]+]] = arith.constant 0 : index
! CHECK:   %[[V_0:[0-9]+]] = fir.load %arg0 : !fir.ref<i32>
! CHECK:   %[[V_1:[0-9]+]] = fir.convert %[[V_0:[0-9]+]] : (i32) -> index
! CHECK:   %[[V_2:[0-9]+]] = arith.cmpi sgt, %[[V_1]], %[[C_0]] : index
! CHECK:   %[[V_3:[0-9]+]] = arith.select %[[V_2]], %[[V_1]], %[[C_0]] : index
! CHECK:   %[[V_4:[0-9]+]] = fir.alloca !fir.array<?x2xf32>, %[[V_3]] {bindc_name = "aa", uniq_name = "_QFss4Eaa"}
! CHECK:   %[[V_5:[0-9]+]] = fir.shape %[[V_3]], %[[C_2]] : (index, index) -> !fir.shape<2>
! CHECK:   cf.br ^bb1(%[[C_0]], %[[C_2]] : index, index)
! CHECK: ^bb1(%[[V_6:[0-9]+]]: index, %[[V_7:[0-9]+]]: index):  // 2 preds: ^bb0, ^bb4
! CHECK:   %[[V_8:[0-9]+]] = arith.cmpi sgt, %[[V_7]], %[[C_0]] : index
! CHECK:   cf.cond_br %[[V_8]], ^bb2(%[[C_0]], %[[V_3:[0-9]+]] : index, index), ^bb5
! CHECK: ^bb2(%[[V_9:[0-9]+]]: index, %[[V_10:[0-9]+]]: index):  // 2 preds: ^bb1, ^bb3
! CHECK:   %[[V_11:[0-9]+]] = arith.cmpi sgt, %[[V_10]], %[[C_0]] : index
! CHECK:   cf.cond_br %[[V_11]], ^bb3, ^bb4
! CHECK: ^bb3:  // pred: ^bb2
! CHECK:   %[[V_12:[0-9]+]] = arith.addi %[[V_9]], %[[C_1]] : index
! CHECK:   %[[V_13:[0-9]+]] = arith.addi %[[V_6]], %[[C_1]] : index
! CHECK:   %[[V_14:[0-9]+]] = fir.array_coor %[[V_4]](%[[V_5]]) %[[V_12]], %[[V_13:[0-9]+]] : (!fir.ref<!fir.array<?x2xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:   fir.store %[[C_st_0]] to %[[V_14:[0-9]+]] : !fir.ref<f32>
! CHECK:   %[[V_15:[0-9]+]] = arith.subi %[[V_10]], %[[C_1]] : index
! CHECK:   cf.br ^bb2(%[[V_12]], %[[V_15:[0-9]+]] : index, index)
! CHECK: ^bb4:  // pred: ^bb2
! CHECK:   %[[V_16:[0-9]+]] = arith.addi %[[V_6]], %[[C_1]] : index
! CHECK:   %[[V_17:[0-9]+]] = arith.subi %[[V_7]], %[[C_1]] : index
! CHECK:   cf.br ^bb1(%[[V_16]], %[[V_17:[0-9]+]] : index, index)
! CHECK: ^bb5:  // pred: ^bb1
! CHECK:   %[[V_18:[0-9]+]] = fir.load %arg0 : !fir.ref<i32>
! CHECK:   %[[V_19:[0-9]+]] = fir.convert %[[V_18:[0-9]+]] : (i32) -> index
! CHECK:   %[[V_20:[0-9]+]] = arith.addi %[[V_19]], %[[C_m1]] : index
! CHECK:   %[[V_21:[0-9]+]] = arith.cmpi sgt, %[[V_20]], %[[C_0]] : index
! CHECK:   %[[V_22:[0-9]+]] = arith.select %[[V_21]], %[[V_20]], %[[C_0]] : index
! CHECK:   %[[V_23:[0-9]+]] = fir.slice %[[C_2]], %[[V_19]], %[[C_1]], %[[C_1]], %[[C_2]], %[[C_1]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:   %[[V_24:[0-9]+]] = fir.allocmem !fir.array<?x2xf32>, %[[V_3]]
! CHECK:   cf.br ^bb6(%[[C_0]], %[[C_2]] : index, index)
! CHECK: ^bb6(%[[V_25:[0-9]+]]: index, %[[V_26:[0-9]+]]: index):  // 2 preds: ^bb5, ^bb9
! CHECK:   %[[V_27:[0-9]+]] = arith.cmpi sgt, %[[V_26]], %[[C_0]] : index
! CHECK:   cf.cond_br %[[V_27]], ^bb7(%[[C_0]], %[[V_3:[0-9]+]] : index, index), ^bb10
! CHECK: ^bb7(%[[V_28:[0-9]+]]: index, %[[V_29:[0-9]+]]: index):  // 2 preds: ^bb6, ^bb8
! CHECK:   %[[V_30:[0-9]+]] = arith.cmpi sgt, %[[V_29]], %[[C_0]] : index
! CHECK:   cf.cond_br %[[V_30]], ^bb8, ^bb9
! CHECK: ^bb8:  // pred: ^bb7
! CHECK:   %[[V_31:[0-9]+]] = arith.addi %[[V_28]], %[[C_1]] : index
! CHECK:   %[[V_32:[0-9]+]] = arith.addi %[[V_25]], %[[C_1]] : index
! CHECK:   %[[V_33:[0-9]+]] = fir.array_coor %[[V_4]](%[[V_5]]) %[[V_31]], %[[V_32:[0-9]+]] : (!fir.ref<!fir.array<?x2xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:   %[[V_34:[0-9]+]] = fir.array_coor %[[V_24]](%[[V_5]]) %[[V_31]], %[[V_32:[0-9]+]] : (!fir.heap<!fir.array<?x2xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:   %[[V_35:[0-9]+]] = fir.load %[[V_33:[0-9]+]] : !fir.ref<f32>
! CHECK:   fir.store %[[V_35]] to %[[V_34:[0-9]+]] : !fir.ref<f32>
! CHECK:   %[[V_36:[0-9]+]] = arith.subi %[[V_29]], %[[C_1]] : index
! CHECK:   cf.br ^bb7(%[[V_31]], %[[V_36:[0-9]+]] : index, index)
! CHECK: ^bb9:  // pred: ^bb7
! CHECK:   %[[V_37:[0-9]+]] = arith.addi %[[V_25]], %[[C_1]] : index
! CHECK:   %[[V_38:[0-9]+]] = arith.subi %[[V_26]], %[[C_1]] : index
! CHECK:   cf.br ^bb6(%[[V_37]], %[[V_38:[0-9]+]] : index, index)
! CHECK: ^bb10:  // pred: ^bb6
! CHECK:   %[[V_39:[0-9]+]] = arith.subi %[[V_18]], %[[C_1_i32]] : i32
! CHECK:   %[[V_40:[0-9]+]] = fir.convert %[[V_39:[0-9]+]] : (i32) -> index
! CHECK:   %[[V_41:[0-9]+]] = fir.slice %[[C_1]], %[[V_40]], %[[C_1]], %[[C_1]], %[[C_2]], %[[C_1]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:   cf.br ^bb11(%[[C_0]], %[[C_2]] : index, index)
! CHECK: ^bb11(%[[V_42:[0-9]+]]: index, %[[V_43:[0-9]+]]: index):  // 2 preds: ^bb10, ^bb14
! CHECK:   %[[V_44:[0-9]+]] = arith.cmpi sgt, %[[V_43]], %[[C_0]] : index
! CHECK:   cf.cond_br %[[V_44]], ^bb12(%[[C_0]], %[[V_22:[0-9]+]] : index, index), ^bb15(%[[C_0]], %[[C_2]] : index, index)
! CHECK: ^bb12(%[[V_45:[0-9]+]]: index, %[[V_46:[0-9]+]]: index):  // 2 preds: ^bb11, ^bb13
! CHECK:   %[[V_47:[0-9]+]] = arith.cmpi sgt, %[[V_46]], %[[C_0]] : index
! CHECK:   cf.cond_br %[[V_47]], ^bb13, ^bb14
! CHECK: ^bb13:  // pred: ^bb12
! CHECK:   %[[V_48:[0-9]+]] = arith.addi %[[V_45]], %[[C_1]] : index
! CHECK:   %[[V_49:[0-9]+]] = arith.addi %[[V_42]], %[[C_1]] : index
! CHECK:   %[[V_50:[0-9]+]] = fir.array_coor %[[V_4]](%[[V_5]]) [%[[V_41]]] %[[V_48]], %[[V_49:[0-9]+]] : (!fir.ref<!fir.array<?x2xf32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f32>
! CHECK:   %[[V_51:[0-9]+]] = fir.load %[[V_50:[0-9]+]] : !fir.ref<f32>
! CHECK:   %[[V_52:[0-9]+]] = arith.addf %[[V_51]], %[[C_st]] fastmath<contract> : f32
! CHECK:   %[[V_53:[0-9]+]] = fir.array_coor %[[V_24]](%[[V_5]]) [%[[V_23]]] %[[V_48]], %[[V_49:[0-9]+]] : (!fir.heap<!fir.array<?x2xf32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f32>
! CHECK:   fir.store %[[V_52]] to %[[V_53:[0-9]+]] : !fir.ref<f32>
! CHECK:   %[[V_54:[0-9]+]] = arith.subi %[[V_46]], %[[C_1]] : index
! CHECK:   cf.br ^bb12(%[[V_48]], %[[V_54:[0-9]+]] : index, index)
! CHECK: ^bb14:  // pred: ^bb12
! CHECK:   %[[V_55:[0-9]+]] = arith.addi %[[V_42]], %[[C_1]] : index
! CHECK:   %[[V_56:[0-9]+]] = arith.subi %[[V_43]], %[[C_1]] : index
! CHECK:   cf.br ^bb11(%[[V_55]], %[[V_56:[0-9]+]] : index, index)
! CHECK: ^bb15(%[[V_57:[0-9]+]]: index, %[[V_58:[0-9]+]]: index):  // 2 preds: ^bb11, ^bb18
! CHECK:   %[[V_59:[0-9]+]] = arith.cmpi sgt, %[[V_58]], %[[C_0]] : index
! CHECK:   cf.cond_br %[[V_59]], ^bb16(%[[C_0]], %[[V_3:[0-9]+]] : index, index), ^bb19
! CHECK: ^bb16(%[[V_60:[0-9]+]]: index, %[[V_61:[0-9]+]]: index):  // 2 preds: ^bb15, ^bb17
! CHECK:   %[[V_62:[0-9]+]] = arith.cmpi sgt, %[[V_61]], %[[C_0]] : index
! CHECK:   cf.cond_br %[[V_62]], ^bb17, ^bb18
! CHECK: ^bb17:  // pred: ^bb16
! CHECK:   %[[V_63:[0-9]+]] = arith.addi %[[V_60]], %[[C_1]] : index
! CHECK:   %[[V_64:[0-9]+]] = arith.addi %[[V_57]], %[[C_1]] : index
! CHECK:   %[[V_65:[0-9]+]] = fir.array_coor %[[V_24]](%[[V_5]]) %[[V_63]], %[[V_64:[0-9]+]] : (!fir.heap<!fir.array<?x2xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:   %[[V_66:[0-9]+]] = fir.array_coor %[[V_4]](%[[V_5]]) %[[V_63]], %[[V_64:[0-9]+]] : (!fir.ref<!fir.array<?x2xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:   %[[V_67:[0-9]+]] = fir.load %[[V_65:[0-9]+]] : !fir.ref<f32>
! CHECK:   fir.store %[[V_67]] to %[[V_66:[0-9]+]] : !fir.ref<f32>
! CHECK:   %[[V_68:[0-9]+]] = arith.subi %[[V_61]], %[[C_1]] : index
! CHECK:   cf.br ^bb16(%[[V_63]], %[[V_68:[0-9]+]] : index, index)
! CHECK: ^bb18:  // pred: ^bb16
! CHECK:   %[[V_69:[0-9]+]] = arith.addi %[[V_57]], %[[C_1]] : index
! CHECK:   %[[V_70:[0-9]+]] = arith.subi %[[V_58]], %[[C_1]] : index
! CHECK:   cf.br ^bb15(%[[V_69]], %[[V_70:[0-9]+]] : index, index)
! CHECK: ^bb19:  // pred: ^bb15
! CHECK:   fir.freemem %[[V_24:[0-9]+]] : !fir.heap<!fir.array<?x2xf32>>
! CHECK:   %[[V_73:[0-9]+]] = fir.call @_FortranAioBeginExternalListOutput(%[[C_6_i32]], %{{.*}}, %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:   %[[V_74:[0-9]+]] = fir.slice %[[C_1]], %[[C_2]], %[[C_1]], %[[C_1]], %[[C_2]], %[[C_1]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:   %[[V_75:[0-9]+]] = fir.embox %[[V_4]](%[[V_5]]) [%[[V_74]]] : (!fir.ref<!fir.array<?x2xf32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<2x?xf32>>
! CHECK:   %[[V_76:[0-9]+]] = fir.convert %[[V_75:[0-9]+]] : (!fir.box<!fir.array<2x?xf32>>) -> !fir.box<none>
! CHECK:   %[[V_77:[0-9]+]] = fir.call @_FortranAioOutputDescriptor(%[[V_73]], %[[V_76]]) fastmath<contract> : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:   %[[V_78:[0-9]+]] = fir.load %arg0 : !fir.ref<i32>
! CHECK:   %[[V_79:[0-9]+]] = arith.subi %[[V_78]], %[[C_1_i32]] : i32
! CHECK:   %[[V_80:[0-9]+]] = fir.convert %[[V_79:[0-9]+]] : (i32) -> index
! CHECK:   %[[V_81:[0-9]+]] = fir.convert %[[V_78:[0-9]+]] : (i32) -> index
! CHECK:   %[[V_82:[0-9]+]] = fir.slice %[[V_80]], %[[V_81]], %[[C_1]], %[[C_1]], %[[C_2]], %[[C_1]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:   %[[V_83:[0-9]+]] = fir.embox %[[V_4]](%[[V_5]]) [%[[V_82]]] : (!fir.ref<!fir.array<?x2xf32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<?x?xf32>>
! CHECK:   %[[V_84:[0-9]+]] = fir.convert %[[V_83:[0-9]+]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
! CHECK:   %[[V_85:[0-9]+]] = fir.call @_FortranAioOutputDescriptor(%[[V_73]], %[[V_84]]) fastmath<contract> : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:   %[[V_86:[0-9]+]] = fir.call @_FortranAioEndIoStatement(%[[V_73]]) fastmath<contract> : (!fir.ref<i8>) -> i32
! CHECK:   return
! CHECK:   }

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
