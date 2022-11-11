! RUN: bbc %s -o - | FileCheck %s

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
! CHECK-SAME:               %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK-DAG:     %[[VAL_1:.*]] = arith.constant -1 : index
! CHECK-DAG:     %[[VAL_2:.*]] = arith.constant -2 : i32
! CHECK-DAG:     %[[VAL_3:.*]] = arith.constant 1 : index
! CHECK-DAG:     %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK-DAG:     %[[VAL_5:.*]] = arith.constant 2 : index
! CHECK-DAG:     %[[VAL_6:.*]] = arith.constant 1 : i32
! CHECK-DAG:     %[[VAL_7:.*]] = arith.constant 7.000000e+00 : f32
! CHECK-DAG:     %[[VAL_8:.*]] = arith.constant -1 : i32
! CHECK:         %[[VAL_10:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:         %[[VAL_11A:.*]] = fir.convert %[[VAL_10]] : (i32) -> index
! CHECK:         %[[CMP:.*]] = arith.cmpi sgt, %[[VAL_11A]], %[[VAL_4]] : index 
! CHECK:         %[[VAL_11:.*]] = arith.select %[[CMP]], %[[VAL_11A]], %[[VAL_4]] : index 
! CHECK:         %[[VAL_12:.*]] = fir.alloca !fir.array<?xf32>, %[[VAL_11]] {bindc_name = "aa", uniq_name = "_QFss2Eaa"}
! CHECK:         %[[VAL_13:.*]] = fir.shape %[[VAL_11]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_2]] : (i32) -> f32
! CHECK:         br ^bb1(%[[VAL_4]], %[[VAL_11]] : index, index)
! CHECK:       ^bb1(%[[VAL_15:.*]]: index, %[[VAL_16:.*]]: index):
! CHECK:         %[[VAL_17:.*]] = arith.cmpi sgt, %[[VAL_16]], %[[VAL_4]] : index
! CHECK:         cond_br %[[VAL_17]], ^bb2, ^bb3
! CHECK:       ^bb2:
! CHECK:         %[[VAL_18:.*]] = arith.addi %[[VAL_15]], %[[VAL_3]] : index
! CHECK:         %[[VAL_19:.*]] = fir.array_coor %[[VAL_12]](%[[VAL_13]]) %[[VAL_18]] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
! CHECK:         fir.store %[[VAL_14]] to %[[VAL_19]] : !fir.ref<f32>
! CHECK:         %[[VAL_20:.*]] = arith.subi %[[VAL_16]], %[[VAL_3]] : index
! CHECK:         br ^bb1(%[[VAL_18]], %[[VAL_20]] : index, index)
! CHECK:       ^bb3:
! CHECK:         %[[VAL_21:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:         %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (i32) -> index
! CHECK:         %[[VAL_23:.*]] = arith.addi %[[VAL_22]], %[[VAL_1]] : index
! CHECK:         %[[VAL_24:.*]] = arith.cmpi sgt, %[[VAL_23]], %[[VAL_4]] : index
! CHECK:         %[[VAL_25:.*]] = arith.select %[[VAL_24]], %[[VAL_23]], %[[VAL_4]] : index
! CHECK:         %[[VAL_26:.*]] = fir.slice %[[VAL_5]], %[[VAL_22]], %[[VAL_3]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_27:.*]] = fir.allocmem !fir.array<?xf32>, %[[VAL_11]]
! CHECK:         br ^bb4(%[[VAL_4]], %[[VAL_11]] : index, index)
! CHECK:       ^bb4(%[[VAL_28:.*]]: index, %[[VAL_29:.*]]: index):
! CHECK:         %[[VAL_30:.*]] = arith.cmpi sgt, %[[VAL_29]], %[[VAL_4]] : index
! CHECK:         cond_br %[[VAL_30]], ^bb5, ^bb6
! CHECK:       ^bb5:
! CHECK:         %[[VAL_31:.*]] = arith.addi %[[VAL_28]], %[[VAL_3]] : index
! CHECK:         %[[VAL_32:.*]] = fir.array_coor %[[VAL_12]](%[[VAL_13]]) %[[VAL_31]] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
! CHECK:         %[[VAL_33:.*]] = fir.array_coor %[[VAL_27]](%[[VAL_13]]) %[[VAL_31]] : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
! CHECK:         %[[VAL_34:.*]] = fir.load %[[VAL_32]] : !fir.ref<f32>
! CHECK:         fir.store %[[VAL_34]] to %[[VAL_33]] : !fir.ref<f32>
! CHECK:         %[[VAL_35:.*]] = arith.subi %[[VAL_29]], %[[VAL_3]] : index
! CHECK:         br ^bb4(%[[VAL_31]], %[[VAL_35]] : index, index)
! CHECK:       ^bb6:
! CHECK:         %[[VAL_36:.*]] = arith.subi %[[VAL_21]], %[[VAL_6]] : i32
! CHECK:         %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : (i32) -> index
! CHECK:         %[[VAL_38:.*]] = fir.slice %[[VAL_3]], %[[VAL_37]], %[[VAL_3]] : (index, index, index) -> !fir.slice<1>
! CHECK:         br ^bb7(%[[VAL_4]], %[[VAL_25]] : index, index)
! CHECK:       ^bb7(%[[VAL_39:.*]]: index, %[[VAL_40:.*]]: index):
! CHECK:         %[[VAL_41:.*]] = arith.cmpi sgt, %[[VAL_40]], %[[VAL_4]] : index
! CHECK:         cond_br %[[VAL_41]], ^bb8, ^bb9(%[[VAL_4]], %[[VAL_11]] : index, index)
! CHECK:       ^bb8:
! CHECK:         %[[VAL_42:.*]] = arith.addi %[[VAL_39]], %[[VAL_3]] : index
! CHECK:         %[[VAL_43:.*]] = fir.array_coor %[[VAL_12]](%[[VAL_13]]) {{\[}}%[[VAL_38]]] %[[VAL_42]] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<f32>
! CHECK:         %[[VAL_44:.*]] = fir.load %[[VAL_43]] : !fir.ref<f32>
! CHECK:         %[[VAL_45:.*]] = arith.addf %[[VAL_44]], %[[VAL_7]] : f32
! CHECK:         %[[VAL_46:.*]] = fir.array_coor %[[VAL_27]](%[[VAL_13]]) {{\[}}%[[VAL_26]]] %[[VAL_42]] : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<f32>
! CHECK:         fir.store %[[VAL_45]] to %[[VAL_46]] : !fir.ref<f32>
! CHECK:         %[[VAL_47:.*]] = arith.subi %[[VAL_40]], %[[VAL_3]] : index
! CHECK:         br ^bb7(%[[VAL_42]], %[[VAL_47]] : index, index)
! CHECK:       ^bb9(%[[VAL_48:.*]]: index, %[[VAL_49:.*]]: index):
! CHECK:         %[[VAL_50:.*]] = arith.cmpi sgt, %[[VAL_49]], %[[VAL_4]] : index
! CHECK:         cond_br %[[VAL_50]], ^bb10, ^bb11
! CHECK:       ^bb10:
! CHECK:         %[[VAL_51:.*]] = arith.addi %[[VAL_48]], %[[VAL_3]] : index
! CHECK:         %[[VAL_52:.*]] = fir.array_coor %[[VAL_27]](%[[VAL_13]]) %[[VAL_51]] : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
! CHECK:         %[[VAL_53:.*]] = fir.array_coor %[[VAL_12]](%[[VAL_13]]) %[[VAL_51]] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
! CHECK:         %[[VAL_54:.*]] = fir.load %[[VAL_52]] : !fir.ref<f32>
! CHECK:         fir.store %[[VAL_54]] to %[[VAL_53]] : !fir.ref<f32>
! CHECK:         %[[VAL_55:.*]] = arith.subi %[[VAL_49]], %[[VAL_3]] : index
! CHECK:         br ^bb9(%[[VAL_51]], %[[VAL_55]] : index, index)
! CHECK:       ^bb11:
! CHECK:         fir.freemem %[[VAL_27]] : !fir.heap<!fir.array<?xf32>>
! CHECK:         %[[VAL_58:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_8]], %{{.*}}, %{{.*}}) : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:         %[[VAL_59:.*]] = fir.slice %[[VAL_3]], %[[VAL_5]], %[[VAL_3]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_60:.*]] = fir.embox %[[VAL_12]](%[[VAL_13]]) {{\[}}%[[VAL_59]]] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<2xf32>>
! CHECK:         %[[VAL_61:.*]] = fir.convert %[[VAL_60]] : (!fir.box<!fir.array<2xf32>>) -> !fir.box<none>
! CHECK:         %[[VAL_62:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_58]], %[[VAL_61]]) : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:         %[[VAL_63:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:         %[[VAL_64:.*]] = arith.subi %[[VAL_63]], %[[VAL_6]] : i32
! CHECK:         %[[VAL_65:.*]] = fir.convert %[[VAL_64]] : (i32) -> index
! CHECK:         %[[VAL_66:.*]] = fir.convert %[[VAL_63]] : (i32) -> index
! CHECK:         %[[VAL_67:.*]] = fir.slice %[[VAL_65]], %[[VAL_66]], %[[VAL_3]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_68:.*]] = fir.embox %[[VAL_12]](%[[VAL_13]]) {{\[}}%[[VAL_67]]] : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK:         %[[VAL_69:.*]] = fir.convert %[[VAL_68]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
! CHECK:         %[[VAL_70:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_58]], %[[VAL_69]]) : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:         %[[VAL_71:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_58]]) : (!fir.ref<i8>) -> i32
! CHECK:         return
! CHECK:       }

! CHECK-LABEL: func @_QPss3(
! CHECK-SAME:               %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK-DAG:     %[[VAL_1:.*]] = arith.constant -1 : index
! CHECK-DAG:     %[[VAL_2:.*]] = arith.constant 2 : index
! CHECK-DAG:     %[[VAL_3:.*]] = arith.constant 1 : index
! CHECK-DAG:     %[[VAL_4:.*]] = arith.constant -2 : i32
! CHECK-DAG:     %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK-DAG:     %[[VAL_6:.*]] = arith.constant 1 : i32
! CHECK-DAG:     %[[VAL_7:.*]] = arith.constant 7.000000e+00 : f32
! CHECK-DAG:     %[[VAL_8:.*]] = arith.constant -1 : i32
! CHECK:         %[[VAL_10:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i32) -> index
! CHECK:         %[[CMP:.*]] = arith.cmpi sgt, %[[VAL_11A]], %[[VAL_5]] : index 
! CHECK:         %[[VAL_11:.*]] = arith.select %[[CMP]], %[[VAL_11A]], %[[VAL_5]] : index 
! CHECK:         %[[VAL_12:.*]] = fir.alloca !fir.array<2x?xf32>, %[[VAL_11]] {bindc_name = "aa", uniq_name = "_QFss3Eaa"}
! CHECK:         %[[VAL_13:.*]] = fir.shape %[[VAL_2]], %[[VAL_11]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_4]] : (i32) -> f32
! CHECK:         br ^bb1(%[[VAL_5]], %[[VAL_11]] : index, index)
! CHECK:       ^bb1(%[[VAL_15:.*]]: index, %[[VAL_16:.*]]: index):
! CHECK:         %[[VAL_17:.*]] = arith.cmpi sgt, %[[VAL_16]], %[[VAL_5]] : index
! CHECK:         cond_br %[[VAL_17]], ^bb2(%[[VAL_5]], %[[VAL_2]] : index, index), ^bb5
! CHECK:       ^bb2(%[[VAL_18:.*]]: index, %[[VAL_19:.*]]: index):
! CHECK:         %[[VAL_20:.*]] = arith.cmpi sgt, %[[VAL_19]], %[[VAL_5]] : index
! CHECK:         cond_br %[[VAL_20]], ^bb3, ^bb4
! CHECK:       ^bb3:
! CHECK:         %[[VAL_21:.*]] = arith.addi %[[VAL_18]], %[[VAL_3]] : index
! CHECK:         %[[VAL_22:.*]] = arith.addi %[[VAL_15]], %[[VAL_3]] : index
! CHECK:         %[[VAL_23:.*]] = fir.array_coor %[[VAL_12]](%[[VAL_13]]) %[[VAL_21]], %[[VAL_22]] : (!fir.ref<!fir.array<2x?xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:         fir.store %[[VAL_14]] to %[[VAL_23]] : !fir.ref<f32>
! CHECK:         %[[VAL_24:.*]] = arith.subi %[[VAL_19]], %[[VAL_3]] : index
! CHECK:         br ^bb2(%[[VAL_21]], %[[VAL_24]] : index, index)
! CHECK:       ^bb4:
! CHECK:         %[[VAL_25:.*]] = arith.addi %[[VAL_15]], %[[VAL_3]] : index
! CHECK:         %[[VAL_26:.*]] = arith.subi %[[VAL_16]], %[[VAL_3]] : index
! CHECK:         br ^bb1(%[[VAL_25]], %[[VAL_26]] : index, index)
! CHECK:       ^bb5:
! CHECK:         %[[VAL_27:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:         %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i32) -> index
! CHECK:         %[[VAL_29:.*]] = arith.addi %[[VAL_28]], %[[VAL_1]] : index
! CHECK:         %[[VAL_30:.*]] = arith.cmpi sgt, %[[VAL_29]], %[[VAL_5]] : index
! CHECK:         %[[VAL_31:.*]] = arith.select %[[VAL_30]], %[[VAL_29]], %[[VAL_5]] : index
! CHECK:         %[[VAL_32:.*]] = fir.slice %[[VAL_3]], %[[VAL_2]], %[[VAL_3]], %[[VAL_2]], %[[VAL_28]], %[[VAL_3]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:         %[[VAL_33:.*]] = fir.allocmem !fir.array<2x?xf32>, %[[VAL_11]]
! CHECK:         br ^bb6(%[[VAL_5]], %[[VAL_11]] : index, index)
! CHECK:       ^bb6(%[[VAL_34:.*]]: index, %[[VAL_35:.*]]: index):
! CHECK:         %[[VAL_36:.*]] = arith.cmpi sgt, %[[VAL_35]], %[[VAL_5]] : index
! CHECK:         cond_br %[[VAL_36]], ^bb7(%[[VAL_5]], %[[VAL_2]] : index, index), ^bb10
! CHECK:       ^bb7(%[[VAL_37:.*]]: index, %[[VAL_38:.*]]: index):
! CHECK:         %[[VAL_39:.*]] = arith.cmpi sgt, %[[VAL_38]], %[[VAL_5]] : index
! CHECK:         cond_br %[[VAL_39]], ^bb8, ^bb9
! CHECK:       ^bb8:
! CHECK:         %[[VAL_40:.*]] = arith.addi %[[VAL_37]], %[[VAL_3]] : index
! CHECK:         %[[VAL_41:.*]] = arith.addi %[[VAL_34]], %[[VAL_3]] : index
! CHECK:         %[[VAL_42:.*]] = fir.array_coor %[[VAL_12]](%[[VAL_13]]) %[[VAL_40]], %[[VAL_41]] : (!fir.ref<!fir.array<2x?xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:         %[[VAL_43:.*]] = fir.array_coor %[[VAL_33]](%[[VAL_13]]) %[[VAL_40]], %[[VAL_41]] : (!fir.heap<!fir.array<2x?xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:         %[[VAL_44:.*]] = fir.load %[[VAL_42]] : !fir.ref<f32>
! CHECK:         fir.store %[[VAL_44]] to %[[VAL_43]] : !fir.ref<f32>
! CHECK:         %[[VAL_45:.*]] = arith.subi %[[VAL_38]], %[[VAL_3]] : index
! CHECK:         br ^bb7(%[[VAL_40]], %[[VAL_45]] : index, index)
! CHECK:       ^bb9:
! CHECK:         %[[VAL_46:.*]] = arith.addi %[[VAL_34]], %[[VAL_3]] : index
! CHECK:         %[[VAL_47:.*]] = arith.subi %[[VAL_35]], %[[VAL_3]] : index
! CHECK:         br ^bb6(%[[VAL_46]], %[[VAL_47]] : index, index)
! CHECK:       ^bb10:
! CHECK:         %[[VAL_48:.*]] = arith.subi %[[VAL_27]], %[[VAL_6]] : i32
! CHECK:         %[[VAL_49:.*]] = fir.convert %[[VAL_48]] : (i32) -> index
! CHECK:         %[[VAL_50:.*]] = fir.slice %[[VAL_3]], %[[VAL_2]], %[[VAL_3]], %[[VAL_3]], %[[VAL_49]], %[[VAL_3]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:         br ^bb11(%[[VAL_5]], %[[VAL_31]] : index, index)
! CHECK:       ^bb11(%[[VAL_51:.*]]: index, %[[VAL_52:.*]]: index):
! CHECK:         %[[VAL_53:.*]] = arith.cmpi sgt, %[[VAL_52]], %[[VAL_5]] : index
! CHECK:         cond_br %[[VAL_53]], ^bb12(%[[VAL_5]], %[[VAL_2]] : index, index), ^bb15(%[[VAL_5]], %[[VAL_11]] : index, index)
! CHECK:       ^bb12(%[[VAL_54:.*]]: index, %[[VAL_55:.*]]: index):
! CHECK:         %[[VAL_56:.*]] = arith.cmpi sgt, %[[VAL_55]], %[[VAL_5]] : index
! CHECK:         cond_br %[[VAL_56]], ^bb13, ^bb14
! CHECK:       ^bb13:
! CHECK:         %[[VAL_57:.*]] = arith.addi %[[VAL_54]], %[[VAL_3]] : index
! CHECK:         %[[VAL_58:.*]] = arith.addi %[[VAL_51]], %[[VAL_3]] : index
! CHECK:         %[[VAL_59:.*]] = fir.array_coor %[[VAL_12]](%[[VAL_13]]) {{\[}}%[[VAL_50]]] %[[VAL_57]], %[[VAL_58]] : (!fir.ref<!fir.array<2x?xf32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f32>
! CHECK:         %[[VAL_60:.*]] = fir.load %[[VAL_59]] : !fir.ref<f32>
! CHECK:         %[[VAL_61:.*]] = arith.addf %[[VAL_60]], %[[VAL_7]] : f32
! CHECK:         %[[VAL_62:.*]] = fir.array_coor %[[VAL_33]](%[[VAL_13]]) {{\[}}%[[VAL_32]]] %[[VAL_57]], %[[VAL_58]] : (!fir.heap<!fir.array<2x?xf32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f32>
! CHECK:         fir.store %[[VAL_61]] to %[[VAL_62]] : !fir.ref<f32>
! CHECK:         %[[VAL_63:.*]] = arith.subi %[[VAL_55]], %[[VAL_3]] : index
! CHECK:         br ^bb12(%[[VAL_57]], %[[VAL_63]] : index, index)
! CHECK:       ^bb14:
! CHECK:         %[[VAL_64:.*]] = arith.addi %[[VAL_51]], %[[VAL_3]] : index
! CHECK:         %[[VAL_65:.*]] = arith.subi %[[VAL_52]], %[[VAL_3]] : index
! CHECK:         br ^bb11(%[[VAL_64]], %[[VAL_65]] : index, index)
! CHECK:       ^bb15(%[[VAL_66:.*]]: index, %[[VAL_67:.*]]: index):
! CHECK:         %[[VAL_68:.*]] = arith.cmpi sgt, %[[VAL_67]], %[[VAL_5]] : index
! CHECK:         cond_br %[[VAL_68]], ^bb16(%[[VAL_5]], %[[VAL_2]] : index, index), ^bb19
! CHECK:       ^bb16(%[[VAL_69:.*]]: index, %[[VAL_70:.*]]: index):
! CHECK:         %[[VAL_71:.*]] = arith.cmpi sgt, %[[VAL_70]], %[[VAL_5]] : index
! CHECK:         cond_br %[[VAL_71]], ^bb17, ^bb18
! CHECK:       ^bb17:
! CHECK:         %[[VAL_72:.*]] = arith.addi %[[VAL_69]], %[[VAL_3]] : index
! CHECK:         %[[VAL_73:.*]] = arith.addi %[[VAL_66]], %[[VAL_3]] : index
! CHECK:         %[[VAL_74:.*]] = fir.array_coor %[[VAL_33]](%[[VAL_13]]) %[[VAL_72]], %[[VAL_73]] : (!fir.heap<!fir.array<2x?xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:         %[[VAL_75:.*]] = fir.array_coor %[[VAL_12]](%[[VAL_13]]) %[[VAL_72]], %[[VAL_73]] : (!fir.ref<!fir.array<2x?xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:         %[[VAL_76:.*]] = fir.load %[[VAL_74]] : !fir.ref<f32>
! CHECK:         fir.store %[[VAL_76]] to %[[VAL_75]] : !fir.ref<f32>
! CHECK:         %[[VAL_77:.*]] = arith.subi %[[VAL_70]], %[[VAL_3]] : index
! CHECK:         br ^bb16(%[[VAL_72]], %[[VAL_77]] : index, index)
! CHECK:       ^bb18:
! CHECK:         %[[VAL_78:.*]] = arith.addi %[[VAL_66]], %[[VAL_3]] : index
! CHECK:         %[[VAL_79:.*]] = arith.subi %[[VAL_67]], %[[VAL_3]] : index
! CHECK:         br ^bb15(%[[VAL_78]], %[[VAL_79]] : index, index)
! CHECK:       ^bb19:
! CHECK:         fir.freemem %[[VAL_33]] : !fir.heap<!fir.array<2x?xf32>>
! CHECK:         %[[VAL_82:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_8]], %{{.*}}, %{{.*}}) : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:         %[[VAL_83:.*]] = fir.slice %[[VAL_3]], %[[VAL_2]], %[[VAL_3]], %[[VAL_3]], %[[VAL_2]], %[[VAL_3]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:         %[[VAL_84:.*]] = fir.embox %[[VAL_12]](%[[VAL_13]]) {{\[}}%[[VAL_83]]] : (!fir.ref<!fir.array<2x?xf32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<?x2xf32>>
! CHECK:         %[[VAL_85:.*]] = fir.convert %[[VAL_84]] : (!fir.box<!fir.array<?x2xf32>>) -> !fir.box<none>
! CHECK:         %[[VAL_86:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_82]], %[[VAL_85]]) : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:         %[[VAL_87:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:         %[[VAL_88:.*]] = arith.subi %[[VAL_87]], %[[VAL_6]] : i32
! CHECK:         %[[VAL_89:.*]] = fir.convert %[[VAL_88]] : (i32) -> index
! CHECK:         %[[VAL_90:.*]] = fir.convert %[[VAL_87]] : (i32) -> index
! CHECK:         %[[VAL_91:.*]] = fir.slice %[[VAL_3]], %[[VAL_2]], %[[VAL_3]], %[[VAL_89]], %[[VAL_90]], %[[VAL_3]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:         %[[VAL_92:.*]] = fir.embox %[[VAL_12]](%[[VAL_13]]) {{\[}}%[[VAL_91]]] : (!fir.ref<!fir.array<2x?xf32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<?x?xf32>>
! CHECK:         %[[VAL_93:.*]] = fir.convert %[[VAL_92]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
! CHECK:         %[[VAL_94:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_82]], %[[VAL_93]]) : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:         %[[VAL_95:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_82]]) : (!fir.ref<i8>) -> i32
! CHECK:         return
! CHECK:       }

! CHECK-LABEL: func @_QPss4(
! CHECK-SAME:               %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK-DAG:     %[[VAL_1:.*]] = arith.constant -1 : index
! CHECK-DAG:     %[[VAL_2:.*]] = arith.constant 2 : index
! CHECK-DAG:     %[[VAL_3:.*]] = arith.constant 1 : index
! CHECK-DAG:     %[[VAL_4:.*]] = arith.constant -2 : i32
! CHECK-DAG:     %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK-DAG:     %[[VAL_6:.*]] = arith.constant 1 : i32
! CHECK-DAG:     %[[VAL_7:.*]] = arith.constant 7.000000e+00 : f32
! CHECK-DAG:     %[[VAL_8:.*]] = arith.constant -1 : i32
! CHECK:         %[[VAL_10:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:         %[[VAL_11A:.*]] = fir.convert %[[VAL_10]] : (i32) -> index
! CHECK:         %[[CMP:.*]] = arith.cmpi sgt, %[[VAL_11A]], %[[VAL_5]] : index 
! CHECK:         %[[VAL_11:.*]] = arith.select %[[CMP]], %[[VAL_11A]], %[[VAL_5]] : index 
! CHECK:         %[[VAL_12:.*]] = fir.alloca !fir.array<?x2xf32>, %[[VAL_11]] {bindc_name = "aa", uniq_name = "_QFss4Eaa"}
! CHECK:         %[[VAL_13:.*]] = fir.shape %[[VAL_11]], %[[VAL_2]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_4]] : (i32) -> f32
! CHECK:         br ^bb1(%[[VAL_5]], %[[VAL_2]] : index, index)
! CHECK:       ^bb1(%[[VAL_15:.*]]: index, %[[VAL_16:.*]]: index):
! CHECK:         %[[VAL_17:.*]] = arith.cmpi sgt, %[[VAL_16]], %[[VAL_5]] : index
! CHECK:         cond_br %[[VAL_17]], ^bb2(%[[VAL_5]], %[[VAL_11]] : index, index), ^bb5
! CHECK:       ^bb2(%[[VAL_18:.*]]: index, %[[VAL_19:.*]]: index):
! CHECK:         %[[VAL_20:.*]] = arith.cmpi sgt, %[[VAL_19]], %[[VAL_5]] : index
! CHECK:         cond_br %[[VAL_20]], ^bb3, ^bb4
! CHECK:       ^bb3:
! CHECK:         %[[VAL_21:.*]] = arith.addi %[[VAL_18]], %[[VAL_3]] : index
! CHECK:         %[[VAL_22:.*]] = arith.addi %[[VAL_15]], %[[VAL_3]] : index
! CHECK:         %[[VAL_23:.*]] = fir.array_coor %[[VAL_12]](%[[VAL_13]]) %[[VAL_21]], %[[VAL_22]] : (!fir.ref<!fir.array<?x2xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:         fir.store %[[VAL_14]] to %[[VAL_23]] : !fir.ref<f32>
! CHECK:         %[[VAL_24:.*]] = arith.subi %[[VAL_19]], %[[VAL_3]] : index
! CHECK:         br ^bb2(%[[VAL_21]], %[[VAL_24]] : index, index)
! CHECK:       ^bb4:
! CHECK:         %[[VAL_25:.*]] = arith.addi %[[VAL_15]], %[[VAL_3]] : index
! CHECK:         %[[VAL_26:.*]] = arith.subi %[[VAL_16]], %[[VAL_3]] : index
! CHECK:         br ^bb1(%[[VAL_25]], %[[VAL_26]] : index, index)
! CHECK:       ^bb5:
! CHECK:         %[[VAL_27:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:         %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i32) -> index
! CHECK:         %[[VAL_29:.*]] = arith.addi %[[VAL_28]], %[[VAL_1]] : index
! CHECK:         %[[VAL_30:.*]] = arith.cmpi sgt, %[[VAL_29]], %[[VAL_5]] : index
! CHECK:         %[[VAL_31:.*]] = arith.select %[[VAL_30]], %[[VAL_29]], %[[VAL_5]] : index
! CHECK:         %[[VAL_32:.*]] = fir.slice %[[VAL_2]], %[[VAL_28]], %[[VAL_3]], %[[VAL_3]], %[[VAL_2]], %[[VAL_3]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:         %[[VAL_33:.*]] = fir.allocmem !fir.array<?x2xf32>, %[[VAL_11]]
! CHECK:         br ^bb6(%[[VAL_5]], %[[VAL_2]] : index, index)
! CHECK:       ^bb6(%[[VAL_34:.*]]: index, %[[VAL_35:.*]]: index):
! CHECK:         %[[VAL_36:.*]] = arith.cmpi sgt, %[[VAL_35]], %[[VAL_5]] : index
! CHECK:         cond_br %[[VAL_36]], ^bb7(%[[VAL_5]], %[[VAL_11]] : index, index), ^bb10
! CHECK:       ^bb7(%[[VAL_37:.*]]: index, %[[VAL_38:.*]]: index):
! CHECK:         %[[VAL_39:.*]] = arith.cmpi sgt, %[[VAL_38]], %[[VAL_5]] : index
! CHECK:         cond_br %[[VAL_39]], ^bb8, ^bb9
! CHECK:       ^bb8:
! CHECK:         %[[VAL_40:.*]] = arith.addi %[[VAL_37]], %[[VAL_3]] : index
! CHECK:         %[[VAL_41:.*]] = arith.addi %[[VAL_34]], %[[VAL_3]] : index
! CHECK:         %[[VAL_42:.*]] = fir.array_coor %[[VAL_12]](%[[VAL_13]]) %[[VAL_40]], %[[VAL_41]] : (!fir.ref<!fir.array<?x2xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:         %[[VAL_43:.*]] = fir.array_coor %[[VAL_33]](%[[VAL_13]]) %[[VAL_40]], %[[VAL_41]] : (!fir.heap<!fir.array<?x2xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:         %[[VAL_44:.*]] = fir.load %[[VAL_42]] : !fir.ref<f32>
! CHECK:         fir.store %[[VAL_44]] to %[[VAL_43]] : !fir.ref<f32>
! CHECK:         %[[VAL_45:.*]] = arith.subi %[[VAL_38]], %[[VAL_3]] : index
! CHECK:         br ^bb7(%[[VAL_40]], %[[VAL_45]] : index, index)
! CHECK:       ^bb9:
! CHECK:         %[[VAL_46:.*]] = arith.addi %[[VAL_34]], %[[VAL_3]] : index
! CHECK:         %[[VAL_47:.*]] = arith.subi %[[VAL_35]], %[[VAL_3]] : index
! CHECK:         br ^bb6(%[[VAL_46]], %[[VAL_47]] : index, index)
! CHECK:       ^bb10:
! CHECK:         %[[VAL_48:.*]] = arith.subi %[[VAL_27]], %[[VAL_6]] : i32
! CHECK:         %[[VAL_49:.*]] = fir.convert %[[VAL_48]] : (i32) -> index
! CHECK:         %[[VAL_50:.*]] = fir.slice %[[VAL_3]], %[[VAL_49]], %[[VAL_3]], %[[VAL_3]], %[[VAL_2]], %[[VAL_3]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:         br ^bb11(%[[VAL_5]], %[[VAL_2]] : index, index)
! CHECK:       ^bb11(%[[VAL_51:.*]]: index, %[[VAL_52:.*]]: index):
! CHECK:         %[[VAL_53:.*]] = arith.cmpi sgt, %[[VAL_52]], %[[VAL_5]] : index
! CHECK:         cond_br %[[VAL_53]], ^bb12(%[[VAL_5]], %[[VAL_31]] : index, index), ^bb15(%[[VAL_5]], %[[VAL_2]] : index, index)
! CHECK:       ^bb12(%[[VAL_54:.*]]: index, %[[VAL_55:.*]]: index):
! CHECK:         %[[VAL_56:.*]] = arith.cmpi sgt, %[[VAL_55]], %[[VAL_5]] : index
! CHECK:         cond_br %[[VAL_56]], ^bb13, ^bb14
! CHECK:       ^bb13:
! CHECK:         %[[VAL_57:.*]] = arith.addi %[[VAL_54]], %[[VAL_3]] : index
! CHECK:         %[[VAL_58:.*]] = arith.addi %[[VAL_51]], %[[VAL_3]] : index
! CHECK:         %[[VAL_59:.*]] = fir.array_coor %[[VAL_12]](%[[VAL_13]]) {{\[}}%[[VAL_50]]] %[[VAL_57]], %[[VAL_58]] : (!fir.ref<!fir.array<?x2xf32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f32>
! CHECK:         %[[VAL_60:.*]] = fir.load %[[VAL_59]] : !fir.ref<f32>
! CHECK:         %[[VAL_61:.*]] = arith.addf %[[VAL_60]], %[[VAL_7]] : f32
! CHECK:         %[[VAL_62:.*]] = fir.array_coor %[[VAL_33]](%[[VAL_13]]) {{\[}}%[[VAL_32]]] %[[VAL_57]], %[[VAL_58]] : (!fir.heap<!fir.array<?x2xf32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f32>
! CHECK:         fir.store %[[VAL_61]] to %[[VAL_62]] : !fir.ref<f32>
! CHECK:         %[[VAL_63:.*]] = arith.subi %[[VAL_55]], %[[VAL_3]] : index
! CHECK:         br ^bb12(%[[VAL_57]], %[[VAL_63]] : index, index)
! CHECK:       ^bb14:
! CHECK:         %[[VAL_64:.*]] = arith.addi %[[VAL_51]], %[[VAL_3]] : index
! CHECK:         %[[VAL_65:.*]] = arith.subi %[[VAL_52]], %[[VAL_3]] : index
! CHECK:         br ^bb11(%[[VAL_64]], %[[VAL_65]] : index, index)
! CHECK:       ^bb15(%[[VAL_66:.*]]: index, %[[VAL_67:.*]]: index):
! CHECK:         %[[VAL_68:.*]] = arith.cmpi sgt, %[[VAL_67]], %[[VAL_5]] : index
! CHECK:         cond_br %[[VAL_68]], ^bb16(%[[VAL_5]], %[[VAL_11]] : index, index), ^bb19
! CHECK:       ^bb16(%[[VAL_69:.*]]: index, %[[VAL_70:.*]]: index):
! CHECK:         %[[VAL_71:.*]] = arith.cmpi sgt, %[[VAL_70]], %[[VAL_5]] : index
! CHECK:         cond_br %[[VAL_71]], ^bb17, ^bb18
! CHECK:       ^bb17:
! CHECK:         %[[VAL_72:.*]] = arith.addi %[[VAL_69]], %[[VAL_3]] : index
! CHECK:         %[[VAL_73:.*]] = arith.addi %[[VAL_66]], %[[VAL_3]] : index
! CHECK:         %[[VAL_74:.*]] = fir.array_coor %[[VAL_33]](%[[VAL_13]]) %[[VAL_72]], %[[VAL_73]] : (!fir.heap<!fir.array<?x2xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:         %[[VAL_75:.*]] = fir.array_coor %[[VAL_12]](%[[VAL_13]]) %[[VAL_72]], %[[VAL_73]] : (!fir.ref<!fir.array<?x2xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
! CHECK:         %[[VAL_76:.*]] = fir.load %[[VAL_74]] : !fir.ref<f32>
! CHECK:         fir.store %[[VAL_76]] to %[[VAL_75]] : !fir.ref<f32>
! CHECK:         %[[VAL_77:.*]] = arith.subi %[[VAL_70]], %[[VAL_3]] : index
! CHECK:         br ^bb16(%[[VAL_72]], %[[VAL_77]] : index, index)
! CHECK:       ^bb18:
! CHECK:         %[[VAL_78:.*]] = arith.addi %[[VAL_66]], %[[VAL_3]] : index
! CHECK:         %[[VAL_79:.*]] = arith.subi %[[VAL_67]], %[[VAL_3]] : index
! CHECK:         br ^bb15(%[[VAL_78]], %[[VAL_79]] : index, index)
! CHECK:       ^bb19:
! CHECK:         fir.freemem %[[VAL_33]] : !fir.heap<!fir.array<?x2xf32>>
! CHECK:         %[[VAL_82:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_8]], %{{.*}}, %{{.*}}) : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
! CHECK:         %[[VAL_83:.*]] = fir.slice %[[VAL_3]], %[[VAL_2]], %[[VAL_3]], %[[VAL_3]], %[[VAL_2]], %[[VAL_3]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:         %[[VAL_84:.*]] = fir.embox %[[VAL_12]](%[[VAL_13]]) {{\[}}%[[VAL_83]]] : (!fir.ref<!fir.array<?x2xf32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<2x?xf32>>
! CHECK:         %[[VAL_85:.*]] = fir.convert %[[VAL_84]] : (!fir.box<!fir.array<2x?xf32>>) -> !fir.box<none>
! CHECK:         %[[VAL_86:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_82]], %[[VAL_85]]) : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:         %[[VAL_87:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:         %[[VAL_88:.*]] = arith.subi %[[VAL_87]], %[[VAL_6]] : i32
! CHECK:         %[[VAL_89:.*]] = fir.convert %[[VAL_88]] : (i32) -> index
! CHECK:         %[[VAL_90:.*]] = fir.convert %[[VAL_87]] : (i32) -> index
! CHECK:         %[[VAL_91:.*]] = fir.slice %[[VAL_89]], %[[VAL_90]], %[[VAL_3]], %[[VAL_3]], %[[VAL_2]], %[[VAL_3]] : (index, index, index, index, index, index) -> !fir.slice<2>
! CHECK:         %[[VAL_92:.*]] = fir.embox %[[VAL_12]](%[[VAL_13]]) {{\[}}%[[VAL_91]]] : (!fir.ref<!fir.array<?x2xf32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<?x?xf32>>
! CHECK:         %[[VAL_93:.*]] = fir.convert %[[VAL_92]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
! CHECK:         %[[VAL_94:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_82]], %[[VAL_93]]) : (!fir.ref<i8>, !fir.box<none>) -> i1
! CHECK:         %[[VAL_95:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_82]]) : (!fir.ref<i8>) -> i32
! CHECK:         return
! CHECK:       }


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
  ! CHECK-LABEL: func @_QFtt1Pr
  function r(x)
    real x(:)
    r = x(1)
  end
end
