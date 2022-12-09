! RUN: bbc %s -o - | fir-opt --canonicalize --cse | FileCheck %s

! CHECK-LABEL: func @_QPissue(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.boxchar<1>{{.*}}, %[[VAL_1:.*]]: !fir.boxchar<1>{{.*}}) {
subroutine issue(c1, c2)
  ! CHECK-DAG: %[[VAL_2:.*]] = arith.constant false
  ! CHECK-DAG: %[[VAL_3:.*]] = arith.constant 32 : i8
  ! CHECK-DAG: %[[VAL_4:.*]] = arith.constant 3 : index
  ! CHECK-DAG: %[[VAL_5:.*]] = arith.constant 4 : index
  ! CHECK-DAG: %[[VAL_6:.*]] = arith.constant 0 : index
  ! CHECK-DAG: %[[VAL_7:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_8:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[VAL_9:.*]] = fir.convert %[[VAL_8]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<3x!fir.char<1,4>>>
  ! CHECK: %[[VAL_10:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[VAL_11:.*]] = fir.convert %[[VAL_10]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<3x!fir.char<1,?>>>
  ! CHECK: %[[VAL_12:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
  ! CHECK: cf.br ^bb1(%[[VAL_6]], %[[VAL_4]] : index, index)
  ! CHECK: ^bb1(%[[VAL_13:.*]]: index, %[[VAL_14:.*]]: index):
  ! CHECK: %[[VAL_15:.*]] = arith.cmpi sgt, %[[VAL_14]], %[[VAL_6]] : index
  ! CHECK: cf.cond_br %[[VAL_15]], ^bb2, ^bb6
  ! CHECK: ^bb2:
  ! CHECK: %[[VAL_16:.*]] = arith.addi %[[VAL_13]], %[[VAL_7]] : index
  ! CHECK: %[[VAL_17:.*]] = fir.array_coor %[[VAL_11]](%[[VAL_12]]) %[[VAL_16]] typeparams %[[VAL_10]]#1 : (!fir.ref<!fir.array<3x!fir.char<1,?>>>, !fir.shape<1>, index, index) -> !fir.ref<!fir.char<1,?>>
  ! CHECK: %[[VAL_18:.*]] = fir.array_coor %[[VAL_9]](%[[VAL_12]]) %[[VAL_16]] : (!fir.ref<!fir.array<3x!fir.char<1,4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.char<1,4>>
  ! CHECK: %[[VAL_19:.*]] = arith.cmpi sgt, %[[VAL_10]]#1, %[[VAL_5]] : index
  ! CHECK: %[[VAL_20:.*]] = arith.select %[[VAL_19]], %[[VAL_5]], %[[VAL_10]]#1 : index
  ! CHECK: %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (index) -> i64
  ! CHECK: %[[VAL_22:.*]] = fir.convert %[[VAL_18]] : (!fir.ref<!fir.char<1,4>>) -> !fir.ref<i8>
  ! CHECK: %[[VAL_23:.*]] = fir.convert %[[VAL_17]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK: fir.call @llvm.memmove.p0.p0.i64(%[[VAL_22]], %[[VAL_23]], %[[VAL_21]], %[[VAL_2]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK: %[[VAL_24:.*]] = fir.undefined !fir.char<1>
  ! CHECK: %[[VAL_25:.*]] = fir.insert_value %[[VAL_24]], %[[VAL_3]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
  ! CHECK: %[[VAL_26:.*]] = arith.subi %[[VAL_5]], %[[VAL_20]] : index
  ! CHECK: cf.br ^bb3(%[[VAL_20]], %[[VAL_26]] : index, index)
  ! CHECK: ^bb3(%[[VAL_27:.*]]: index, %[[VAL_28:.*]]: index):
  ! CHECK: %[[VAL_29:.*]] = arith.cmpi sgt, %[[VAL_28]], %[[VAL_6]] : index
  ! CHECK: cf.cond_br %[[VAL_29]], ^bb4, ^bb5
  ! CHECK: ^bb4:
  ! CHECK: %[[VAL_30:.*]] = fir.convert %[[VAL_18]] : (!fir.ref<!fir.char<1,4>>) -> !fir.ref<!fir.array<4x!fir.char<1>>>
  ! CHECK: %[[VAL_31:.*]] = fir.coordinate_of %[[VAL_30]], %[[VAL_27]] : (!fir.ref<!fir.array<4x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK: fir.store %[[VAL_25]] to %[[VAL_31]] : !fir.ref<!fir.char<1>>
  ! CHECK: %[[VAL_32:.*]] = arith.addi %[[VAL_27]], %[[VAL_7]] : index
  ! CHECK: %[[VAL_33:.*]] = arith.subi %[[VAL_28]], %[[VAL_7]] : index
  ! CHECK: cf.br ^bb3(%[[VAL_32]], %[[VAL_33]] : index, index)
  ! CHECK: ^bb5:
  ! CHECK: %[[VAL_34:.*]] = arith.subi %[[VAL_14]], %[[VAL_7]] : index
  ! CHECK: cf.br ^bb1(%[[VAL_16]], %[[VAL_34]] : index, index)

  character(4) :: c1(3)
  character(*) :: c2(3)
  c1 = c2
  ! CHECK:         return
  ! CHECK:       }
end subroutine

! CHECK-LABEL: func @_QQmain() {
program p
  ! CHECK-DAG: %[[VAL_0:.*]] = arith.constant 4 : index
  ! CHECK-DAG: %[[VAL_1:.*]] = arith.constant 3 : index
  ! CHECK-DAG: %[[VAL_2:.*]] = arith.constant -1 : i32
  ! CHECK: %[[VAL_5:.*]] = fir.alloca !fir.array<3x!fir.char<1,4>> {bindc_name = "c1", uniq_name = "_QFEc1"}
  ! CHECK: %[[VAL_6:.*]] = fir.address_of(@_QFEc2) : !fir.ref<!fir.array<3x!fir.char<1,4>>>
  ! CHECK: %[[VAL_7:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,
  ! CHECK: %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
  ! CHECK: %[[VAL_9:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_2]], %[[VAL_8]], %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  ! CHECK: %[[VAL_10:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_11:.*]] = fir.embox %[[VAL_6]](%[[VAL_10]]) : (!fir.ref<!fir.array<3x!fir.char<1,4>>>, !fir.shape<1>) -> !fir.box<!fir.array<3x!fir.char<1,4>>>
  ! CHECK: %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (!fir.box<!fir.array<3x!fir.char<1,4>>>) -> !fir.box<none>
  ! CHECK: %[[VAL_13:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_9]], %[[VAL_12]]) {{.*}}: (!fir.ref<i8>, !fir.box<none>) -> i1
  ! CHECK: %[[VAL_14:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_9]]) {{.*}}: (!fir.ref<i8>) -> i32
  ! CHECK: %[[VAL_15:.*]] = fir.convert %[[VAL_5]] : (!fir.ref<!fir.array<3x!fir.char<1,4>>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK: %[[VAL_16:.*]] = fir.emboxchar %[[VAL_15]], %[[VAL_0]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK: %[[VAL_17:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.array<3x!fir.char<1,4>>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK: %[[VAL_18:.*]] = fir.emboxchar %[[VAL_17]], %[[VAL_0]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK: fir.call @_QPissue(%[[VAL_16]], %[[VAL_18]]) {{.*}}: (!fir.boxchar<1>, !fir.boxchar<1>) -> ()
  ! CHECK: %[[VAL_19:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_2]], %[[VAL_8]], %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  ! CHECK: %[[VAL_20:.*]] = fir.embox %[[VAL_5]](%[[VAL_10]]) : (!fir.ref<!fir.array<3x!fir.char<1,4>>>, !fir.shape<1>) -> !fir.box<!fir.array<3x!fir.char<1,4>>>
  ! CHECK: %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (!fir.box<!fir.array<3x!fir.char<1,4>>>) -> !fir.box<none>
  ! CHECK: %[[VAL_22:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_19]], %[[VAL_21]]) {{.*}}: (!fir.ref<i8>, !fir.box<none>) -> i1
  ! CHECK: %[[VAL_23:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_19]]) {{.*}}: (!fir.ref<i8>) -> i32
  ! CHECK: fir.call @_QPcharlit() {{.*}}: () -> ()
  character(4) :: c1(3)
  character(4) :: c2(3) = ["abcd", "    ", "    "]
  print *, c2
  call issue(c1, c2)
  print *, c1
  call charlit
  ! CHECK:         return
  ! CHECK:       }
end program p

! CHECK-LABEL: func @_QPcharlit() {
subroutine charlit
  ! CHECK-DAG: %[[VAL_0:.*]] = arith.constant -1 : i32
  ! CHECK-DAG: %[[VAL_3:.*]] = arith.constant 3 : index
  ! CHECK-DAG: %[[VAL_4:.*]] = arith.constant false
  ! CHECK-DAG: %[[VAL_5:.*]] = arith.constant 4 : index
  ! CHECK-DAG: %[[VAL_6:.*]] = arith.constant 0 : index
  ! CHECK-DAG: %[[VAL_7:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_8:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,
  ! CHECK: %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
  ! CHECK: %[[VAL_10:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_0]], %[[VAL_9]], %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  ! CHECK: %[[VAL_11:.*]] = fir.address_of(@_QQro.4x3xc1.1636b396a657de68ffb870a885ac44b4) : !fir.ref<!fir.array<4x!fir.char<1,3>>>
  ! CHECK: %[[VAL_12:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_13:.*]] = fir.allocmem !fir.array<4x!fir.char<1,3>>
  ! CHECK: cf.br ^bb1(%[[VAL_6]], %[[VAL_5]] : index, index)
  ! CHECK: ^bb1(%[[VAL_14:.*]]: index, %[[VAL_15:.*]]: index):
  ! CHECK: %[[VAL_16:.*]] = arith.cmpi sgt, %[[VAL_15]], %[[VAL_6]] : index
  ! CHECK: cond_br %[[VAL_16]], ^bb2, ^bb3
  ! CHECK: ^bb2:
  ! CHECK: %[[VAL_17:.*]] = arith.addi %[[VAL_14]], %[[VAL_7]] : index
  ! CHECK: %[[VAL_18:.*]] = fir.array_coor %[[VAL_11]](%[[VAL_12]]) %[[VAL_17]] : (!fir.ref<!fir.array<4x!fir.char<1,3>>>, !fir.shape<1>, index) -> !fir.ref<!fir.char<1,3>>
  ! CHECK: %[[VAL_19:.*]] = fir.array_coor %[[VAL_13]](%[[VAL_12]]) %[[VAL_17]] : (!fir.heap<!fir.array<4x!fir.char<1,3>>>, !fir.shape<1>, index) -> !fir.ref<!fir.char<1,3>>
  ! CHECK: %[[VAL_20:.*]] = fir.convert %[[VAL_3]] : (index) -> i64
  ! CHECK: %[[VAL_21:.*]] = fir.convert %[[VAL_19]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<i8>
  ! CHECK: %[[VAL_22:.*]] = fir.convert %[[VAL_18]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<i8>
  ! CHECK: fir.call @llvm.memmove.p0.p0.i64(%[[VAL_21]], %[[VAL_22]], %[[VAL_20]], %[[VAL_4]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK: %[[VAL_23:.*]] = arith.subi %[[VAL_15]], %[[VAL_7]] : index
  ! CHECK: cf.br ^bb1(%[[VAL_17]], %[[VAL_23]] : index, index)
  ! CHECK: ^bb3:
  ! CHECK: %[[VAL_24:.*]] = fir.embox %[[VAL_13]](%[[VAL_12]]) : (!fir.heap<!fir.array<4x!fir.char<1,3>>>, !fir.shape<1>) -> !fir.box<!fir.array<4x!fir.char<1,3>>>
  ! CHECK: %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (!fir.box<!fir.array<4x!fir.char<1,3>>>) -> !fir.box<none>
  ! CHECK: %[[VAL_26:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_10]], %[[VAL_25]]) {{.*}}: (!fir.ref<i8>, !fir.box<none>) -> i1
  ! CHECK: fir.freemem %[[VAL_13]]
  ! CHECK: %[[VAL_27:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_10]]) {{.*}}: (!fir.ref<i8>) -> i32
  ! CHECK: %[[VAL_28:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_0]], %[[VAL_9]], %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  ! CHECK: %[[VAL_29:.*]] = fir.allocmem !fir.array<4x!fir.char<1,3>>
  ! CHECK: br ^bb4(%[[VAL_6]], %[[VAL_5]] : index, index)
  ! CHECK: ^bb4(%[[VAL_30:.*]]: index, %[[VAL_31:.*]]: index):
  ! CHECK: %[[VAL_32:.*]] = arith.cmpi sgt, %[[VAL_31]], %[[VAL_6]] : index
  ! CHECK: cond_br %[[VAL_32]], ^bb5, ^bb6
  ! CHECK: ^bb5:
  ! CHECK: %[[VAL_33:.*]] = arith.addi %[[VAL_30]], %[[VAL_7]] : index
  ! CHECK: %[[VAL_34:.*]] = fir.array_coor %[[VAL_11]](%[[VAL_12]]) %[[VAL_33]] : (!fir.ref<!fir.array<4x!fir.char<1,3>>>, !fir.shape<1>, index) -> !fir.ref<!fir.char<1,3>>
  ! CHECK: %[[VAL_35:.*]] = fir.array_coor %[[VAL_29]](%[[VAL_12]]) %[[VAL_33]] : (!fir.heap<!fir.array<4x!fir.char<1,3>>>, !fir.shape<1>, index) -> !fir.ref<!fir.char<1,3>>
  ! CHECK: %[[VAL_36:.*]] = fir.convert %[[VAL_3]] : (index) -> i64
  ! CHECK: %[[VAL_37:.*]] = fir.convert %[[VAL_35]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<i8>
  ! CHECK: %[[VAL_38:.*]] = fir.convert %[[VAL_34]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<i8>
  ! CHECK: fir.call @llvm.memmove.p0.p0.i64(%[[VAL_37]], %[[VAL_38]], %[[VAL_36]], %[[VAL_4]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK: %[[VAL_39:.*]] = arith.subi %[[VAL_31]], %[[VAL_7]] : index
  ! CHECK: br ^bb4(%[[VAL_33]], %[[VAL_39]] : index, index)
  ! CHECK: ^bb6:
  ! CHECK: %[[VAL_40:.*]] = fir.embox %[[VAL_29]](%[[VAL_12]]) : (!fir.heap<!fir.array<4x!fir.char<1,3>>>, !fir.shape<1>) -> !fir.box<!fir.array<4x!fir.char<1,3>>>
  ! CHECK: %[[VAL_41:.*]] = fir.convert %[[VAL_40]] : (!fir.box<!fir.array<4x!fir.char<1,3>>>) -> !fir.box<none>
  ! CHECK: %[[VAL_42:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_28]], %[[VAL_41]]) {{.*}}: (!fir.ref<i8>, !fir.box<none>) -> i1
  ! CHECK: fir.freemem %[[VAL_29]] : !fir.heap<!fir.array<4x!fir.char<1,3>>>
  ! CHECK: %[[VAL_43:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_28]]) {{.*}}: (!fir.ref<i8>) -> i32
  ! CHECK: %[[VAL_44:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[VAL_0]], %[[VAL_9]], %{{.*}}) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  ! CHECK: %[[VAL_45:.*]] = fir.allocmem !fir.array<4x!fir.char<1,3>>
  ! CHECK: br ^bb7(%[[VAL_6]], %[[VAL_5]] : index, index)
  ! CHECK: ^bb7(%[[VAL_46:.*]]: index, %[[VAL_47:.*]]: index):
  ! CHECK: %[[VAL_48:.*]] = arith.cmpi sgt, %[[VAL_47]], %[[VAL_6]] : index
  ! CHECK: cond_br %[[VAL_48]], ^bb8, ^bb9
  ! CHECK: ^bb8:
  ! CHECK: %[[VAL_49:.*]] = arith.addi %[[VAL_46]], %[[VAL_7]] : index
  ! CHECK: %[[VAL_50:.*]] = fir.array_coor %[[VAL_11]](%[[VAL_12]]) %[[VAL_49]] : (!fir.ref<!fir.array<4x!fir.char<1,3>>>, !fir.shape<1>, index) -> !fir.ref<!fir.char<1,3>>
  ! CHECK: %[[VAL_51:.*]] = fir.array_coor %[[VAL_45]](%[[VAL_12]]) %[[VAL_49]] : (!fir.heap<!fir.array<4x!fir.char<1,3>>>, !fir.shape<1>, index) -> !fir.ref<!fir.char<1,3>>
  ! CHECK: %[[VAL_52:.*]] = fir.convert %[[VAL_3]] : (index) -> i64
  ! CHECK: %[[VAL_53:.*]] = fir.convert %[[VAL_51]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<i8>
  ! CHECK: %[[VAL_54:.*]] = fir.convert %[[VAL_50]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<i8>
  ! CHECK: fir.call @llvm.memmove.p0.p0.i64(%[[VAL_53]], %[[VAL_54]], %[[VAL_52]], %[[VAL_4]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK: %[[VAL_55:.*]] = arith.subi %[[VAL_47]], %[[VAL_7]] : index
  ! CHECK: br ^bb7(%[[VAL_49]], %[[VAL_55]] : index, index)
  ! CHECK: ^bb9:
  ! CHECK: %[[VAL_56:.*]] = fir.embox %[[VAL_45]](%[[VAL_12]]) : (!fir.heap<!fir.array<4x!fir.char<1,3>>>, !fir.shape<1>) -> !fir.box<!fir.array<4x!fir.char<1,3>>>
  ! CHECK: %[[VAL_57:.*]] = fir.convert %[[VAL_56]] : (!fir.box<!fir.array<4x!fir.char<1,3>>>) -> !fir.box<none>
  ! CHECK: %[[VAL_58:.*]] = fir.call @_FortranAioOutputDescriptor(%[[VAL_44]], %[[VAL_57]]) {{.*}}: (!fir.ref<i8>, !fir.box<none>) -> i1
  ! CHECK: fir.freemem %[[VAL_45]] : !fir.heap<!fir.array<4x!fir.char<1,3>>>
  ! CHECK: %[[VAL_59:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_44]]) {{.*}}: (!fir.ref<i8>) -> i32
  print*, ['AA ', 'MM ', 'MM ', 'ZZ ']
  print*, ['AA ', 'MM ', 'MM ', 'ZZ ']
  print*, ['AA ', 'MM ', 'MM ', 'ZZ ']
  ! CHECK:         return
  ! CHECK:       }
end

! CHECK: fir.global internal @_QQro.4x3xc1.1636b396a657de68ffb870a885ac44b4 constant : !fir.array<4x!fir.char<1,3>>
! CHECK: AA
! CHECK: MM
! CHECK: ZZ
! CHECK-NOT: fir.global internal @_QQro.4x3xc1
! CHECK-NOT: AA
! CHECK-NOT: MM
! CHECK-NOT: ZZ
