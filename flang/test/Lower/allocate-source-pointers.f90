! RUN: bbc --use-desc-for-alloc=false -emit-fir %s -o - | FileCheck %s

! Test lowering of pointers for allocate statements with source.

! CHECK-LABEL: func.func @_QPtest_pointer_scalar(
! CHECK-SAME:                                    %[[VAL_0:.*]]: !fir.ref<f32> {fir.bindc_name = "a"}) {
! CHECK:         %[[VAL_1:.*]] = fir.address_of(@_QFtest_pointer_scalarEx1) : !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK:         %[[VAL_2:.*]] = fir.address_of(@_QFtest_pointer_scalarEx2) : !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK:         %[[VAL_3:.*]] = arith.constant false
! CHECK:         %[[VAL_4:.*]] = fir.absent !fir.box<none>
! CHECK:         %[[VAL_7:.*]] = fir.embox %[[VAL_0]] : (!fir.ref<f32>) -> !fir.box<f32>
! CHECK:         %[[VAL_8:.*]] = fir.zero_bits !fir.ptr<f32>
! CHECK:         %[[VAL_9:.*]] = fir.embox %[[VAL_8]] : (!fir.ptr<f32>) -> !fir.box<!fir.ptr<f32>>
! CHECK:         fir.store %[[VAL_9]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_7]] : (!fir.box<f32>) -> !fir.box<none>
! CHECK:         %[[VAL_13:.*]] = fir.call @_FortranAPointerAllocateSource(%[[VAL_10]], %[[VAL_11]], %[[VAL_3]], %[[VAL_4]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

subroutine test_pointer_scalar(a)
  real, save, pointer :: x1, x2
  real :: a

  allocate(x1, x2, source = a)
end

! CHECK-LABEL: func.func @_QPtest_pointer_2d_array(
! CHECK-SAME:                                      %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
! CHECK-SAME:                                      %[[VAL_1:.*]]: !fir.ref<!fir.array<?x?xi32>> {fir.bindc_name = "a"}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "sss", uniq_name = "_QFtest_pointer_2d_arrayEsss"}
! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x?xi32>>> {bindc_name = "x1", uniq_name = "_QFtest_pointer_2d_arrayEx1"}
! CHECK:         %[[VAL_4:.*]] = fir.zero_bits !fir.ptr<!fir.array<?x?xi32>>
! CHECK:         %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_6:.*]] = fir.shape %[[VAL_5]], %[[VAL_5]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_7:.*]] = fir.embox %[[VAL_4]](%[[VAL_6]]) : (!fir.ptr<!fir.array<?x?xi32>>, !fir.shape<2>) -> !fir.box<!fir.ptr<!fir.array<?x?xi32>>>
! CHECK:         fir.store %[[VAL_7]] to %[[VAL_3]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xi32>>>>
! CHECK:         %[[VAL_8:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x?xi32>>> {bindc_name = "x2", uniq_name = "_QFtest_pointer_2d_arrayEx2"}
! CHECK:         %[[VAL_13:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x?xi32>>> {bindc_name = "x3", uniq_name = "_QFtest_pointer_2d_arrayEx3"}
! CHECK:         %[[VAL_18:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:         %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i32) -> i64
! CHECK:         %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i64) -> index
! CHECK:         %[[VAL_21:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_22:.*]] = arith.cmpi sgt, %[[VAL_20]], %[[VAL_21]] : index
! CHECK:         %[[VAL_23:.*]] = arith.select %[[VAL_22]], %[[VAL_20]], %[[VAL_21]] : index
! CHECK:         %[[VAL_24:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:         %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (i32) -> i64
! CHECK:         %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (i64) -> index
! CHECK:         %[[VAL_27:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_28:.*]] = arith.cmpi sgt, %[[VAL_26]], %[[VAL_27]] : index
! CHECK:         %[[VAL_29:.*]] = arith.select %[[VAL_28]], %[[VAL_26]], %[[VAL_27]] : index
! CHECK:         %[[VAL_30:.*]] = arith.constant false
! CHECK:         %[[VAL_31:.*]] = fir.absent !fir.box<none>
! CHECK:         %[[VAL_34:.*]] = fir.shape %[[VAL_23]], %[[VAL_29]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_35:.*]] = fir.embox %[[VAL_1]](%[[VAL_34]]) : (!fir.ref<!fir.array<?x?xi32>>, !fir.shape<2>) -> !fir.box<!fir.array<?x?xi32>>
! CHECK:         %[[VAL_36:.*]] = fir.zero_bits !fir.ptr<!fir.array<?x?xi32>>
! CHECK:         %[[VAL_37:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_38:.*]] = fir.shape %[[VAL_37]], %[[VAL_37]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_39:.*]] = fir.embox %[[VAL_36]](%[[VAL_38]]) : (!fir.ptr<!fir.array<?x?xi32>>, !fir.shape<2>) -> !fir.box<!fir.ptr<!fir.array<?x?xi32>>>
! CHECK:         fir.store %[[VAL_39]] to %[[VAL_3]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xi32>>>>
! CHECK:         %[[VAL_40:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_41:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_42:.*]]:3 = fir.box_dims %[[VAL_35]], %[[VAL_41]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
! CHECK:         %[[VAL_43:.*]] = arith.addi %[[VAL_42]]#1, %[[VAL_40]] : index
! CHECK:         %[[VAL_44:.*]] = arith.subi %[[VAL_43]], %[[VAL_40]] : index
! CHECK:         %[[VAL_45:.*]] = arith.constant 0 : i32
! CHECK:         %[[VAL_46:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[VAL_47:.*]] = fir.convert %[[VAL_40]] : (index) -> i64
! CHECK:         %[[VAL_48:.*]] = fir.convert %[[VAL_44]] : (index) -> i64
! CHECK:         %[[VAL_49:.*]] = fir.call @_FortranAPointerSetBounds(%[[VAL_46]], %[[VAL_45]], %[[VAL_47]], %[[VAL_48]]) {{.*}}: (!fir.ref<!fir.box<none>>, i32, i64, i64) -> none
! CHECK:         %[[VAL_50:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_51:.*]]:3 = fir.box_dims %[[VAL_35]], %[[VAL_50]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
! CHECK:         %[[VAL_52:.*]] = arith.addi %[[VAL_51]]#1, %[[VAL_40]] : index
! CHECK:         %[[VAL_53:.*]] = arith.subi %[[VAL_52]], %[[VAL_40]] : index
! CHECK:         %[[VAL_54:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_55:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[VAL_56:.*]] = fir.convert %[[VAL_40]] : (index) -> i64
! CHECK:         %[[VAL_57:.*]] = fir.convert %[[VAL_53]] : (index) -> i64
! CHECK:         %[[VAL_58:.*]] = fir.call @_FortranAPointerSetBounds(%[[VAL_55]], %[[VAL_54]], %[[VAL_56]], %[[VAL_57]]) {{.*}}: (!fir.ref<!fir.box<none>>, i32, i64, i64) -> none
! CHECK:         %[[VAL_59:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[VAL_60:.*]] = fir.convert %[[VAL_35]] : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
! CHECK:         %[[VAL_62:.*]] = fir.call @_FortranAPointerAllocateSource(%[[VAL_59]], %[[VAL_60]], %[[VAL_30]], %[[VAL_31]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:         %[[VAL_76:.*]] = fir.call @_FortranAPointerSetBounds(
! CHECK:         %[[VAL_85:.*]] = fir.call @_FortranAPointerSetBounds(
! CHECK:         %[[VAL_89:.*]] = fir.call @_FortranAPointerAllocateSource(
! CHECK:         %[[VAL_90:.*]] = arith.constant true
! CHECK:         %[[VAL_122:.*]] = fir.call @_FortranAPointerSetBounds(
! CHECK:         %[[VAL_131:.*]] = fir.call @_FortranAPointerSetBounds(
! CHECK:         %[[VAL_135:.*]] = fir.call @_FortranAPointerAllocateSource(%{{.*}}, %{{.*}}, %[[VAL_90]]

subroutine test_pointer_2d_array(n, a)
  integer, pointer :: x1(:,:), x2(:,:), x3(:,:)
  integer :: n, sss, a(n, n)

  allocate(x1, x2, source = a)
  allocate(x3, source = a(1:3:2, 2:3), stat=sss)
end

! CHECK-LABEL: func.func @_QPtest_pointer_with_shapespec(
! CHECK-SAME:                                            %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
! CHECK-SAME:                                            %[[VAL_1:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "a"},
! CHECK-SAME:                                            %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "m"}) {
! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>> {bindc_name = "x1", uniq_name = "_QFtest_pointer_with_shapespecEx1"}
! CHECK:         %[[VAL_4:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
! CHECK:         %[[VAL_5:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_6:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_7:.*]] = fir.embox %[[VAL_4]](%[[VAL_6]]) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:         fir.store %[[VAL_7]] to %[[VAL_3]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_8:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>> {bindc_name = "x2", uniq_name = "_QFtest_pointer_with_shapespecEx2"}
! CHECK:         %[[VAL_9:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
! CHECK:         %[[VAL_10:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_11:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_12:.*]] = fir.embox %[[VAL_9]](%[[VAL_11]]) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:         fir.store %[[VAL_12]] to %[[VAL_8]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_13:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i32) -> i64
! CHECK:         %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i64) -> index
! CHECK:         %[[VAL_16:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_17:.*]] = arith.cmpi sgt, %[[VAL_15]], %[[VAL_16]] : index
! CHECK:         %[[VAL_18:.*]] = arith.select %[[VAL_17]], %[[VAL_15]], %[[VAL_16]] : index
! CHECK:         %[[VAL_19:.*]] = arith.constant false
! CHECK:         %[[VAL_20:.*]] = fir.absent !fir.box<none>
! CHECK:         %[[VAL_23:.*]] = fir.shape %[[VAL_18]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_24:.*]] = fir.embox %[[VAL_1]](%[[VAL_23]]) : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:         %[[VAL_25:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
! CHECK:         %[[VAL_26:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_27:.*]] = fir.shape %[[VAL_26]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_28:.*]] = fir.embox %[[VAL_25]](%[[VAL_27]]) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:         fir.store %[[VAL_28]] to %[[VAL_3]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_29:.*]] = arith.constant 2 : i32
! CHECK:         %[[VAL_30:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:         %[[VAL_31:.*]] = arith.constant 0 : i32
! CHECK:         %[[VAL_32:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[VAL_33:.*]] = fir.convert %[[VAL_29]] : (i32) -> i64
! CHECK:         %[[VAL_34:.*]] = fir.convert %[[VAL_30]] : (i32) -> i64
! CHECK:         %[[VAL_35:.*]] = fir.call @_FortranAPointerSetBounds(%[[VAL_32]], %[[VAL_31]], %[[VAL_33]], %[[VAL_34]]) {{.*}}: (!fir.ref<!fir.box<none>>, i32, i64, i64) -> none
! CHECK:         %[[VAL_36:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[VAL_37:.*]] = fir.convert %[[VAL_24]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
! CHECK:         %[[VAL_39:.*]] = fir.call @_FortranAPointerAllocateSource(%[[VAL_36]], %[[VAL_37]], %[[VAL_19]], %[[VAL_20]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:         %[[VAL_40:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
! CHECK:         %[[VAL_41:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_42:.*]] = fir.shape %[[VAL_41]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_43:.*]] = fir.embox %[[VAL_40]](%[[VAL_42]]) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:         fir.store %[[VAL_43]] to %[[VAL_8]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_44:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_45:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:         %[[VAL_46:.*]] = arith.constant 0 : i32
! CHECK:         %[[VAL_47:.*]] = fir.convert %[[VAL_8]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[VAL_48:.*]] = fir.convert %[[VAL_44]] : (index) -> i64
! CHECK:         %[[VAL_49:.*]] = fir.convert %[[VAL_45]] : (i32) -> i64
! CHECK:         %[[VAL_50:.*]] = fir.call @_FortranAPointerSetBounds(%[[VAL_47]], %[[VAL_46]], %[[VAL_48]], %[[VAL_49]]) {{.*}}: (!fir.ref<!fir.box<none>>, i32, i64, i64) -> none
! CHECK:         %[[VAL_51:.*]] = fir.convert %[[VAL_8]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[VAL_52:.*]] = fir.convert %[[VAL_24]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
! CHECK:         %[[VAL_54:.*]] = fir.call @_FortranAPointerAllocateSource(%[[VAL_51]], %[[VAL_52]], %[[VAL_19]], %[[VAL_20]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:         return
! CHECK:       }

subroutine test_pointer_with_shapespec(n, a, m)
  integer, pointer :: x1(:), x2(:)
  integer :: n, m, a(n)

  allocate(x1(2:m), x2(n), source = a)
end

! CHECK-LABEL: func.func @_QPtest_pointer_from_const(
! CHECK-SAME:                                        %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
! CHECK-SAME:                                        %[[VAL_1:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "a"}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>> {bindc_name = "x1", uniq_name = "_QFtest_pointer_from_constEx1"}
! CHECK:         %[[VAL_3:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
! CHECK:         %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_6:.*]] = fir.embox %[[VAL_3]](%[[VAL_5]]) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:         fir.store %[[VAL_6]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_7:.*]] = arith.constant false
! CHECK:         %[[VAL_8:.*]] = fir.absent !fir.box<none>
! CHECK:         %[[VAL_11:.*]] = arith.constant 5 : index
! CHECK:         %[[VAL_13:.*]] = arith.constant 5 : index
! CHECK:         %[[VAL_14:.*]] = fir.shape %[[VAL_13]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_15:.*]] = fir.array_load %[[VAL_12:.*]](%[[VAL_14]]) : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> !fir.array<5xi32>
! CHECK:         %[[VAL_16:.*]] = fir.allocmem !fir.array<5xi32>
! CHECK:         %[[VAL_17:.*]] = fir.shape %[[VAL_11]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_18:.*]] = fir.array_load %[[VAL_16]](%[[VAL_17]]) : (!fir.heap<!fir.array<5xi32>>, !fir.shape<1>) -> !fir.array<5xi32>
! CHECK:         %[[VAL_19:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_20:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_21:.*]] = arith.subi %[[VAL_11]], %[[VAL_19]] : index
! CHECK:         %[[VAL_22:.*]] = fir.do_loop %[[VAL_23:.*]] = %[[VAL_20]] to %[[VAL_21]] step %[[VAL_19]] unordered iter_args(%[[VAL_24:.*]] = %[[VAL_18]]) -> (!fir.array<5xi32>) {
! CHECK:           %[[VAL_25:.*]] = fir.array_fetch %[[VAL_15]], %[[VAL_23]] : (!fir.array<5xi32>, index) -> i32
! CHECK:           %[[VAL_26:.*]] = fir.array_update %[[VAL_24]], %[[VAL_25]], %[[VAL_23]] : (!fir.array<5xi32>, i32, index) -> !fir.array<5xi32>
! CHECK:           fir.result %[[VAL_26]] : !fir.array<5xi32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_18]], %[[VAL_27:.*]] to %[[VAL_16]] : !fir.array<5xi32>, !fir.array<5xi32>, !fir.heap<!fir.array<5xi32>>
! CHECK:         %[[VAL_28:.*]] = fir.shape %[[VAL_11]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_29:.*]] = fir.embox %[[VAL_16]](%[[VAL_28]]) : (!fir.heap<!fir.array<5xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<5xi32>>
! CHECK:         %[[VAL_30:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
! CHECK:         %[[VAL_31:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_32:.*]] = fir.shape %[[VAL_31]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_33:.*]] = fir.embox %[[VAL_30]](%[[VAL_32]]) : (!fir.ptr<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:         fir.store %[[VAL_33]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_34:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_35:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_36:.*]]:3 = fir.box_dims %[[VAL_29]], %[[VAL_35]] : (!fir.box<!fir.array<5xi32>>, index) -> (index, index, index)
! CHECK:         %[[VAL_37:.*]] = arith.addi %[[VAL_36]]#1, %[[VAL_34]] : index
! CHECK:         %[[VAL_38:.*]] = arith.subi %[[VAL_37]], %[[VAL_34]] : index
! CHECK:         %[[VAL_39:.*]] = arith.constant 0 : i32
! CHECK:         %[[VAL_40:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[VAL_41:.*]] = fir.convert %[[VAL_34]] : (index) -> i64
! CHECK:         %[[VAL_42:.*]] = fir.convert %[[VAL_38]] : (index) -> i64
! CHECK:         %[[VAL_43:.*]] = fir.call @_FortranAPointerSetBounds(%[[VAL_40]], %[[VAL_39]], %[[VAL_41]], %[[VAL_42]]) {{.*}}: (!fir.ref<!fir.box<none>>, i32, i64, i64) -> none
! CHECK:         %[[VAL_44:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[VAL_45:.*]] = fir.convert %[[VAL_29]] : (!fir.box<!fir.array<5xi32>>) -> !fir.box<none>
! CHECK:         %[[VAL_47:.*]] = fir.call @_FortranAPointerAllocateSource(%[[VAL_44]], %[[VAL_45]], %[[VAL_7]], %[[VAL_8]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:         fir.freemem %[[VAL_16]] : !fir.heap<!fir.array<5xi32>>
! CHECK:         return
! CHECK:       }

subroutine test_pointer_from_const(n, a)
  integer, pointer :: x1(:)
  integer :: n, a(n)

  allocate(x1, source = [1, 2, 3, 4, 5])
end

! CHECK-LABEL: func.func @_QPtest_pointer_chararray(
! CHECK-SAME:                                       %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
! CHECK-SAME:                                       %[[VAL_1:.*]]: !fir.boxchar<1> {fir.bindc_name = "a"}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,4>>>> {bindc_name = "x1", uniq_name = "_QFtest_pointer_chararrayEx1"}
! CHECK:         %[[VAL_3:.*]] = fir.zero_bits !fir.ptr<!fir.array<?x!fir.char<1,4>>>
! CHECK:         %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_6:.*]] = fir.embox %[[VAL_3]](%[[VAL_5]]) : (!fir.ptr<!fir.array<?x!fir.char<1,4>>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,4>>>>
! CHECK:         fir.store %[[VAL_6]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,4>>>>>
! CHECK:         %[[VAL_7:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
! CHECK:         %[[VAL_9:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> i64
! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i64) -> index
! CHECK:         %[[VAL_12:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_13:.*]] = arith.cmpi sgt, %[[VAL_11]], %[[VAL_12]] : index
! CHECK:         %[[VAL_14:.*]] = arith.select %[[VAL_13]], %[[VAL_11]], %[[VAL_12]] : index
! CHECK:         %[[VAL_15:.*]] = arith.constant false
! CHECK:         %[[VAL_16:.*]] = fir.absent !fir.box<none>
! CHECK:         %[[VAL_19:.*]] = fir.shape %[[VAL_14]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_20:.*]] = fir.embox %[[VAL_8]](%[[VAL_19]]) typeparams %[[VAL_7]]#1 : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:         %[[VAL_21:.*]] = fir.zero_bits !fir.ptr<!fir.array<?x!fir.char<1,4>>>
! CHECK:         %[[VAL_22:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_23:.*]] = fir.shape %[[VAL_22]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_24:.*]] = fir.embox %[[VAL_21]](%[[VAL_23]]) : (!fir.ptr<!fir.array<?x!fir.char<1,4>>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.char<1,4>>>>
! CHECK:         fir.store %[[VAL_24]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,4>>>>>
! CHECK:         %[[VAL_25:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_26:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_27:.*]]:3 = fir.box_dims %[[VAL_20]], %[[VAL_26]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_28:.*]] = arith.addi %[[VAL_27]]#1, %[[VAL_25]] : index
! CHECK:         %[[VAL_29:.*]] = arith.subi %[[VAL_28]], %[[VAL_25]] : index
! CHECK:         %[[VAL_30:.*]] = arith.constant 0 : i32
! CHECK:         %[[VAL_31:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,4>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[VAL_32:.*]] = fir.convert %[[VAL_25]] : (index) -> i64
! CHECK:         %[[VAL_33:.*]] = fir.convert %[[VAL_29]] : (index) -> i64
! CHECK:         %[[VAL_34:.*]] = fir.call @_FortranAPointerSetBounds(%[[VAL_31]], %[[VAL_30]], %[[VAL_32]], %[[VAL_33]]) {{.*}}: (!fir.ref<!fir.box<none>>, i32, i64, i64) -> none
! CHECK:         %[[VAL_35:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.char<1,4>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[VAL_36:.*]] = fir.convert %[[VAL_20]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.box<none>
! CHECK:         %[[VAL_38:.*]] = fir.call @_FortranAPointerAllocateSource(%[[VAL_35]], %[[VAL_36]], %[[VAL_15]], %[[VAL_16]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:         return
! CHECK:       }

subroutine test_pointer_chararray(n, a)
  character(4), pointer :: x1(:)
  integer :: n
  character(*) :: a(n)

  allocate(x1, source = a)
end

! CHECK-LABEL: func.func @_QPtest_pointer_char(
! CHECK-SAME:                                  %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
! CHECK-SAME:                                  %[[VAL_1:.*]]: !fir.boxchar<1> {fir.bindc_name = "a"}) {
! CHECK:         %[[VAL_2:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.char<1,?>>> {bindc_name = "x1", uniq_name = "_QFtest_pointer_charEx1"}
! CHECK:         %[[VAL_4:.*]] = fir.alloca !fir.ptr<!fir.char<1,?>> {uniq_name = "_QFtest_pointer_charEx1.addr"}
! CHECK:         %[[VAL_5:.*]] = fir.alloca index {uniq_name = "_QFtest_pointer_charEx1.len"}
! CHECK:         %[[VAL_6:.*]] = fir.zero_bits !fir.ptr<!fir.char<1,?>>
! CHECK:         fir.store %[[VAL_6]] to %[[VAL_4]] : !fir.ref<!fir.ptr<!fir.char<1,?>>>
! CHECK:         %[[VAL_7:.*]] = arith.constant false
! CHECK:         %[[VAL_8:.*]] = fir.absent !fir.box<none>
! CHECK:         %[[VAL_11:.*]] = fir.embox %[[VAL_2]]#0 typeparams %[[VAL_2]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK:         %[[VAL_12:.*]] = fir.zero_bits !fir.ptr<!fir.char<1,?>>
! CHECK:         %[[VAL_13:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_14:.*]] = fir.embox %[[VAL_12]] typeparams %[[VAL_13]] : (!fir.ptr<!fir.char<1,?>>, index) -> !fir.box<!fir.ptr<!fir.char<1,?>>>
! CHECK:         fir.store %[[VAL_14]] to %[[VAL_3]] : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK:         %[[VAL_15:.*]] = fir.box_elesize %[[VAL_11]] : (!fir.box<!fir.char<1,?>>) -> index
! CHECK:         %[[VAL_16:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[VAL_17:.*]] = fir.convert %[[VAL_15]] : (index) -> i64
! CHECK:         %[[VAL_18:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_19:.*]] = arith.constant 0 : i32
! CHECK:         %[[VAL_20:.*]] = arith.constant 0 : i32
! CHECK:         %[[VAL_21:.*]] = fir.call @_FortranAPointerNullifyCharacter(%[[VAL_16]], %[[VAL_17]], %[[VAL_18]], %[[VAL_19]], %[[VAL_20]]) {{.*}}: (!fir.ref<!fir.box<none>>, i64, i32, i32, i32) -> none
! CHECK:         %[[VAL_22:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[VAL_23:.*]] = fir.convert %[[VAL_11]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK:         %[[VAL_25:.*]] = fir.call @_FortranAPointerAllocateSource(%[[VAL_22]], %[[VAL_23]], %[[VAL_7]], %[[VAL_8]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:         %[[VAL_26:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK:         %[[VAL_27:.*]] = fir.box_elesize %[[VAL_26]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> index
! CHECK:         %[[VAL_28:.*]] = fir.box_addr %[[VAL_26]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> !fir.ptr<!fir.char<1,?>>
! CHECK:         fir.store %[[VAL_28]] to %[[VAL_4]] : !fir.ref<!fir.ptr<!fir.char<1,?>>>
! CHECK:         fir.store %[[VAL_27]] to %[[VAL_5]] : !fir.ref<index>
! CHECK:         return
! CHECK:       }

subroutine test_pointer_char(n, a)
  character(:), pointer :: x1
  integer :: n
  character(*) :: a

  allocate(x1, source = a)
end

! CHECK-LABEL: func.func @_QPtest_pointer_derived_type(
! CHECK-SAME:                                          %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFtest_pointer_derived_typeTt{x:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>> {fir.bindc_name = "y"}) {
! CHECK:         %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFtest_pointer_derived_typeTt{x:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>> {bindc_name = "z", uniq_name = "_QFtest_pointer_derived_typeEz"}
! CHECK:         %[[VAL_2:.*]] = fir.zero_bits !fir.ptr<!fir.array<?x!fir.type<_QFtest_pointer_derived_typeTt{x:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>
! CHECK:         %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_4:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_5:.*]] = fir.embox %[[VAL_2]](%[[VAL_4]]) : (!fir.ptr<!fir.array<?x!fir.type<_QFtest_pointer_derived_typeTt{x:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFtest_pointer_derived_typeTt{x:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>
! CHECK:         fir.store %[[VAL_5]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFtest_pointer_derived_typeTt{x:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>>
! CHECK:         %[[VAL_6:.*]] = arith.constant false
! CHECK:         %[[VAL_7:.*]] = fir.absent !fir.box<none>
! CHECK:         %[[VAL_10:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFtest_pointer_derived_typeTt{x:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>>
! CHECK:         %[[VAL_11:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_12:.*]]:3 = fir.box_dims %[[VAL_10]], %[[VAL_11]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFtest_pointer_derived_typeTt{x:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_13:.*]] = fir.shift %[[VAL_12]]#0 : (index) -> !fir.shift<1>
! CHECK:         %[[VAL_14:.*]] = fir.rebox %[[VAL_10]](%[[VAL_13]]) : (!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFtest_pointer_derived_typeTt{x:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>, !fir.shift<1>) -> !fir.box<!fir.array<?x!fir.type<_QFtest_pointer_derived_typeTt{x:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>
! CHECK:         %[[VAL_15:.*]] = fir.zero_bits !fir.ptr<!fir.array<?x!fir.type<_QFtest_pointer_derived_typeTt{x:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>
! CHECK:         %[[VAL_16:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_17:.*]] = fir.shape %[[VAL_16]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_18:.*]] = fir.embox %[[VAL_15]](%[[VAL_17]]) : (!fir.ptr<!fir.array<?x!fir.type<_QFtest_pointer_derived_typeTt{x:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFtest_pointer_derived_typeTt{x:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>
! CHECK:         fir.store %[[VAL_18]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFtest_pointer_derived_typeTt{x:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>>
! CHECK:         %[[VAL_19:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_20:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_21:.*]]:3 = fir.box_dims %[[VAL_14]], %[[VAL_20]] : (!fir.box<!fir.array<?x!fir.type<_QFtest_pointer_derived_typeTt{x:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_22:.*]] = arith.addi %[[VAL_21]]#1, %[[VAL_12]]#0 : index
! CHECK:         %[[VAL_23:.*]] = arith.subi %[[VAL_22]], %[[VAL_19]] : index
! CHECK:         %[[VAL_24:.*]] = arith.constant 0 : i32
! CHECK:         %[[VAL_25:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFtest_pointer_derived_typeTt{x:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[VAL_26:.*]] = fir.convert %[[VAL_12]]#0 : (index) -> i64
! CHECK:         %[[VAL_27:.*]] = fir.convert %[[VAL_23]] : (index) -> i64
! CHECK:         %[[VAL_28:.*]] = fir.call @_FortranAPointerSetBounds(%[[VAL_25]], %[[VAL_24]], %[[VAL_26]], %[[VAL_27]]) {{.*}}: (!fir.ref<!fir.box<none>>, i32, i64, i64) -> none
! CHECK:         %[[VAL_29:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QFtest_pointer_derived_typeTt{x:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[VAL_30:.*]] = fir.convert %[[VAL_14]] : (!fir.box<!fir.array<?x!fir.type<_QFtest_pointer_derived_typeTt{x:!fir.box<!fir.ptr<!fir.array<?xi32>>>}>>>) -> !fir.box<none>
! CHECK:         %[[VAL_32:.*]] = fir.call @_FortranAPointerAllocateSource(%[[VAL_29]], %[[VAL_30]], %[[VAL_6]], %[[VAL_7]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:         return
! CHECK:       }

subroutine test_pointer_derived_type(y)
  type t
    integer, pointer :: x(:)
  end type
  type(t), pointer :: z(:), y(:)

  allocate(z, source=y)
end
