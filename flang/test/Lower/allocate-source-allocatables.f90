! RUN: bbc --use-desc-for-alloc=false -emit-fir -hlfir=false %s -o - | FileCheck %s

! Test lowering of allocatables for allocate statements with source.

! CHECK-LABEL: func.func @_QPtest_allocatable_scalar(
! CHECK-SAME:                                        %[[VAL_0:.*]]: !fir.ref<f32> {fir.bindc_name = "a"}) {
! CHECK:         %[[VAL_1:.*]] = fir.address_of(@_QFtest_allocatable_scalarEx1) : !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:         %[[VAL_2:.*]] = fir.address_of(@_QFtest_allocatable_scalarEx2) : !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:         %[[VAL_3:.*]] = arith.constant false
! CHECK:         %[[VAL_4:.*]] = fir.absent !fir.box<none>
! CHECK:         %[[VAL_7:.*]] = fir.embox %[[VAL_0]] : (!fir.ref<f32>) -> !fir.box<f32>
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_7]] : (!fir.box<f32>) -> !fir.box<none>
! CHECK:         %[[VAL_11:.*]] = fir.call @_FortranAAllocatableAllocateSource(%[[VAL_8]], %[[VAL_9]], %[[VAL_3]], %[[VAL_4]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[VAL_13:.*]] = fir.convert %[[VAL_7]] : (!fir.box<f32>) -> !fir.box<none>
! CHECK:         %[[VAL_15:.*]] = fir.call @_FortranAAllocatableAllocateSource(%[[VAL_12]], %[[VAL_13]], %[[VAL_3]], %[[VAL_4]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:         return
! CHECK:       }

subroutine test_allocatable_scalar(a)
  real, save, allocatable :: x1, x2
  real :: a

  allocate(x1, x2, source = a)
end

! CHECK-LABEL: func.func @_QPtest_allocatable_2d_array(
! CHECK-SAME:                                          %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
! CHECK-SAME:                                          %[[VAL_1:.*]]: !fir.ref<!fir.array<?x?xi32>> {fir.bindc_name = "a"}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "sss", uniq_name = "_QFtest_allocatable_2d_arrayEsss"}
! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xi32>>> {bindc_name = "x1", uniq_name = "_QFtest_allocatable_2d_arrayEx1"}
! CHECK:         %[[VAL_4:.*]] = fir.alloca !fir.heap<!fir.array<?x?xi32>> {uniq_name = "_QFtest_allocatable_2d_arrayEx1.addr"}
! CHECK:         %[[VAL_5:.*]] = fir.alloca index {uniq_name = "_QFtest_allocatable_2d_arrayEx1.lb0"}
! CHECK:         %[[VAL_6:.*]] = fir.alloca index {uniq_name = "_QFtest_allocatable_2d_arrayEx1.ext0"}
! CHECK:         %[[VAL_7:.*]] = fir.alloca index {uniq_name = "_QFtest_allocatable_2d_arrayEx1.lb1"}
! CHECK:         %[[VAL_8:.*]] = fir.alloca index {uniq_name = "_QFtest_allocatable_2d_arrayEx1.ext1"}
! CHECK:         %[[VAL_9:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?xi32>>
! CHECK:         fir.store %[[VAL_9]] to %[[VAL_4]] : !fir.ref<!fir.heap<!fir.array<?x?xi32>>>
! CHECK:         %[[VAL_10:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xi32>>> {bindc_name = "x2", uniq_name = "_QFtest_allocatable_2d_arrayEx2"}
! CHECK:         %[[VAL_17:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xi32>>> {bindc_name = "x3", uniq_name = "_QFtest_allocatable_2d_arrayEx3"}
! CHECK:         %[[VAL_24:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:         %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (i32) -> i64
! CHECK:         %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (i64) -> index
! CHECK:         %[[VAL_27:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_28:.*]] = arith.cmpi sgt, %[[VAL_26]], %[[VAL_27]] : index
! CHECK:         %[[VAL_29:.*]] = arith.select %[[VAL_28]], %[[VAL_26]], %[[VAL_27]] : index
! CHECK:         %[[VAL_30:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:         %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (i32) -> i64
! CHECK:         %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (i64) -> index
! CHECK:         %[[VAL_33:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_34:.*]] = arith.cmpi sgt, %[[VAL_32]], %[[VAL_33]] : index
! CHECK:         %[[VAL_35:.*]] = arith.select %[[VAL_34]], %[[VAL_32]], %[[VAL_33]] : index
! CHECK:         %[[VAL_36:.*]] = arith.constant false
! CHECK:         %[[VAL_37:.*]] = fir.absent !fir.box<none>
! CHECK:         %[[VAL_40:.*]] = fir.shape %[[VAL_29]], %[[VAL_35]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_41:.*]] = fir.embox %[[VAL_1]](%[[VAL_40]]) : (!fir.ref<!fir.array<?x?xi32>>, !fir.shape<2>) -> !fir.box<!fir.array<?x?xi32>>
! CHECK:         %[[VAL_42:.*]] = fir.load %[[VAL_5]] : !fir.ref<index>
! CHECK:         %[[VAL_43:.*]] = fir.load %[[VAL_6]] : !fir.ref<index>
! CHECK:         %[[VAL_44:.*]] = fir.load %[[VAL_7]] : !fir.ref<index>
! CHECK:         %[[VAL_45:.*]] = fir.load %[[VAL_8]] : !fir.ref<index>
! CHECK:         %[[VAL_46:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.heap<!fir.array<?x?xi32>>>
! CHECK:         %[[VAL_47:.*]] = fir.shape_shift %[[VAL_42]], %[[VAL_43]], %[[VAL_44]], %[[VAL_45]] : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:         %[[VAL_48:.*]] = fir.embox %[[VAL_46]](%[[VAL_47]]) : (!fir.heap<!fir.array<?x?xi32>>, !fir.shapeshift<2>) -> !fir.box<!fir.heap<!fir.array<?x?xi32>>>
! CHECK:         fir.store %[[VAL_48]] to %[[VAL_3]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
! CHECK:         %[[VAL_49:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_50:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_51:.*]]:3 = fir.box_dims %[[VAL_41]], %[[VAL_50]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
! CHECK:         %[[VAL_52:.*]] = arith.addi %[[VAL_51]]#1, %[[VAL_49]] : index
! CHECK:         %[[VAL_53:.*]] = arith.subi %[[VAL_52]], %[[VAL_49]] : index
! CHECK:         %[[VAL_54:.*]] = arith.constant 0 : i32
! CHECK:         %[[VAL_55:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[VAL_56:.*]] = fir.convert %[[VAL_49]] : (index) -> i64
! CHECK:         %[[VAL_57:.*]] = fir.convert %[[VAL_53]] : (index) -> i64
! CHECK:         fir.call @_FortranAAllocatableSetBounds(%[[VAL_55]], %[[VAL_54]], %[[VAL_56]], %[[VAL_57]]) {{.*}}: (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:         %[[VAL_59:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_60:.*]]:3 = fir.box_dims %[[VAL_41]], %[[VAL_59]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
! CHECK:         %[[VAL_61:.*]] = arith.addi %[[VAL_60]]#1, %[[VAL_49]] : index
! CHECK:         %[[VAL_62:.*]] = arith.subi %[[VAL_61]], %[[VAL_49]] : index
! CHECK:         %[[VAL_63:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_64:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[VAL_65:.*]] = fir.convert %[[VAL_49]] : (index) -> i64
! CHECK:         %[[VAL_66:.*]] = fir.convert %[[VAL_62]] : (index) -> i64
! CHECK:         fir.call @_FortranAAllocatableSetBounds(%[[VAL_64]], %[[VAL_63]], %[[VAL_65]], %[[VAL_66]]) {{.*}}: (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:         %[[VAL_68:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[VAL_69:.*]] = fir.convert %[[VAL_41]] : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
! CHECK:         %[[VAL_71:.*]] = fir.call @_FortranAAllocatableAllocateSource(%[[VAL_68]], %[[VAL_69]], %[[VAL_36]], %[[VAL_37]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:         fir.call @_FortranAAllocatableSetBounds(
! CHECK:         fir.call @_FortranAAllocatableSetBounds(
! CHECK:         %[[VAL_107:.*]] = fir.call @_FortranAAllocatableAllocateSource(
! CHECK:         %[[VAL_114:.*]] = arith.constant true
! CHECK:         fir.call @_FortranAAllocatableSetBounds(
! CHECK:         fir.call @_FortranAAllocatableSetBounds(
! CHECK:         %[[VAL_162:.*]] = fir.call @_FortranAAllocatableAllocateSource(%{{.*}}, %{{.*}}, %[[VAL_114]]

subroutine test_allocatable_2d_array(n, a)
  integer, allocatable :: x1(:,:), x2(:,:), x3(:,:)
  integer :: n, sss, a(n, n)

  allocate(x1, x2, source = a)
  allocate(x3, source = a(1:3:2, 2:3), stat=sss)
end

! CHECK-LABEL: func.func @_QPtest_allocatable_with_shapespec(
! CHECK-SAME:                                                %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
! CHECK-SAME:                                                %[[VAL_1:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "a"},
! CHECK-SAME:                                                %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "m"}) {
! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "x1", uniq_name = "_QFtest_allocatable_with_shapespecEx1"}
! CHECK:         %[[VAL_4:.*]] = fir.alloca !fir.heap<!fir.array<?xi32>> {uniq_name = "_QFtest_allocatable_with_shapespecEx1.addr"}
! CHECK:         %[[VAL_5:.*]] = fir.alloca index {uniq_name = "_QFtest_allocatable_with_shapespecEx1.lb0"}
! CHECK:         %[[VAL_6:.*]] = fir.alloca index {uniq_name = "_QFtest_allocatable_with_shapespecEx1.ext0"}
! CHECK:         %[[VAL_7:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! CHECK:         fir.store %[[VAL_7]] to %[[VAL_4]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:         %[[VAL_8:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "x2", uniq_name = "_QFtest_allocatable_with_shapespecEx2"}
! CHECK:         %[[VAL_9:.*]] = fir.alloca !fir.heap<!fir.array<?xi32>> {uniq_name = "_QFtest_allocatable_with_shapespecEx2.addr"}
! CHECK:         %[[VAL_10:.*]] = fir.alloca index {uniq_name = "_QFtest_allocatable_with_shapespecEx2.lb0"}
! CHECK:         %[[VAL_11:.*]] = fir.alloca index {uniq_name = "_QFtest_allocatable_with_shapespecEx2.ext0"}
! CHECK:         %[[VAL_12:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! CHECK:         fir.store %[[VAL_12]] to %[[VAL_9]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
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
! CHECK:         %[[VAL_25:.*]] = fir.load %[[VAL_5]] : !fir.ref<index>
! CHECK:         %[[VAL_26:.*]] = fir.load %[[VAL_6]] : !fir.ref<index>
! CHECK:         %[[VAL_27:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:         %[[VAL_28:.*]] = fir.shape_shift %[[VAL_25]], %[[VAL_26]] : (index, index) -> !fir.shapeshift<1>
! CHECK:         %[[VAL_29:.*]] = fir.embox %[[VAL_27]](%[[VAL_28]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:         fir.store %[[VAL_29]] to %[[VAL_3]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_30:.*]] = arith.constant 2 : i32
! CHECK:         %[[VAL_31:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:         %[[VAL_32:.*]] = arith.constant 0 : i32
! CHECK:         %[[VAL_33:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[VAL_34:.*]] = fir.convert %[[VAL_30]] : (i32) -> i64
! CHECK:         %[[VAL_35:.*]] = fir.convert %[[VAL_31]] : (i32) -> i64
! CHECK:         fir.call @_FortranAAllocatableSetBounds(%[[VAL_33]], %[[VAL_32]], %[[VAL_34]], %[[VAL_35]]) {{.*}}: (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:         %[[VAL_37:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[VAL_38:.*]] = fir.convert %[[VAL_24]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
! CHECK:         %[[VAL_40:.*]] = fir.call @_FortranAAllocatableAllocateSource(%[[VAL_37]], %[[VAL_38]], %[[VAL_19]], %[[VAL_20]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:         %[[VAL_41:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_42:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_43:.*]]:3 = fir.box_dims %[[VAL_41]], %[[VAL_42]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_44:.*]] = fir.box_addr %[[VAL_41]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:         fir.store %[[VAL_44]] to %[[VAL_4]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:         fir.store %[[VAL_43]]#1 to %[[VAL_6]] : !fir.ref<index>
! CHECK:         fir.store %[[VAL_43]]#0 to %[[VAL_5]] : !fir.ref<index>
! CHECK:         %[[VAL_45:.*]] = fir.load %[[VAL_10]] : !fir.ref<index>
! CHECK:         %[[VAL_46:.*]] = fir.load %[[VAL_11]] : !fir.ref<index>
! CHECK:         %[[VAL_47:.*]] = fir.load %[[VAL_9]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:         %[[VAL_48:.*]] = fir.shape_shift %[[VAL_45]], %[[VAL_46]] : (index, index) -> !fir.shapeshift<1>
! CHECK:         %[[VAL_49:.*]] = fir.embox %[[VAL_47]](%[[VAL_48]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:         fir.store %[[VAL_49]] to %[[VAL_8]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_50:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_51:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:         %[[VAL_52:.*]] = arith.constant 0 : i32
! CHECK:         %[[VAL_53:.*]] = fir.convert %[[VAL_8]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[VAL_54:.*]] = fir.convert %[[VAL_50]] : (index) -> i64
! CHECK:         %[[VAL_55:.*]] = fir.convert %[[VAL_51]] : (i32) -> i64
! CHECK:         fir.call @_FortranAAllocatableSetBounds(%[[VAL_53]], %[[VAL_52]], %[[VAL_54]], %[[VAL_55]]) {{.*}}: (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:         %[[VAL_57:.*]] = fir.convert %[[VAL_8]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[VAL_58:.*]] = fir.convert %[[VAL_24]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
! CHECK:         %[[VAL_60:.*]] = fir.call @_FortranAAllocatableAllocateSource(%[[VAL_57]], %[[VAL_58]], %[[VAL_19]], %[[VAL_20]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

subroutine test_allocatable_with_shapespec(n, a, m)
  integer, allocatable :: x1(:), x2(:)
  integer :: n, m, a(n)

  allocate(x1(2:m), x2(n), source = a)
end

! CHECK-LABEL: func.func @_QPtest_allocatable_from_const(
! CHECK-SAME:                                            %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
! CHECK-SAME:                                            %[[VAL_1:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "a"}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "x1", uniq_name = "_QFtest_allocatable_from_constEx1"}
! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.heap<!fir.array<?xi32>> {uniq_name = "_QFtest_allocatable_from_constEx1.addr"}
! CHECK:         %[[VAL_4:.*]] = fir.alloca index {uniq_name = "_QFtest_allocatable_from_constEx1.lb0"}
! CHECK:         %[[VAL_5:.*]] = fir.alloca index {uniq_name = "_QFtest_allocatable_from_constEx1.ext0"}
! CHECK:         %[[VAL_6:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! CHECK:         fir.store %[[VAL_6]] to %[[VAL_3]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
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
! CHECK:         %[[VAL_27:.*]] = fir.do_loop %[[VAL_23:.*]] = %[[VAL_20]] to %[[VAL_21]] step %[[VAL_19]] unordered iter_args(%[[VAL_24:.*]] = %[[VAL_18]]) -> (!fir.array<5xi32>) {
! CHECK:           %[[VAL_25:.*]] = fir.array_fetch %[[VAL_15]], %[[VAL_23]] : (!fir.array<5xi32>, index) -> i32
! CHECK:           %[[VAL_26:.*]] = fir.array_update %[[VAL_24]], %[[VAL_25]], %[[VAL_23]] : (!fir.array<5xi32>, i32, index) -> !fir.array<5xi32>
! CHECK:           fir.result %[[VAL_26]] : !fir.array<5xi32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_18]], %[[VAL_27]] to %[[VAL_16]] : !fir.array<5xi32>, !fir.array<5xi32>, !fir.heap<!fir.array<5xi32>>
! CHECK:         %[[VAL_28:.*]] = fir.shape %[[VAL_11]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_29:.*]] = fir.embox %[[VAL_16]](%[[VAL_28]]) : (!fir.heap<!fir.array<5xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<5xi32>>
! CHECK:         %[[VAL_30:.*]] = fir.load %[[VAL_4]] : !fir.ref<index>
! CHECK:         %[[VAL_31:.*]] = fir.load %[[VAL_5]] : !fir.ref<index>
! CHECK:         %[[VAL_32:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:         %[[VAL_33:.*]] = fir.shape_shift %[[VAL_30]], %[[VAL_31]] : (index, index) -> !fir.shapeshift<1>
! CHECK:         %[[VAL_34:.*]] = fir.embox %[[VAL_32]](%[[VAL_33]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:         fir.store %[[VAL_34]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_35:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_36:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_37:.*]]:3 = fir.box_dims %[[VAL_29]], %[[VAL_36]] : (!fir.box<!fir.array<5xi32>>, index) -> (index, index, index)
! CHECK:         %[[VAL_38:.*]] = arith.addi %[[VAL_37]]#1, %[[VAL_35]] : index
! CHECK:         %[[VAL_39:.*]] = arith.subi %[[VAL_38]], %[[VAL_35]] : index
! CHECK:         %[[VAL_40:.*]] = arith.constant 0 : i32
! CHECK:         %[[VAL_41:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[VAL_42:.*]] = fir.convert %[[VAL_35]] : (index) -> i64
! CHECK:         %[[VAL_43:.*]] = fir.convert %[[VAL_39]] : (index) -> i64
! CHECK:         fir.call @_FortranAAllocatableSetBounds(%[[VAL_41]], %[[VAL_40]], %[[VAL_42]], %[[VAL_43]]) {{.*}}: (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:         %[[VAL_45:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[VAL_46:.*]] = fir.convert %[[VAL_29]] : (!fir.box<!fir.array<5xi32>>) -> !fir.box<none>
! CHECK:         %[[VAL_48:.*]] = fir.call @_FortranAAllocatableAllocateSource(%[[VAL_45]], %[[VAL_46]], %[[VAL_7]], %[[VAL_8]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:         %[[VAL_49:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_50:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_51:.*]]:3 = fir.box_dims %[[VAL_49]], %[[VAL_50]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_52:.*]] = fir.box_addr %[[VAL_49]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:         fir.store %[[VAL_52]] to %[[VAL_3]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! CHECK:         fir.store %[[VAL_51]]#1 to %[[VAL_5]] : !fir.ref<index>
! CHECK:         fir.store %[[VAL_51]]#0 to %[[VAL_4]] : !fir.ref<index>
! CHECK:         fir.freemem %[[VAL_16]] : !fir.heap<!fir.array<5xi32>>
! CHECK:         return
! CHECK:       }

subroutine test_allocatable_from_const(n, a)
  integer, allocatable :: x1(:)
  integer :: n, a(n)

  allocate(x1, source = [1, 2, 3, 4, 5])
end

! CHECK-LABEL: func.func @_QPtest_allocatable_chararray(
! CHECK-SAME:                                           %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
! CHECK-SAME:                                           %[[VAL_1:.*]]: !fir.boxchar<1> {fir.bindc_name = "a"}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,4>>>> {bindc_name = "x1", uniq_name = "_QFtest_allocatable_chararrayEx1"}
! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.heap<!fir.array<?x!fir.char<1,4>>> {uniq_name = "_QFtest_allocatable_chararrayEx1.addr"}
! CHECK:         %[[VAL_4:.*]] = fir.alloca index {uniq_name = "_QFtest_allocatable_chararrayEx1.lb0"}
! CHECK:         %[[VAL_5:.*]] = fir.alloca index {uniq_name = "_QFtest_allocatable_chararrayEx1.ext0"}
! CHECK:         %[[VAL_6:.*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.char<1,4>>>
! CHECK:         fir.store %[[VAL_6]] to %[[VAL_3]] : !fir.ref<!fir.heap<!fir.array<?x!fir.char<1,4>>>>
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
! CHECK:         %[[VAL_21:.*]] = fir.load %[[VAL_4]] : !fir.ref<index>
! CHECK:         %[[VAL_22:.*]] = fir.load %[[VAL_5]] : !fir.ref<index>
! CHECK:         %[[VAL_23:.*]] = fir.load %[[VAL_3]] : !fir.ref<!fir.heap<!fir.array<?x!fir.char<1,4>>>>
! CHECK:         %[[VAL_24:.*]] = fir.shape_shift %[[VAL_21]], %[[VAL_22]] : (index, index) -> !fir.shapeshift<1>
! CHECK:         %[[VAL_25:.*]] = fir.embox %[[VAL_23]](%[[VAL_24]]) : (!fir.heap<!fir.array<?x!fir.char<1,4>>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,4>>>>
! CHECK:         fir.store %[[VAL_25]] to %[[VAL_2]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,4>>>>>
! CHECK:         %[[VAL_26:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_27:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_28:.*]]:3 = fir.box_dims %[[VAL_20]], %[[VAL_27]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_29:.*]] = arith.addi %[[VAL_28]]#1, %[[VAL_26]] : index
! CHECK:         %[[VAL_30:.*]] = arith.subi %[[VAL_29]], %[[VAL_26]] : index
! CHECK:         %[[VAL_31:.*]] = arith.constant 0 : i32
! CHECK:         %[[VAL_32:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,4>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[VAL_33:.*]] = fir.convert %[[VAL_26]] : (index) -> i64
! CHECK:         %[[VAL_34:.*]] = fir.convert %[[VAL_30]] : (index) -> i64
! CHECK:         fir.call @_FortranAAllocatableSetBounds(%[[VAL_32]], %[[VAL_31]], %[[VAL_33]], %[[VAL_34]]) {{.*}}: (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:         %[[VAL_36:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,4>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[VAL_37:.*]] = fir.convert %[[VAL_20]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.box<none>
! CHECK:         %[[VAL_39:.*]] = fir.call @_FortranAAllocatableAllocateSource(%[[VAL_36]], %[[VAL_37]], %[[VAL_15]], %[[VAL_16]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

subroutine test_allocatable_chararray(n, a)
  character(4), allocatable :: x1(:)
  integer :: n
  character(*) :: a(n)

  allocate(x1, source = a)
end

! CHECK-LABEL: func.func @_QPtest_allocatable_char(
! CHECK-SAME:                                      %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"},
! CHECK-SAME:                                      %[[VAL_1:.*]]: !fir.boxchar<1> {fir.bindc_name = "a"}) {
! CHECK:         %[[VAL_2:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>> {bindc_name = "x1", uniq_name = "_QFtest_allocatable_charEx1"}
! CHECK:         %[[VAL_4:.*]] = fir.alloca !fir.heap<!fir.char<1,?>> {uniq_name = "_QFtest_allocatable_charEx1.addr"}
! CHECK:         %[[VAL_5:.*]] = fir.alloca index {uniq_name = "_QFtest_allocatable_charEx1.len"}
! CHECK:         %[[VAL_6:.*]] = fir.zero_bits !fir.heap<!fir.char<1,?>>
! CHECK:         fir.store %[[VAL_6]] to %[[VAL_4]] : !fir.ref<!fir.heap<!fir.char<1,?>>>
! CHECK:         %[[VAL_7:.*]] = arith.constant false
! CHECK:         %[[VAL_8:.*]] = fir.absent !fir.box<none>
! CHECK:         %[[VAL_11:.*]] = fir.embox %[[VAL_2]]#0 typeparams %[[VAL_2]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK:         %[[VAL_12:.*]] = fir.load %[[VAL_5]] : !fir.ref<index>
! CHECK:         %[[VAL_13:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.heap<!fir.char<1,?>>>
! CHECK:         %[[VAL_14:.*]] = fir.embox %[[VAL_13]] typeparams %[[VAL_12]] : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
! CHECK:         fir.store %[[VAL_14]] to %[[VAL_3]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:         %[[VAL_15:.*]] = fir.box_elesize %[[VAL_11]] : (!fir.box<!fir.char<1,?>>) -> index
! CHECK:         %[[VAL_16:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[VAL_17:.*]] = fir.convert %[[VAL_15]] : (index) -> i64
! CHECK:         %[[VAL_18:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_19:.*]] = arith.constant 0 : i32
! CHECK:         %[[VAL_20:.*]] = arith.constant 0 : i32
! CHECK:         fir.call @_FortranAAllocatableInitCharacterForAllocate(%[[VAL_16]], %[[VAL_17]], %[[VAL_18]], %[[VAL_19]], %[[VAL_20]]) {{.*}}: (!fir.ref<!fir.box<none>>, i64, i32, i32, i32) -> ()
! CHECK:         %[[VAL_22:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[VAL_23:.*]] = fir.convert %[[VAL_11]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK:         %[[VAL_25:.*]] = fir.call @_FortranAAllocatableAllocateSource(%[[VAL_22]], %[[VAL_23]], %[[VAL_7]], %[[VAL_8]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

subroutine test_allocatable_char(n, a)
  character(:), allocatable :: x1
  integer :: n
  character(*) :: a

  allocate(x1, source = a)
end

! CHECK-LABEL: func.func @_QPtest_allocatable_derived_type(
! CHECK-SAME:                                              %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFtest_allocatable_derived_typeTt{x:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>> {fir.bindc_name = "y"}) {
! CHECK:         %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.type<_QFtest_allocatable_derived_typeTt{x:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>> {bindc_name = "z", uniq_name = "_QFtest_allocatable_derived_typeEz"}
! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.heap<!fir.array<?x!fir.type<_QFtest_allocatable_derived_typeTt{x:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>> {uniq_name = "_QFtest_allocatable_derived_typeEz.addr"}
! CHECK:         %[[VAL_3:.*]] = fir.alloca index {uniq_name = "_QFtest_allocatable_derived_typeEz.lb0"}
! CHECK:         %[[VAL_4:.*]] = fir.alloca index {uniq_name = "_QFtest_allocatable_derived_typeEz.ext0"}
! CHECK:         %[[VAL_5:.*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.type<_QFtest_allocatable_derived_typeTt{x:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>
! CHECK:         fir.store %[[VAL_5]] to %[[VAL_2]] : !fir.ref<!fir.heap<!fir.array<?x!fir.type<_QFtest_allocatable_derived_typeTt{x:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>
! CHECK:         %[[VAL_6:.*]] = arith.constant false
! CHECK:         %[[VAL_7:.*]] = fir.absent !fir.box<none>
! CHECK:         %[[VAL_10:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFtest_allocatable_derived_typeTt{x:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>>
! CHECK:         %[[VAL_11:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_12:.*]]:3 = fir.box_dims %[[VAL_10]], %[[VAL_11]] : (!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFtest_allocatable_derived_typeTt{x:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_13:.*]] = fir.shift %[[VAL_12]]#0 : (index) -> !fir.shift<1>
! CHECK:         %[[VAL_14:.*]] = fir.rebox %[[VAL_10]](%[[VAL_13]]) : (!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFtest_allocatable_derived_typeTt{x:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>, !fir.shift<1>) -> !fir.box<!fir.array<?x!fir.type<_QFtest_allocatable_derived_typeTt{x:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>
! CHECK:         %[[VAL_15:.*]] = fir.load %[[VAL_3]] : !fir.ref<index>
! CHECK:         %[[VAL_16:.*]] = fir.load %[[VAL_4]] : !fir.ref<index>
! CHECK:         %[[VAL_17:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.heap<!fir.array<?x!fir.type<_QFtest_allocatable_derived_typeTt{x:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>
! CHECK:         %[[VAL_18:.*]] = fir.shape_shift %[[VAL_15]], %[[VAL_16]] : (index, index) -> !fir.shapeshift<1>
! CHECK:         %[[VAL_19:.*]] = fir.embox %[[VAL_17]](%[[VAL_18]]) : (!fir.heap<!fir.array<?x!fir.type<_QFtest_allocatable_derived_typeTt{x:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?x!fir.type<_QFtest_allocatable_derived_typeTt{x:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>
! CHECK:         fir.store %[[VAL_19]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFtest_allocatable_derived_typeTt{x:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>>
! CHECK:         %[[VAL_20:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_21:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_22:.*]]:3 = fir.box_dims %[[VAL_14]], %[[VAL_21]] : (!fir.box<!fir.array<?x!fir.type<_QFtest_allocatable_derived_typeTt{x:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>, index) -> (index, index, index)
! CHECK:         %[[VAL_23:.*]] = arith.addi %[[VAL_22]]#1, %[[VAL_12]]#0 : index
! CHECK:         %[[VAL_24:.*]] = arith.subi %[[VAL_23]], %[[VAL_20]] : index
! CHECK:         %[[VAL_25:.*]] = arith.constant 0 : i32
! CHECK:         %[[VAL_26:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFtest_allocatable_derived_typeTt{x:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[VAL_27:.*]] = fir.convert %[[VAL_12]]#0 : (index) -> i64
! CHECK:         %[[VAL_28:.*]] = fir.convert %[[VAL_24]] : (index) -> i64
! CHECK:         fir.call @_FortranAAllocatableSetBounds(%[[VAL_26]], %[[VAL_25]], %[[VAL_27]], %[[VAL_28]]) {{.*}}: (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:         %[[VAL_30:.*]] = fir.convert %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFtest_allocatable_derived_typeTt{x:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:         %[[VAL_31:.*]] = fir.convert %[[VAL_14]] : (!fir.box<!fir.array<?x!fir.type<_QFtest_allocatable_derived_typeTt{x:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>) -> !fir.box<none>
! CHECK:         %[[VAL_33:.*]] = fir.call @_FortranAAllocatableAllocateSource(%[[VAL_30]], %[[VAL_31]], %[[VAL_6]], %[[VAL_7]], %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32

subroutine test_allocatable_derived_type(y)
  type t
    integer, allocatable :: x(:)
  end type
  type(t), allocatable :: z(:), y(:)

  allocate(z, source=y)
end
