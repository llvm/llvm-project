! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

! CHECK-LABEL: eoshift_test1
subroutine eoshift_test1(arr, shift)
    logical, dimension(3) :: arr, res
    integer :: shift
  ! CHECK: %[[resBox:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>
  ! CHECK: %[[res:.*]] = fir.alloca !fir.array<3x!fir.logical<4>> {bindc_name = "res", uniq_name = "_QFeoshift_test1Eres"}
  ! CHECK: %[[resLoad:.*]] = fir.array_load %[[res]]({{.*}}) : (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.shape<1>) -> !fir.array<3x!fir.logical<4>>
  ! CHECK: %[[arr:.*]] = fir.embox %arg0({{.*}}) : (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.shape<1>) -> !fir.box<!fir.array<3x!fir.logical<4>>>
  ! CHECK: %[[bits:.*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.logical<4>>>
  ! CHECK: %[[init:.*]] = fir.embox %[[bits]]({{.*}}) : (!fir.heap<!fir.array<?x!fir.logical<4>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>
  ! CHECK: fir.store %[[init]] to %[[resBox]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>
  ! CHECK:  %[[boundBox:.*]] = fir.absent !fir.box<none>
  ! CHECK: %[[shift:.*]] = fir.load %arg1 : !fir.ref<i32>
  
    res = eoshift(arr, shift)
  
  ! CHECK: %[[resIRBox:.*]] = fir.convert %[[resBox]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: %[[arrBox:.*]] = fir.convert %[[arr]] : (!fir.box<!fir.array<3x!fir.logical<4>>>) -> !fir.box<none>
  ! CHECK: %[[shiftBox:.*]] = fir.convert %[[shift]] : (i32) -> i64
  ! CHECK: %[[tmp:.*]] = fir.call @_FortranAEoshiftVector(%[[resIRBox]], %[[arrBox]], %[[shiftBox]], %[[boundBox]], {{.*}}, {{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, i64, !fir.box<none>, !fir.ref<i8>, i32) -> none
  ! CHECK: fir.array_merge_store %[[resLoad]], {{.*}} to %[[res]] : !fir.array<3x!fir.logical<4>>, !fir.array<3x!fir.logical<4>>, !fir.ref<!fir.array<3x!fir.logical<4>>>
  end subroutine eoshift_test1
  
  ! CHECK-LABEL: eoshift_test2
  subroutine eoshift_test2(arr, shift, bound, dim)
    integer, dimension(3,3) :: arr, res
    integer, dimension(3) :: shift
    integer :: bound, dim
  ! CHECK: %[[resBox:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xi32>>>
  ! CHECK: %[[res:.*]] = fir.alloca !fir.array<3x3xi32> {bindc_name = "res", uniq_name = "_QFeoshift_test2Eres"}
  !CHECK: %[[resLoad:.*]] = fir.array_load %[[res]]({{.*}}) : (!fir.ref<!fir.array<3x3xi32>>, !fir.shape<2>) -> !fir.array<3x3xi32>
    
    res = eoshift(arr, shift, bound, dim)
  
  ! CHECK: %[[arr:.*]] = fir.embox %arg0({{.*}}) : (!fir.ref<!fir.array<3x3xi32>>, !fir.shape<2>) -> !fir.box<!fir.array<3x3xi32>>
  ! CHECK: %[[boundBox:.*]] = fir.embox %arg2 : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK: %[[dim:.*]] = fir.load %arg3 : !fir.ref<i32>
  ! CHECK: %[[shiftBox:.*]] = fir.embox %arg1({{.*}}) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<3xi32>>
  ! CHECK: %[[resIRBox:.*]] = fir.convert %[[resBox]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: %[[arrBox:.*]] = fir.convert %[[arr]] : (!fir.box<!fir.array<3x3xi32>>) -> !fir.box<none>
  ! CHECK: %[[shiftBoxNone:.*]] = fir.convert %[[shiftBox]] : (!fir.box<!fir.array<3xi32>>) -> !fir.box<none>
  ! CHECK: %[[boundBoxNone:.*]] = fir.convert %[[boundBox]] : (!fir.box<i32>) -> !fir.box<none>
  
  ! CHECK: %[[tmp:.*]] = fir.call @_FortranAEoshift(%[[resIRBox]], %[[arrBox]], %[[shiftBoxNone]], %[[boundBoxNone]], %[[dim]], {{.*}}, {{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none
  ! CHECK: fir.array_merge_store %[[resLoad]], {{.*}} to %[[res]] : !fir.array<3x3xi32>, !fir.array<3x3xi32>, !fir.ref<!fir.array<3x3xi32>>
  end subroutine eoshift_test2
  
  ! CHECK-LABEL: eoshift_test3
  subroutine eoshift_test3(arr, shift, dim)
    character(4), dimension(3,3) :: arr, res
    integer :: shift, dim
  
  ! CHECK: %[[resBox:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,4>>>>
  ! CHECK: %[[arr:.*]]:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[array:.*]] = fir.convert %[[arr]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<3x3x!fir.char<1,4>>>
  ! CHECK: %[[res:.*]] = fir.alloca !fir.array<3x3x!fir.char<1,4>> {bindc_name = "res", uniq_name = "_QFeoshift_test3Eres"}
  ! CHECK: %[[resLoad:.*]] = fir.array_load %[[res]]({{.*}}) : (!fir.ref<!fir.array<3x3x!fir.char<1,4>>>, !fir.shape<2>) -> !fir.array<3x3x!fir.char<1,4>>
  ! CHECK: %[[arrayBox:.*]] = fir.embox %[[array]]({{.*}}) : (!fir.ref<!fir.array<3x3x!fir.char<1,4>>>, !fir.shape<2>) -> !fir.box<!fir.array<3x3x!fir.char<1,4>>>
  ! CHECK: %[[dim:.*]] = fir.load %arg2 : !fir.ref<i32>
  
    res = eoshift(arr, SHIFT=shift, DIM=dim)
  
  ! CHECK: %[[boundBox:.*]] = fir.absent !fir.box<none>
  ! CHECK: %[[shiftBox:.*]] = fir.embox %arg1 : (!fir.ref<i32>) -> !fir.box<i32>
  ! CHECK: %[[resIRBox:.*]] = fir.convert %[[resBox]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?x!fir.char<1,4>>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: %[[arrayBoxNone:.*]] = fir.convert %[[arrayBox]] : (!fir.box<!fir.array<3x3x!fir.char<1,4>>>) -> !fir.box<none>
  ! CHECK: %[[shiftBoxNone:.*]] = fir.convert %[[shiftBox]] : (!fir.box<i32>) -> !fir.box<none>
  ! CHECK: %[[tmp:.*]] = fir.call @_FortranAEoshift(%[[resIRBox]], %[[arrayBoxNone]], %[[shiftBoxNone]], %[[boundBox]], %[[dim]], {{.*}}, {{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none
  ! CHECK: fir.array_merge_store %[[resLoad]], {{.*}} to %[[res]] : !fir.array<3x3x!fir.char<1,4>>, !fir.array<3x3x!fir.char<1,4>>, !fir.ref<!fir.array<3x3x!fir.char<1,4>>>
  end subroutine eoshift_test3
  
  ! CHECK-LABEL: func @_QPeoshift_test_dynamic_optional(
  ! CHECK-SAME:  %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xi32>>
  ! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<i32>
  ! CHECK-SAME:  %[[VAL_2:.*]]: !fir.ref<!fir.array<10xi32>>
  subroutine eoshift_test_dynamic_optional(array, shift, boundary)
    type t
      integer :: i
    end type
    integer :: array(:, :)
    integer :: shift
    integer, optional :: boundary(10)
    call next(eoshift(array, shift, boundary))
  ! CHECK:  %[[VAL_4:.*]] = arith.constant 10 : index
  ! CHECK:  %[[VAL_5:.*]] = fir.is_present %[[VAL_2]] : (!fir.ref<!fir.array<10xi32>>) -> i1
  ! CHECK:  %[[VAL_6:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
  ! CHECK:  %[[VAL_7:.*]] = fir.embox %[[VAL_2]](%[[VAL_6]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
  ! CHECK:  %[[VAL_8:.*]] = fir.absent !fir.box<!fir.array<10xi32>>
  ! CHECK:  %[[VAL_9:.*]] = arith.select %[[VAL_5]], %[[VAL_7]], %[[VAL_8]] : !fir.box<!fir.array<10xi32>>
  ! CHECK:  %[[VAL_21:.*]] = fir.convert %[[VAL_9]] : (!fir.box<!fir.array<10xi32>>) -> !fir.box<none>
  ! CHECK:  fir.call @_FortranAEoshift(%{{.*}}, %{{.*}}, %{{.*}}, %[[VAL_21]], %{{.*}}, %{{.*}}, %{{.*}}) {{.*}}: (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none
  end subroutine
