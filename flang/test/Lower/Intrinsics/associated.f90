! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

! CHECK-LABEL: associated_test
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.box<!fir.ptr<f32>>>{{.*}}, %[[arg1:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>{{.*}})
subroutine associated_test(scalar, array)
    real, pointer :: scalar, array(:)
    real, target :: ziel
    ! CHECK: %[[ziel:.*]] = fir.alloca f32 {bindc_name = "ziel"
    ! CHECK: %[[scalar:.*]] = fir.load %[[arg0]] : !fir.ref<!fir.box<!fir.ptr<f32>>>
    ! CHECK: %[[addr0:.*]] = fir.box_addr %[[scalar]] : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
    ! CHECK: %[[addrToInt0:.*]] = fir.convert %[[addr0]]
    ! CHECK: cmpi ne, %[[addrToInt0]], %c0{{.*}}
    print *, associated(scalar)
    ! CHECK: %[[array:.*]] = fir.load %[[arg1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
    ! CHECK: %[[addr1:.*]] = fir.box_addr %[[array]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
    ! CHECK: %[[addrToInt1:.*]] = fir.convert %[[addr1]]
    ! CHECK: cmpi ne, %[[addrToInt1]], %c0{{.*}}
    print *, associated(array)
    ! CHECK: %[[zbox0:.*]] = fir.embox %[[ziel]] : (!fir.ref<f32>) -> !fir.box<f32>
    ! CHECK: %[[scalar:.*]] = fir.load %[[arg0]] : !fir.ref<!fir.box<!fir.ptr<f32>>>
    ! CHECK: %[[sbox:.*]] = fir.convert %[[scalar]] : (!fir.box<!fir.ptr<f32>>) -> !fir.box<none>
    ! CHECK: %[[zbox:.*]] = fir.convert %[[zbox0]] : (!fir.box<f32>) -> !fir.box<none>
    ! CHECK: fir.call @_FortranAPointerIsAssociatedWith(%[[sbox]], %[[zbox]]) {{.*}}: (!fir.box<none>, !fir.box<none>) -> i1
    print *, associated(scalar, ziel)
  end subroutine
  
  subroutine test_func_results()
    interface
      function get_pointer()
        real, pointer :: get_pointer(:) 
      end function
    end interface
    ! CHECK: %[[result:.*]] = fir.call @_QPget_pointer() {{.*}}: () -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
    ! CHECK: fir.save_result %[[result]] to %[[box_storage:.*]] : !fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
    ! CHECK: %[[box:.*]] = fir.load %[[box_storage]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
    ! CHECK: %[[addr:.*]] = fir.box_addr %[[box]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
    ! CHECK: %[[addr_cast:.*]] = fir.convert %[[addr]] : (!fir.ptr<!fir.array<?xf32>>) -> i64
    ! CHECK:  arith.cmpi ne, %[[addr_cast]], %c0{{.*}} : i64
    print *, associated(get_pointer())
  end subroutine
  
  ! CHECK-LABEL: func @_QPtest_optional_target_1(
  ! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {fir.bindc_name = "p"},
  ! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<!fir.array<10xf32>> {fir.bindc_name = "optionales_ziel", fir.optional, fir.target}) {
  subroutine test_optional_target_1(p, optionales_ziel)
    real, pointer :: p(:)
    real, optional, target :: optionales_ziel(10)
    print *, associated(p, optionales_ziel)
  ! CHECK:  %[[VAL_2:.*]] = arith.constant 10 : index
  ! CHECK:  %[[VAL_3:.*]] = fir.is_present %[[VAL_1]] : (!fir.ref<!fir.array<10xf32>>) -> i1
  ! CHECK:  %[[VAL_4:.*]] = fir.if %[[VAL_3]] -> (!fir.box<!fir.array<10xf32>>) {
  ! CHECK:    %[[VAL_5:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
  ! CHECK:    %[[VAL_6:.*]] = fir.embox %[[VAL_1]](%[[VAL_5]]) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xf32>>
  ! CHECK:    fir.result %[[VAL_6]] : !fir.box<!fir.array<10xf32>>
  ! CHECK:  } else {
  ! CHECK:    %[[VAL_8:.*]] = fir.absent !fir.box<!fir.array<10xf32>>
  ! CHECK:    fir.result %[[VAL_8]] : !fir.box<!fir.array<10xf32>>
  ! CHECK:  }
  ! CHECK:  %[[VAL_13:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK:  %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.box<none>
  ! CHECK:  %[[VAL_15:.*]] = fir.convert %[[VAL_4]] : (!fir.box<!fir.array<10xf32>>) -> !fir.box<none>
  ! CHECK:  fir.call @_FortranAPointerIsAssociatedWith(%[[VAL_14]], %[[VAL_15]]) {{.*}}: (!fir.box<none>, !fir.box<none>) -> i1
  end subroutine
  
  ! CHECK-LABEL: func @_QPtest_optional_target_2(
  ! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {fir.bindc_name = "p"},
  ! CHECK-SAME:  %[[VAL_1:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "optionales_ziel", fir.optional, fir.target}) {
  subroutine test_optional_target_2(p, optionales_ziel)
    real, pointer :: p(:)
    real, optional, target :: optionales_ziel(:)
    print *, associated(p, optionales_ziel)
  ! CHECK:  %[[VAL_7:.*]] = fir.is_present %[[VAL_1]] : (!fir.box<!fir.array<?xf32>>) -> i1
  ! CHECK:  %[[VAL_8:.*]] = fir.if %[[VAL_7]] -> (!fir.box<!fir.array<?xf32>>) {
  ! CHECK:    fir.result %[[VAL_1]] : !fir.box<!fir.array<?xf32>>
  ! CHECK:  } else {
  ! CHECK:    %[[VAL_10:.*]] = fir.absent !fir.box<!fir.array<?xf32>>
  ! CHECK:    fir.result %[[VAL_10]] : !fir.box<!fir.array<?xf32>>
  ! CHECK:  }
  ! CHECK:  %[[VAL_10:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK:  %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.box<none>
  ! CHECK:  %[[VAL_12:.*]] = fir.convert %[[VAL_8]] : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
  ! CHECK:  fir.call @_FortranAPointerIsAssociatedWith(%[[VAL_11]], %[[VAL_12]]) {{.*}}: (!fir.box<none>, !fir.box<none>) -> i1
  end subroutine
  
  ! CHECK-LABEL: func @_QPtest_optional_target_3(
  ! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {fir.bindc_name = "p"},
  ! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {fir.bindc_name = "optionales_ziel", fir.optional}) {
  subroutine test_optional_target_3(p, optionales_ziel)
    real, pointer :: p(:)
    real, optional, pointer :: optionales_ziel(:)
    print *, associated(p, optionales_ziel)
  ! CHECK:  %[[VAL_8:.*]] = fir.is_present %[[VAL_1]] : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> i1
  ! CHECK:  %[[VAL_9:.*]] = fir.if %[[VAL_8]] -> (!fir.box<!fir.ptr<!fir.array<?xf32>>>) {
  ! CHECK:    %[[VAL_10:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK:    fir.result %[[VAL_10]] : !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK:  } else {
  ! CHECK:    %[[VAL_12:.*]] = fir.absent !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK:    fir.result %[[VAL_12]] : !fir.box<!fir.ptr<!fir.array<?xf32>>>
  ! CHECK:  }
  ! CHECK:  %[[VAL_11:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK:  %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.box<none>
  ! CHECK:  %[[VAL_13:.*]] = fir.convert %[[VAL_9]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.box<none>
  ! CHECK:  fir.call @_FortranAPointerIsAssociatedWith(%[[VAL_12]], %[[VAL_13]]) {{.*}}: (!fir.box<none>, !fir.box<none>) -> i1
  end subroutine
  
  ! CHECK-LABEL: func @_QPtest_optional_target_4(
  ! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {fir.bindc_name = "p"},
  ! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {fir.bindc_name = "optionales_ziel", fir.optional, fir.target}) {
  subroutine test_optional_target_4(p, optionales_ziel)
    real, pointer :: p(:)
    real, optional, allocatable, target :: optionales_ziel(:)
    print *, associated(p, optionales_ziel)
  ! CHECK:  %[[VAL_8:.*]] = fir.is_present %[[VAL_1]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> i1
  ! CHECK:  %[[VAL_9:.*]] = fir.if %[[VAL_8]] -> (!fir.box<!fir.heap<!fir.array<?xf32>>>) {
  ! CHECK:    %[[VAL_10:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK:    fir.result %[[VAL_10]] : !fir.box<!fir.heap<!fir.array<?xf32>>>
  ! CHECK:  } else {
  ! CHECK:    %[[VAL_12:.*]] = fir.absent !fir.box<!fir.heap<!fir.array<?xf32>>>
  ! CHECK:    fir.result %[[VAL_12]] : !fir.box<!fir.heap<!fir.array<?xf32>>>
  ! CHECK:  }
  ! CHECK:  %[[VAL_11:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK:  %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.box<none>
  ! CHECK:  %[[VAL_13:.*]] = fir.convert %[[VAL_9]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.box<none>
  ! CHECK:  fir.call @_FortranAPointerIsAssociatedWith(%[[VAL_12]], %[[VAL_13]]) {{.*}}: (!fir.box<none>, !fir.box<none>) -> i1
  end subroutine
  
  ! CHECK-LABEL: func @_QPtest_pointer_target(
  ! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {fir.bindc_name = "p"},
  ! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {fir.bindc_name = "pointer_ziel"}) {
  subroutine test_pointer_target(p, pointer_ziel)
    real, pointer :: p(:)
    real, pointer :: pointer_ziel(:)
    print *, associated(p, pointer_ziel)
  ! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK:  %[[VAL_8:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK:  %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.box<none>
  ! CHECK:  %[[VAL_10:.*]] = fir.convert %[[VAL_7]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.box<none>
  ! CHECK:  fir.call @_FortranAPointerIsAssociatedWith(%[[VAL_9]], %[[VAL_10]]) {{.*}}: (!fir.box<none>, !fir.box<none>) -> i1
  end subroutine
  
  ! CHECK-LABEL: func @_QPtest_allocatable_target(
  ! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {fir.bindc_name = "p"},
  ! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {fir.bindc_name = "allocatable_ziel", fir.target}) {
  subroutine test_allocatable_target(p, allocatable_ziel)
    real, pointer :: p(:)
    real, allocatable, target :: allocatable_ziel(:)
  ! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK:  %[[VAL_8:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK:  %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.box<none>
  ! CHECK:  %[[VAL_10:.*]] = fir.convert %[[VAL_7]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.box<none>
  ! CHECK:  fir.call @_FortranAPointerIsAssociatedWith(%[[VAL_9]], %[[VAL_10]]) {{.*}}: (!fir.box<none>, !fir.box<none>) -> i1
    print *, associated(p, allocatable_ziel)
  end subroutine

subroutine test_optional_argument(a, b)
  integer, pointer :: a
  integer, optional, pointer :: b
  logical :: assoc

  assoc = associated(a, b)
end subroutine

! CHECK-LABEL: func.func @_QPtest_optional_argument(
! CHECK-SAME: %[[A:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "a"}, %[[B:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "b", fir.optional}) {
! CHECK: %[[IS_PRESENT_B:.*]] = fir.is_present %[[B]] : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> i1
! CHECK: %[[BOX_B:.*]] = fir.if %[[IS_PRESENT_B]] -> (!fir.box<!fir.ptr<i32>>) {
! CHECK:   %[[LOADED_B:.*]] = fir.load %[[B]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:   fir.result %[[LOADED_B]] : !fir.box<!fir.ptr<i32>>
! CHECK: } else {
! CHECK:   %[[ABSENT_B:.*]] = fir.absent !fir.box<!fir.ptr<i32>>
! CHECK:   fir.result %[[ABSENT_B]] : !fir.box<!fir.ptr<i32>>
! CHECK: }
! CHECK: %[[LOADED_A:.*]] = fir.load %[[A]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK: %[[BOX_NONE_A:.*]] = fir.convert %[[LOADED_A]] : (!fir.box<!fir.ptr<i32>>) -> !fir.box<none>
! CHECK: %[[BOX_NONE_B:.*]] = fir.convert %[[BOX_B]] : (!fir.box<!fir.ptr<i32>>) -> !fir.box<none>
! CHECK: %{{.*}} fir.call @_FortranAPointerIsAssociatedWith(%[[BOX_NONE_A]], %[[BOX_NONE_B]]) fastmath<contract> : (!fir.box<none>, !fir.box<none>) -> i1
