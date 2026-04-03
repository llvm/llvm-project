! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPiall_test_1(
! CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?xi8>>{{.*}}) -> i8 {
integer(1) function iall_test_1(a)
integer(1) :: a(:)
! CHECK-DAG:  %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK-DAG:  %[[ARG_DECL:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[SCOPE]] arg 1 {{.*}} : (!fir.box<!fir.array<?xi8>>, !fir.dscope) -> (!fir.box<!fir.array<?xi8>>, !fir.box<!fir.array<?xi8>>)
! CHECK-DAG:  %[[RES_ALLOCA:.*]] = fir.alloca i8
! CHECK-DAG:  %[[RES_DECL:.*]]:2 = hlfir.declare %[[RES_ALLOCA]] {{.*}} : (!fir.ref<i8>) -> (!fir.ref<i8>, !fir.ref<i8>)
! CHECK-DAG:  %[[ABSENT:.*]] = fir.absent !fir.box<i1>
! CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
! CHECK-DAG:  %[[CONV_ARG:.*]] = fir.convert %[[ARG_DECL]]#1 : (!fir.box<!fir.array<?xi8>>) -> !fir.box<none>
! CHECK-DAG:  %[[CONV_C0:.*]] = fir.convert %[[C0]] : (index) -> i32
! CHECK-DAG:  %[[CONV_ABSENT:.*]] = fir.convert %[[ABSENT]] : (!fir.box<i1>) -> !fir.box<none>
iall_test_1 = iall(a)
! CHECK:  %[[RESULT:.*]] = fir.call @_FortranAIAll1(%[[CONV_ARG]], %{{.*}}, %{{.*}}, %[[CONV_C0]], %[[CONV_ABSENT]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i8
! CHECK:  hlfir.assign %[[RESULT]] to %[[RES_DECL]]#0 : i8, !fir.ref<i8>
end function

! CHECK-LABEL: func @_QPiall_test_2(
! CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?xi16>>{{.*}}) -> i16 {
integer(2) function iall_test_2(a)
integer(2) :: a(:)
! CHECK-DAG:  %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK-DAG:  %[[ARG_DECL:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[SCOPE]] arg 1 {{.*}} : (!fir.box<!fir.array<?xi16>>, !fir.dscope) -> (!fir.box<!fir.array<?xi16>>, !fir.box<!fir.array<?xi16>>)
! CHECK-DAG:  %[[RES_ALLOCA:.*]] = fir.alloca i16
! CHECK-DAG:  %[[RES_DECL:.*]]:2 = hlfir.declare %[[RES_ALLOCA]] {{.*}} : (!fir.ref<i16>) -> (!fir.ref<i16>, !fir.ref<i16>)
! CHECK-DAG:  %[[ABSENT:.*]] = fir.absent !fir.box<i1>
! CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
! CHECK-DAG:  %[[CONV_ARG:.*]] = fir.convert %[[ARG_DECL]]#1 : (!fir.box<!fir.array<?xi16>>) -> !fir.box<none>
! CHECK-DAG:  %[[CONV_C0:.*]] = fir.convert %[[C0]] : (index) -> i32
! CHECK-DAG:  %[[CONV_ABSENT:.*]] = fir.convert %[[ABSENT]] : (!fir.box<i1>) -> !fir.box<none>
iall_test_2 = iall(a)
! CHECK:  %[[RESULT:.*]] = fir.call @_FortranAIAll2(%[[CONV_ARG]], %{{.*}}, %{{.*}}, %[[CONV_C0]], %[[CONV_ABSENT]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i16
! CHECK:  hlfir.assign %[[RESULT]] to %[[RES_DECL]]#0 : i16, !fir.ref<i16>
end function

! CHECK-LABEL: func @_QPiall_test_4(
! CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}) -> i32 {
integer function iall_test_4(a)
integer :: a(:)
! CHECK-DAG:  %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK-DAG:  %[[ARG_DECL:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[SCOPE]] arg 1 {{.*}} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK-DAG:  %[[RES_ALLOCA:.*]] = fir.alloca i32
! CHECK-DAG:  %[[RES_DECL:.*]]:2 = hlfir.declare %[[RES_ALLOCA]] {{.*}} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-DAG:  %[[ABSENT:.*]] = fir.absent !fir.box<i1>
! CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
! CHECK-DAG:  %[[CONV_ARG:.*]] = fir.convert %[[ARG_DECL]]#1 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
! CHECK-DAG:  %[[CONV_C0:.*]] = fir.convert %[[C0]] : (index) -> i32
! CHECK-DAG:  %[[CONV_ABSENT:.*]] = fir.convert %[[ABSENT]] : (!fir.box<i1>) -> !fir.box<none>
iall_test_4 = iall(a)
! CHECK:  %[[RESULT:.*]] = fir.call @_FortranAIAll4(%[[CONV_ARG]], %{{.*}}, %{{.*}}, %[[CONV_C0]], %[[CONV_ABSENT]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32
! CHECK:  hlfir.assign %[[RESULT]] to %[[RES_DECL]]#0 : i32, !fir.ref<i32>
end function

! CHECK-LABEL: func @_QPiall_test_8(
! CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?xi64>>{{.*}}) -> i64 {
integer(8) function iall_test_8(a)
integer(8) :: a(:)
! CHECK-DAG:  %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK-DAG:  %[[ARG_DECL:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[SCOPE]] arg 1 {{.*}} : (!fir.box<!fir.array<?xi64>>, !fir.dscope) -> (!fir.box<!fir.array<?xi64>>, !fir.box<!fir.array<?xi64>>)
! CHECK-DAG:  %[[RES_ALLOCA:.*]] = fir.alloca i64
! CHECK-DAG:  %[[RES_DECL:.*]]:2 = hlfir.declare %[[RES_ALLOCA]] {{.*}} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK-DAG:  %[[ABSENT:.*]] = fir.absent !fir.box<i1>
! CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
! CHECK-DAG:  %[[CONV_ARG:.*]] = fir.convert %[[ARG_DECL]]#1 : (!fir.box<!fir.array<?xi64>>) -> !fir.box<none>
! CHECK-DAG:  %[[CONV_C0:.*]] = fir.convert %[[C0]] : (index) -> i32
! CHECK-DAG:  %[[CONV_ABSENT:.*]] = fir.convert %[[ABSENT]] : (!fir.box<i1>) -> !fir.box<none>
iall_test_8 = iall(a)
! CHECK:  %[[RESULT:.*]] = fir.call @_FortranAIAll8(%[[CONV_ARG]], %{{.*}}, %{{.*}}, %[[CONV_C0]], %[[CONV_ABSENT]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i64
! CHECK:  hlfir.assign %[[RESULT]] to %[[RES_DECL]]#0 : i64, !fir.ref<i64>
end function

! CHECK-LABEL: func @_QPiall_test_16(
! CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?xi128>>{{.*}}) -> i128 {
integer(16) function iall_test_16(a)
integer(16) :: a(:)
! CHECK-DAG:  %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK-DAG:  %[[ARG_DECL:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[SCOPE]] arg 1 {{.*}} : (!fir.box<!fir.array<?xi128>>, !fir.dscope) -> (!fir.box<!fir.array<?xi128>>, !fir.box<!fir.array<?xi128>>)
! CHECK-DAG:  %[[RES_ALLOCA:.*]] = fir.alloca i128
! CHECK-DAG:  %[[RES_DECL:.*]]:2 = hlfir.declare %[[RES_ALLOCA]] {{.*}} : (!fir.ref<i128>) -> (!fir.ref<i128>, !fir.ref<i128>)
! CHECK-DAG:  %[[ABSENT:.*]] = fir.absent !fir.box<i1>
! CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
! CHECK-DAG:  %[[CONV_ARG:.*]] = fir.convert %[[ARG_DECL]]#1 : (!fir.box<!fir.array<?xi128>>) -> !fir.box<none>
! CHECK-DAG:  %[[CONV_C0:.*]] = fir.convert %[[C0]] : (index) -> i32
! CHECK-DAG:  %[[CONV_ABSENT:.*]] = fir.convert %[[ABSENT]] : (!fir.box<i1>) -> !fir.box<none>
iall_test_16 = iall(a)
! CHECK:  %[[RESULT:.*]] = fir.call @_FortranAIAll16(%[[CONV_ARG]], %{{.*}}, %{{.*}}, %[[CONV_C0]], %[[CONV_ABSENT]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i128
! CHECK:  hlfir.assign %[[RESULT]] to %[[RES_DECL]]#0 : i128, !fir.ref<i128>
end function

! CHECK-LABEL: func @_QPiall_test2(
! CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?x?xi32>>{{.*}}, %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}) {
subroutine iall_test2(a,r)
integer :: a(:,:)
integer :: r(:)
! CHECK-DAG:  %[[BOX_ALLOCA:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK-DAG:  %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK-DAG:  %[[ARG0_DECL:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[SCOPE]] arg 1 {{.*}} : (!fir.box<!fir.array<?x?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?xi32>>, !fir.box<!fir.array<?x?xi32>>)
! CHECK-DAG:  %[[ARG1_DECL:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[SCOPE]] arg 2 {{.*}} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK-DAG:  %[[C2:.*]] = arith.constant 2 : i32
! CHECK-DAG:  %[[ABSENT:.*]] = fir.absent !fir.box<i1>
! CHECK-DAG:  %[[CONV_BOX:.*]] = fir.convert %[[BOX_ALLOCA]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK-DAG:  %[[CONV_ARG0:.*]] = fir.convert %[[ARG0_DECL]]#1 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
! CHECK-DAG:  %[[CONV_ABSENT:.*]] = fir.convert %[[ABSENT]] : (!fir.box<i1>) -> !fir.box<none>
r = iall(a,dim=2)
! CHECK:  fir.call @_FortranAIAllDim(%[[CONV_BOX]], %[[CONV_ARG0]], %[[C2]], %{{.*}}, %{{.*}}, %[[CONV_ABSENT]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>) -> ()
! CHECK:  %[[BOX_LOAD:.*]] = fir.load %[[BOX_ALLOCA]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:  %[[ADDR:.*]] = fir.box_addr %[[BOX_LOAD]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:  %[[TMP_DECL:.*]]:2 = hlfir.declare %[[ADDR]]({{.*}}) {uniq_name = ".tmp.intrinsic_result"} : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
! CHECK:  %[[EXPR:.*]] = hlfir.as_expr %[[TMP_DECL]]#0 move %{{.*}} : (!fir.box<!fir.array<?xi32>>, i1) -> !hlfir.expr<?xi32>
! CHECK:  hlfir.assign %[[EXPR]] to %[[ARG1_DECL]]#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
! CHECK:  hlfir.destroy %[[EXPR]] : !hlfir.expr<?xi32>
end subroutine

! CHECK-LABEL: func @_QPiall_test_optional(
! CHECK-SAME: %[[MASK:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>{{.*}}, %[[X:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}) -> i32 {
integer function iall_test_optional(mask, x)
integer :: x(:)
logical, optional :: mask(:)
! CHECK:  %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:  %[[RES_ALLOCA:.*]] = fir.alloca i32
! CHECK:  %[[RES_DECL:.*]]:2 = hlfir.declare %[[RES_ALLOCA]]
! CHECK:  %[[MASK_DECL:.*]]:2 = hlfir.declare %[[MASK]] dummy_scope %[[SCOPE]] arg 1 {fortran_attrs = #fir.var_attrs<optional>, {{.*}}}
! CHECK:  %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] dummy_scope %[[SCOPE]] arg 2 {{.*}}
iall_test_optional = iall(x, mask=mask)
! CHECK:  %[[CONV_X:.*]] = fir.convert %[[X_DECL]]#1 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
! CHECK:  %[[CONV_MASK:.*]] = fir.convert %[[MASK_DECL]]#1 : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<none>
! CHECK:  %[[RESULT:.*]] = fir.call @_FortranAIAll4(%[[CONV_X]], %{{.*}}, %{{.*}}, %{{.*}}, %[[CONV_MASK]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32
! CHECK:  hlfir.assign %[[RESULT]] to %[[RES_DECL]]#0 : i32, !fir.ref<i32>
end function

! CHECK-LABEL: func @_QPiall_test_optional_2(
! CHECK-SAME: %[[MASK:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>{{.*}}, %[[X:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}) -> i32 {
integer function iall_test_optional_2(mask, x)
integer :: x(:)
logical, pointer :: mask(:)
! CHECK:  %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:  %[[RES_ALLOCA:.*]] = fir.alloca i32
! CHECK:  %[[RES_DECL:.*]]:2 = hlfir.declare %[[RES_ALLOCA]]
! CHECK:  %[[MASK_DECL:.*]]:2 = hlfir.declare %[[MASK]] dummy_scope %[[SCOPE]] arg 1 {fortran_attrs = #fir.var_attrs<pointer>, {{.*}}}
! CHECK:  %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] dummy_scope %[[SCOPE]] arg 2 {{.*}}
iall_test_optional_2 = iall(x, mask=mask)
! CHECK:  %[[MASK_LOAD1:.*]] = fir.load %[[MASK_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>
! CHECK:  %[[MASK_ADDR:.*]] = fir.box_addr %[[MASK_LOAD1]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>) -> !fir.ptr<!fir.array<?x!fir.logical<4>>>
! CHECK:  %[[MASK_PTR_CONV:.*]] = fir.convert %[[MASK_ADDR]] : (!fir.ptr<!fir.array<?x!fir.logical<4>>>) -> i64
! CHECK:  %[[C0_I64:.*]] = arith.constant 0 : i64
! CHECK:  %[[IS_PRESENT:.*]] = arith.cmpi ne, %[[MASK_PTR_CONV]], %[[C0_I64]] : i64
! CHECK:  %[[MASK_LOAD2:.*]] = fir.load %[[MASK_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>
! CHECK:  %[[ABSENT:.*]] = fir.absent !fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>
! CHECK:  %[[SELECTED:.*]] = arith.select %[[IS_PRESENT]], %[[MASK_LOAD2]], %[[ABSENT]] : !fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>
! CHECK:  %[[CONV_X:.*]] = fir.convert %[[X_DECL]]#1 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
! CHECK:  %[[CONV_MASK:.*]] = fir.convert %[[SELECTED]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>) -> !fir.box<none>
! CHECK:  %[[RESULT:.*]] = fir.call @_FortranAIAll4(%[[CONV_X]], %{{.*}}, %{{.*}}, %{{.*}}, %[[CONV_MASK]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32
! CHECK:  hlfir.assign %[[RESULT]] to %[[RES_DECL]]#0 : i32, !fir.ref<i32>
end function

! CHECK-LABEL: func @_QPiall_test_optional_3(
! CHECK-SAME: %[[MASK:.*]]: !fir.ref<!fir.array<10x!fir.logical<4>>>{{.*}}, %[[X:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}) -> i32 {
integer function iall_test_optional_3(mask, x)
integer :: x(:)
logical, optional :: mask(10)
! CHECK:  %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:  %[[RES_ALLOCA:.*]] = fir.alloca i32
! CHECK:  %[[RES_DECL:.*]]:2 = hlfir.declare %[[RES_ALLOCA]]
! CHECK:  %[[MASK_DECL:.*]]:2 = hlfir.declare %[[MASK]]({{.*}}) dummy_scope %[[SCOPE]] arg 1 {fortran_attrs = #fir.var_attrs<optional>, {{.*}}}
! CHECK:  %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] dummy_scope %[[SCOPE]] arg 2 {{.*}}
iall_test_optional_3 = iall(x, mask=mask)
! CHECK:  %[[IS_PRESENT:.*]] = fir.is_present %[[MASK_DECL]]#0 : (!fir.ref<!fir.array<10x!fir.logical<4>>>) -> i1
! CHECK:  %[[EMBOX:.*]] = fir.embox %[[MASK_DECL]]#0(%{{.*}}) : (!fir.ref<!fir.array<10x!fir.logical<4>>>, !fir.shape<1>) -> !fir.box<!fir.array<10x!fir.logical<4>>>
! CHECK:  %[[ABSENT:.*]] = fir.absent !fir.box<!fir.array<10x!fir.logical<4>>>
! CHECK:  %[[SELECTED:.*]] = arith.select %[[IS_PRESENT]], %[[EMBOX]], %[[ABSENT]] : !fir.box<!fir.array<10x!fir.logical<4>>>
! CHECK:  %[[CONV_X:.*]] = fir.convert %[[X_DECL]]#1 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
! CHECK:  %[[CONV_MASK:.*]] = fir.convert %[[SELECTED]] : (!fir.box<!fir.array<10x!fir.logical<4>>>) -> !fir.box<none>
! CHECK:  %[[RESULT:.*]] = fir.call @_FortranAIAll4(%[[CONV_X]], %{{.*}}, %{{.*}}, %{{.*}}, %[[CONV_MASK]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32
! CHECK:  hlfir.assign %[[RESULT]] to %[[RES_DECL]]#0 : i32, !fir.ref<i32>
end function

! CHECK-LABEL: func @_QPiall_test_optional_4(
! CHECK-SAME: %[[X:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}, %[[USE_MASK:.*]]: !fir.ref<!fir.logical<4>>{{.*}}) -> i32 {
integer function iall_test_optional_4(x, use_mask)
! Test that local allocatable tracked in local variables
! are dealt as optional argument correctly.
integer :: x(:)
logical :: use_mask
logical, allocatable :: mask(:)
! CHECK:  %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:  %[[RES_ALLOCA:.*]] = fir.alloca i32
! CHECK:  %[[RES_DECL:.*]]:2 = hlfir.declare %[[RES_ALLOCA]]
! CHECK:  %[[MASK_ALLOCA:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>
! CHECK:  %[[MASK_DECL:.*]]:2 = hlfir.declare %[[MASK_ALLOCA]] {fortran_attrs = #fir.var_attrs<allocatable>, {{.*}}}
! CHECK:  %[[USE_MASK_DECL:.*]]:2 = hlfir.declare %[[USE_MASK]] dummy_scope %[[SCOPE]] arg 2 {{.*}}
! CHECK:  %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] dummy_scope %[[SCOPE]] arg 1 {{.*}}
if (use_mask) then
  allocate(mask(size(x, 1)))
  call set_mask(mask)
  ! CHECK: fir.call @_QPset_mask(%{{.*}}) fastmath<contract> : (!fir.ref<!fir.array<?x!fir.logical<4>>>) -> ()
end if
iall_test_optional_4 = iall(x, mask=mask)
! CHECK:  %[[MASK_LOAD1:.*]] = fir.load %[[MASK_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>
! CHECK:  %[[MASK_ADDR1:.*]] = fir.box_addr %[[MASK_LOAD1]] : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>) -> !fir.heap<!fir.array<?x!fir.logical<4>>>
! CHECK:  %[[MASK_PTR_CONV:.*]] = fir.convert %[[MASK_ADDR1]] : (!fir.heap<!fir.array<?x!fir.logical<4>>>) -> i64
! CHECK:  %[[C0_I64:.*]] = arith.constant 0 : i64
! CHECK:  %[[IS_ALLOCATED:.*]] = arith.cmpi ne, %[[MASK_PTR_CONV]], %[[C0_I64]] : i64
! CHECK:  %[[MASK_LOAD2:.*]] = fir.load %[[MASK_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>
! CHECK:  %[[MASK_ADDR2:.*]] = fir.box_addr %[[MASK_LOAD2]] : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>) -> !fir.heap<!fir.array<?x!fir.logical<4>>>
! CHECK:  %[[EMBOX:.*]] = fir.embox %[[MASK_ADDR2]](%{{.*}}) : (!fir.heap<!fir.array<?x!fir.logical<4>>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK:  %[[ABSENT:.*]] = fir.absent !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK:  %[[SELECTED:.*]] = arith.select %[[IS_ALLOCATED]], %[[EMBOX]], %[[ABSENT]] : !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK:  %[[CONV_X:.*]] = fir.convert %[[X_DECL]]#1 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
! CHECK:  %[[CONV_MASK:.*]] = fir.convert %[[SELECTED]] : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<none>
! CHECK:  %[[RESULT:.*]] = fir.call @_FortranAIAll4(%[[CONV_X]], %{{.*}}, %{{.*}}, %{{.*}}, %[[CONV_MASK]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32
! CHECK:  hlfir.assign %[[RESULT]] to %[[RES_DECL]]#0 : i32, !fir.ref<i32>
end function
