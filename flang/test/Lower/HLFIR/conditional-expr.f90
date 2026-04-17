! Test lowering of conditional expressions (Fortran 2023)
! RUN: %flang_fc1 -emit-hlfir -funsigned -o - %s 2>&1 | FileCheck %s

! CHECK-LABEL: func.func @_QPtest_scalar_integer(
! CHECK-SAME:    %[[FLAG:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "flag"},
! CHECK-SAME:    %[[X:.*]]: !fir.ref<i32> {fir.bindc_name = "x"},
! CHECK-SAME:    %[[Y:.*]]: !fir.ref<i32> {fir.bindc_name = "y"})
subroutine test_scalar_integer(flag, x, y)
  logical :: flag
  integer :: x, y, result
  ! CHECK-DAG: %[[FLAG_DECL:.*]]:2 = hlfir.declare %[[FLAG]]
  ! CHECK-DAG: %[[X_DECL:.*]]:2 = hlfir.declare %[[X]]
  ! CHECK-DAG: %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y]]

  result = (flag ? x : y)
  ! CHECK: %[[FLAG_LOAD:.*]] = fir.load %[[FLAG_DECL]]#0
  ! CHECK: %[[FLAG_CONV:.*]] = fir.convert %[[FLAG_LOAD]] : (!fir.logical<4>) -> i1
  ! CHECK: %[[RESULT:.*]] = fir.if %[[FLAG_CONV]] -> (i32) {
  ! CHECK:   %[[X_LOAD:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<i32>
  ! CHECK:   fir.result %[[X_LOAD]] : i32
  ! CHECK: } else {
  ! CHECK:   %[[Y_LOAD:.*]] = fir.load %[[Y_DECL]]#0 : !fir.ref<i32>
  ! CHECK:   fir.result %[[Y_LOAD]] : i32
  ! CHECK: }
  ! CHECK: hlfir.assign %[[RESULT]] to %{{.*}} : i32, !fir.ref<i32>
end subroutine

! CHECK-LABEL: func.func @_QPtest_scalar_real(
subroutine test_scalar_real(flag, x, y)
  logical :: flag
  real :: x, y, result
  result = (flag ? x : y)
  ! CHECK: %[[RESULT:.*]] = fir.if {{.*}} -> (f32) {
  ! CHECK:   fir.result {{.*}} : f32
  ! CHECK: } else {
  ! CHECK:   fir.result {{.*}} : f32
  ! CHECK: }
end subroutine

! CHECK-LABEL: func.func @_QPtest_scalar_complex(
subroutine test_scalar_complex(flag, x, y)
  logical :: flag
  complex :: x, y, result
  result = (flag ? x : y)
  ! CHECK: %[[RESULT:.*]] = fir.if {{.*}} -> (complex<f32>) {
  ! CHECK:   fir.result {{.*}} : complex<f32>
  ! CHECK: } else {
  ! CHECK:   fir.result {{.*}} : complex<f32>
  ! CHECK: }
end subroutine

! CHECK-LABEL: func.func @_QPtest_scalar_logical(
subroutine test_scalar_logical(flag, x, y)
  logical :: flag, x, y, result
  result = (flag ? x : y)
  ! CHECK: %[[RESULT:.*]] = fir.if {{.*}} -> (!fir.logical<4>) {
  ! CHECK:   fir.result {{.*}} : !fir.logical<4>
  ! CHECK: } else {
  ! CHECK:   fir.result {{.*}} : !fir.logical<4>
  ! CHECK: }
end subroutine

! CHECK-LABEL: func.func @_QPtest_scalar_unsigned(
subroutine test_scalar_unsigned(flag, x, y)
  logical :: flag
  unsigned :: x, y, result
  result = (flag ? x : y)
  ! CHECK: %[[RESULT:.*]] = fir.if {{.*}} -> (ui32) {
  ! CHECK:   fir.result {{.*}} : ui32
  ! CHECK: } else {
  ! CHECK:   fir.result {{.*}} : ui32
  ! CHECK: }
end subroutine

! CHECK-LABEL: func.func @_QPtest_logical_literal(
subroutine test_logical_literal(flag)
  logical :: flag, result
  result = (flag ? .true. : .false.)
  ! CHECK: %[[RESULT:.*]] = fir.if {{.*}} -> (!fir.logical<4>) {
  ! CHECK:   %[[TRUE:.*]] = arith.constant true
  ! CHECK:   %[[CONV:.*]] = fir.convert %[[TRUE]] : (i1) -> !fir.logical<4>
  ! CHECK:   fir.result %[[CONV]] : !fir.logical<4>
  ! CHECK: } else {
  ! CHECK:   %[[FALSE:.*]] = arith.constant false
  ! CHECK:   %[[CONV:.*]] = fir.convert %[[FALSE]] : (i1) -> !fir.logical<4>
  ! CHECK:   fir.result %[[CONV]] : !fir.logical<4>
  ! CHECK: }
end subroutine

! CHECK-LABEL: func.func @_QPtest_multi_branch(
subroutine test_multi_branch(x)
  integer :: x, result
  ! Multi-branch: x > 10 ? 100 : x > 5 ? 50 : 0
  result = (x > 10 ? 100 : x > 5 ? 50 : 0)
  ! Outer condition: x > 10
  ! CHECK: arith.cmpi sgt
  ! CHECK: %[[OUTER:.*]] = fir.if {{.*}} -> (i32) {
  ! CHECK:   fir.result {{.*}} : i32
  ! CHECK: } else {
  ! Inner conditional: x > 5 ? 50 : 0
  ! CHECK:   arith.cmpi sgt
  ! CHECK:   %[[INNER:.*]] = fir.if {{.*}} -> (i32) {
  ! CHECK:     fir.result {{.*}} : i32
  ! CHECK:   } else {
  ! CHECK:     fir.result {{.*}} : i32
  ! CHECK:   }
  ! CHECK:   fir.result %[[INNER]] : i32
  ! CHECK: }
end subroutine

! CHECK-LABEL: func.func @_QPtest_char_constant_len(
subroutine test_char_constant_len(flag)
  logical :: flag
  character(len=5) :: str1, str2, result
  str1 = "HELLO"
  str2 = "WORLD"
  result = (flag ? str1 : str2)
  ! Constant length: use scalar temp path.
  ! CHECK: %[[TEMP:.*]] = fir.alloca !fir.char<1,5> {bindc_name = ".cond.scalar"
  ! CHECK: %[[TEMP_DECL:.*]]:2 = hlfir.declare %[[TEMP]] typeparams {{.*}} {uniq_name = ".cond.result"}
  ! CHECK: fir.if
  ! CHECK:   hlfir.assign {{.*}} to %[[TEMP_DECL]]#0
  ! CHECK: } else {
  ! CHECK:   hlfir.assign {{.*}} to %[[TEMP_DECL]]#0
  ! CHECK: }
end subroutine

! CHECK-LABEL: func.func @_QPtest_char_deferred_len(
subroutine test_char_deferred_len(flag)
  logical :: flag
  character(len=:), allocatable :: str1, str2, result
  str1 = "SHORT"
  str2 = "A MUCH LONGER STRING"
  ! Result length comes from selected branch
  result = (flag ? str1 : str2)
  ! CHECK-DAG: %[[BOX_ALLOC:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>> {bindc_name = ".cond.char"
  ! CHECK-DAG: %[[UNALLOC:.*]] = fir.zero_bits !fir.heap<!fir.char<1,?>>
  ! CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  ! CHECK: %[[BOX:.*]] = fir.embox %[[UNALLOC]] typeparams %[[C0]]
  ! CHECK: fir.store %[[BOX]] to %{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
  ! CHECK: %[[BOX_DECL:.*]]:2 = hlfir.declare %[[BOX_ALLOC]] {uniq_name = ".cond.result"}
  ! CHECK: fir.if
  ! CHECK:   hlfir.assign {{.*}} to %[[BOX_DECL]]#0 realloc temporary_lhs
  ! CHECK: } else {
  ! CHECK:   hlfir.assign {{.*}} to %[[BOX_DECL]]#0 realloc temporary_lhs
  ! CHECK: }
end subroutine

! CHECK-LABEL: func.func @_QPtest_array(
subroutine test_array(flag)
  logical :: flag
  integer :: arr1(10), arr2(10), result(10)
  arr1 = 1
  arr2 = 2
  result = (flag ? arr1 : arr2)
  ! CHECK: %[[BOX_ALLOC:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<{{.*}}xi32>>> {bindc_name = ".cond.array"
  ! CHECK: %[[UNALLOC:.*]] = fir.zero_bits !fir.heap<!fir.array<{{.*}}xi32>>
  ! CHECK: %[[SHAPE:.*]] = fir.shape
  ! CHECK: %[[BOX:.*]] = fir.embox %[[UNALLOC]](%[[SHAPE]])
  ! CHECK: fir.store %[[BOX]] to %[[BOX_ALLOC]]
  ! CHECK: %[[BOX_DECL:.*]]:2 = hlfir.declare %[[BOX_ALLOC]] {uniq_name = ".cond.result"}
  ! CHECK: fir.if
  ! CHECK:   hlfir.assign {{.*}} to %[[BOX_DECL]]#0 realloc temporary_lhs
  ! CHECK: } else {
  ! CHECK:   hlfir.assign {{.*}} to %[[BOX_DECL]]#0 realloc temporary_lhs
  ! CHECK: }
end subroutine

! CHECK-LABEL: func.func @_QPtest_derived_type(
subroutine test_derived_type(flag)
  type :: point
    real :: x, y
  end type
  logical :: flag
  type(point) :: p1, p2, result
  p1 = point(1.0, 2.0)
  p2 = point(3.0, 4.0)
  result = (flag ? p1 : p2)
  ! CHECK: %[[TEMP:.*]] = fir.alloca !fir.type<_QFtest_derived_typeTpoint{x:f32,y:f32}> {bindc_name = ".cond.scalar"
  ! CHECK: %[[TEMP_DECL:.*]]:2 = hlfir.declare %[[TEMP]] {uniq_name = ".cond.result"}
  ! CHECK: fir.if
  ! CHECK:   hlfir.assign {{.*}} to %[[TEMP_DECL]]#0 : !fir.ref<!fir.type<_QFtest_derived_typeTpoint{x:f32,y:f32}>>, !fir.ref<!fir.type<_QFtest_derived_typeTpoint{x:f32,y:f32}>>
  ! CHECK: } else {
  ! CHECK:   hlfir.assign {{.*}} to %[[TEMP_DECL]]#0 : !fir.ref<!fir.type<_QFtest_derived_typeTpoint{x:f32,y:f32}>>, !fir.ref<!fir.type<_QFtest_derived_typeTpoint{x:f32,y:f32}>>
  ! CHECK: }
end subroutine

! CHECK-LABEL: func.func @_QPtest_nested_conditionals(
subroutine test_nested_conditionals(flag1, flag2, x, y, z)
  logical :: flag1, flag2
  integer :: x, y, z, result
  ! Nested: flag1 ? (flag2 ? x : y) : z
  result = (flag1 ? (flag2 ? x : y) : z)
  ! Outer conditional
  ! CHECK: %[[OUTER:.*]] = fir.if {{%.*}} -> (i32) {
  ! Inner conditional
  ! CHECK:   %[[INNER:.*]] = fir.if {{%.*}} -> (i32) {
  ! CHECK:     fir.result {{.*}} : i32
  ! CHECK:   } else {
  ! CHECK:     fir.result {{.*}} : i32
  ! CHECK:   }
  ! CHECK:   fir.result %[[INNER]] : i32
  ! CHECK: } else {
  ! CHECK:   fir.result {{.*}} : i32
  ! CHECK: }
end subroutine

! CHECK-LABEL: func.func @_QPtest_in_expression(
subroutine test_in_expression(flag, x, y)
  logical :: flag
  integer :: x, y, z
  ! Conditional in larger expression: (flag ? x : y) + 10
  z = (flag ? x : y) + 10
  ! CHECK: %[[COND_RESULT:.*]] = fir.if {{.*}} -> (i32) {
  ! CHECK:   fir.result {{.*}} : i32
  ! CHECK: } else {
  ! CHECK:   fir.result {{.*}} : i32
  ! CHECK: }
  ! CHECK: %[[C10:.*]] = arith.constant 10
  ! CHECK: %[[SUM:.*]] = arith.addi %[[COND_RESULT]], %[[C10]]
  ! CHECK: hlfir.assign %[[SUM]]
end subroutine

! CHECK-LABEL: func.func @_QPtest_assumed_length_char(
subroutine test_assumed_length_char(flag, str1, str2)
  logical :: flag
  character(len=*) :: str1, str2
  character(len=100) :: result
  result = (flag ? str1 : str2)
  ! Deferred length path since len=* is not constant
  ! CHECK: %[[BOX_ALLOC:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>> {bindc_name = ".cond.char"
  ! CHECK: fir.if
  ! CHECK:   hlfir.assign {{.*}} to {{.*}} realloc temporary_lhs
  ! CHECK: } else {
  ! CHECK:   hlfir.assign {{.*}} to {{.*}} realloc temporary_lhs
  ! CHECK: }
end subroutine

! CHECK-LABEL: func.func @_QPtest_different_kinds(
subroutine test_different_kinds(flag)
  logical :: flag
  integer(kind=4) :: i4_1, i4_2, i4_result
  integer(kind=8) :: i8_1, i8_2, i8_result

  i4_1 = 1
  i4_2 = 2
  i4_result = (flag ? i4_1 : i4_2)
  ! CHECK: %{{.*}} = fir.if {{.*}} -> (i32) {
  ! CHECK:   fir.result {{.*}} : i32
  ! CHECK: } else {
  ! CHECK:   fir.result {{.*}} : i32
  ! CHECK: }

  i8_1 = 3
  i8_2 = 4
  i8_result = (flag ? i8_1 : i8_2)
  ! CHECK: %{{.*}} = fir.if {{.*}} -> (i64) {
  ! CHECK:   fir.result {{.*}} : i64
  ! CHECK: } else {
  ! CHECK:   fir.result {{.*}} : i64
  ! CHECK: }
end subroutine

! CHECK-LABEL: func.func @_QPtest_array_section(
subroutine test_array_section(flag)
  logical :: flag
  integer :: arr1(20), arr2(20), result(10)
  result = (flag ? arr1(1:10) : arr2(11:20))
  ! CHECK: %[[BOX_ALLOC:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<{{.*}}xi32>>> {bindc_name = ".cond.array"
  ! CHECK: fir.if
  ! CHECK:   hlfir.assign {{.*}} to {{.*}} realloc temporary_lhs
  ! CHECK: } else {
  ! CHECK:   hlfir.assign {{.*}} to {{.*}} realloc temporary_lhs
  ! CHECK: }
end subroutine

! CHECK-LABEL: func.func @_QPtest_noncontiguous_section(
subroutine test_noncontiguous_section(flag)
  logical :: flag
  integer :: arr1(20), arr2(20), result(5)
  ! Non-contiguous stride-2 sections: result must be contiguous.
  result = (flag ? arr1(1:10:2) : arr2(2:10:2))
  ! CHECK: %[[BOX_ALLOC:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<{{.*}}xi32>>> {bindc_name = ".cond.array"
  ! CHECK: fir.if
  ! CHECK:   hlfir.assign {{.*}} to {{.*}} realloc temporary_lhs
  ! CHECK: } else {
  ! CHECK:   hlfir.assign {{.*}} to {{.*}} realloc temporary_lhs
  ! CHECK: }
end subroutine
