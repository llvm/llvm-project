! RUN: bbc -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! Test LOC intrinsic

! CHECK-LABEL: func.func @_QPloc_scalar() {
subroutine loc_scalar()
  integer(8) :: p
  integer :: x
  p = loc(x)
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ep
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2:[a-z0-9]*]]  {{.*}}Ex
! CHECK:  %[[VAL_4:.*]] = fir.embox %[[VAL_3]]#1 : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK:  %[[VAL_5:.*]] = fir.box_addr %[[VAL_4]] : (!fir.box<i32>) -> !fir.ref<i32>
! CHECK:  %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (!fir.ref<i32>) -> i64
! CHECK:  hlfir.assign %[[VAL_6]] to %[[VAL_1]]#0 : i64, !fir.ref<i64>
end

! CHECK-LABEL: func.func @_QPloc_char() {
subroutine loc_char()
  integer(8) :: p
  character(5) :: x = "abcde"
  p = loc(x)
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ep
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_2:[a-z0-9]*]] typeparams %[[VAL_3:[a-z0-9]*]]  {{.*}}Ex
! CHECK:  %[[VAL_5:.*]] = fir.embox %[[VAL_4]]#1 : (!fir.ref<!fir.char<1,5>>) -> !fir.box<!fir.char<1,5>>
! CHECK:  %[[VAL_6:.*]] = fir.box_addr %[[VAL_5]] : (!fir.box<!fir.char<1,5>>) -> !fir.ref<!fir.char<1,5>>
! CHECK:  %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.char<1,5>>) -> i64
! CHECK:  hlfir.assign %[[VAL_7]] to %[[VAL_1]]#0 : i64, !fir.ref<i64>
end

! CHECK-LABEL: func.func @_QPloc_substring() {
subroutine loc_substring()
  integer(8) :: p
  character(5) :: x = "abcde"
  p = loc(x(2:))
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ep
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_2:[a-z0-9]*]] typeparams %[[VAL_3:[a-z0-9]*]]  {{.*}}Ex
! CHECK:  %[[VAL_5:.*]] = arith.constant 2 : index
! CHECK:  %[[VAL_6:.*]] = arith.constant 5 : index
! CHECK:  %[[VAL_7:.*]] = arith.constant 4 : index
! CHECK:  %[[VAL_8:.*]] = hlfir.designate %[[VAL_4]]#0  substr %[[VAL_5]], %[[VAL_6]]  typeparams %[[VAL_7]] : (!fir.ref<!fir.char<1,5>>, index, index, index) -> !fir.ref<!fir.char<1,4>>
! CHECK:  %[[VAL_9:.*]] = fir.embox %[[VAL_8]] : (!fir.ref<!fir.char<1,4>>) -> !fir.box<!fir.char<1,4>>
! CHECK:  %[[VAL_10:.*]] = fir.box_addr %[[VAL_9]] : (!fir.box<!fir.char<1,4>>) -> !fir.ref<!fir.char<1,4>>
! CHECK:  %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (!fir.ref<!fir.char<1,4>>) -> i64
! CHECK:  hlfir.assign %[[VAL_11]] to %[[VAL_1]]#0 : i64, !fir.ref<i64>
end

! CHECK-LABEL: func.func @_QPloc_array() {
subroutine loc_array
  integer(8) :: p
  integer :: x(10)
  p = loc(x)
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ep
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_3:[a-z0-9]*]](%[[VAL_4:[a-z0-9]*]])  {{.*}}Ex
! CHECK:  %[[VAL_7:.*]] = fir.embox %[[VAL_5]]#1(%{{.*}}) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
! CHECK:  %[[VAL_8:.*]] = fir.box_addr %[[VAL_7]] : (!fir.box<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>>
! CHECK:  %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (!fir.ref<!fir.array<10xi32>>) -> i64
! CHECK:  hlfir.assign %[[VAL_9]] to %[[VAL_1]]#0 : i64, !fir.ref<i64>
end

! CHECK-LABEL: func.func @_QPloc_chararray() {
subroutine loc_chararray()
  integer(8) :: p
  character(5) :: x(2)
  p = loc(x)
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ep
! CHECK:  %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_4:[a-z0-9]*]](%[[VAL_5:[a-z0-9]*]]) typeparams %[[VAL_2:[a-z0-9]*]]  {{.*}}Ex
! CHECK:  %[[VAL_8:.*]] = fir.embox %[[VAL_6]]#1(%{{.*}}) : (!fir.ref<!fir.array<2x!fir.char<1,5>>>, !fir.shape<1>) -> !fir.box<!fir.array<2x!fir.char<1,5>>>
! CHECK:  %[[VAL_9:.*]] = fir.box_addr %[[VAL_8]] : (!fir.box<!fir.array<2x!fir.char<1,5>>>) -> !fir.ref<!fir.array<2x!fir.char<1,5>>>
! CHECK:  %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (!fir.ref<!fir.array<2x!fir.char<1,5>>>) -> i64
! CHECK:  hlfir.assign %[[VAL_10]] to %[[VAL_1]]#0 : i64, !fir.ref<i64>
end

! CHECK-LABEL: func.func @_QPloc_arrayelement() {
subroutine loc_arrayelement()
  integer(8) :: p
  integer :: x(10)
  p = loc(x(7))
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ep
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_3:[a-z0-9]*]](%[[VAL_4:[a-z0-9]*]])  {{.*}}Ex
! CHECK:  %[[VAL_6:.*]] = arith.constant 7 : index
! CHECK:  %[[VAL_7:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_6]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
! CHECK:  %[[VAL_8:.*]] = fir.embox %[[VAL_7]] : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK:  %[[VAL_9:.*]] = fir.box_addr %[[VAL_8]] : (!fir.box<i32>) -> !fir.ref<i32>
! CHECK:  %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (!fir.ref<i32>) -> i64
! CHECK:  hlfir.assign %[[VAL_10]] to %[[VAL_1]]#0 : i64, !fir.ref<i64>
end

! CHECK-LABEL: func.func @_QPloc_arraysection(
! CHECK-SAME: %[[arg:.*]]: !fir.ref<i32> {{.*}}) {
subroutine loc_arraysection(i)
  integer(8) :: p
  integer :: i
  real :: x(11)
  p = loc(x(i:))
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ei
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2:[a-z0-9]*]]  {{.*}}Ep
! CHECK:  %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_5:[a-z0-9]*]](%[[VAL_6:[a-z0-9]*]])  {{.*}}Ex
! CHECK:  %[[VAL_19:.*]] = hlfir.designate %[[VAL_7]]#0 (%{{.*}}:%{{.*}}:%{{.*}})  shape %{{.*}} : (!fir.ref<!fir.array<11xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK:  %[[VAL_20:.*]] = fir.box_addr %[[VAL_19]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:  %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (!fir.ref<!fir.array<?xf32>>) -> i64
! CHECK:  hlfir.assign %[[VAL_21]] to %[[VAL_3]]#0 : i64, !fir.ref<i64>
end

! CHECK-LABEL: func.func @_QPloc_non_save_pointer_scalar() {
subroutine loc_non_save_pointer_scalar()
  integer(8) :: p
  real, pointer :: x
  real, target :: t
  x => t
  p = loc(x)
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ep
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2:[a-z0-9]*]]  {{.*}}Et
! CHECK:  %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_4:[a-z0-9]*]]  {{.*}}Ex
! CHECK:  %[[VAL_8:.*]] = fir.embox %[[VAL_3]]#1 : (!fir.ref<f32>) -> !fir.box<!fir.ptr<f32>>
! CHECK:  fir.store %[[VAL_8]] to %[[VAL_7]]#1 : !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK:  %[[VAL_9:.*]] = fir.load %[[VAL_7]]#1 : !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK:  %[[VAL_10:.*]] = fir.box_addr %[[VAL_9]] : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
! CHECK:  %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (!fir.ptr<f32>) -> i64
! CHECK:  hlfir.assign %[[VAL_11]] to %[[VAL_1]]#0 : i64, !fir.ref<i64>
end

! CHECK-LABEL: func.func @_QPloc_save_pointer_scalar() {
subroutine loc_save_pointer_scalar()
  integer :: p
  real, pointer, save :: x
  p = loc(x)
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ep
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2:[a-z0-9]*]]  {{.*}}Ex
! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_3]]#1 : !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK:  %[[VAL_5:.*]] = fir.box_addr %[[VAL_4]] : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
! CHECK:  %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (!fir.ptr<f32>) -> i64
! CHECK:  %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i64) -> i32
! CHECK:  hlfir.assign %[[VAL_7]] to %[[VAL_1]]#0 : i32, !fir.ref<i32>
end

! CHECK-LABEL: func.func @_QPloc_derived_type() {
subroutine loc_derived_type
  integer(8) :: p
  type dt
    integer :: i
  end type
  type(dt) :: xdt
  p = loc(xdt)
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ep
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2:[a-z0-9]*]]  {{.*}}Exdt
! CHECK:  %[[VAL_4:.*]] = fir.embox %[[VAL_3]]#1 : (!fir.ref<!fir.type<_QFloc_derived_typeTdt{i:i32}>>) -> !fir.box<!fir.type<_QFloc_derived_typeTdt{i:i32}>>
! CHECK:  %[[VAL_5:.*]] = fir.box_addr %[[VAL_4]] : (!fir.box<!fir.type<_QFloc_derived_typeTdt{i:i32}>>) -> !fir.ref<!fir.type<_QFloc_derived_typeTdt{i:i32}>>
! CHECK:  %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (!fir.ref<!fir.type<_QFloc_derived_typeTdt{i:i32}>>) -> i64
! CHECK:  hlfir.assign %[[VAL_6]] to %[[VAL_1]]#0 : i64, !fir.ref<i64>
end

! CHECK-LABEL: func.func @_QPloc_pointer_array() {
subroutine loc_pointer_array
  integer(8) :: p
  integer, pointer :: x(:)
  p = loc(x)
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ep
! CHECK:  %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_2:[a-z0-9]*]]  {{.*}}Ex
! CHECK:  %[[VAL_8:.*]] = fir.load %[[VAL_7]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:  %[[VAL_9:.*]] = fir.box_addr %[[VAL_8]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
! CHECK:  %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (!fir.ptr<!fir.array<?xi32>>) -> i64
! CHECK:  hlfir.assign %[[VAL_10]] to %[[VAL_1]]#0 : i64, !fir.ref<i64>
end

! CHECK-LABEL: func.func @_QPloc_allocatable_array() {
subroutine loc_allocatable_array
  integer(8) :: p
  integer, allocatable :: x(:)
  p = loc(x)
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ep
! CHECK:  %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_2:[a-z0-9]*]]  {{.*}}Ex
! CHECK:  %[[VAL_8:.*]] = fir.load %[[VAL_7]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:  %[[VAL_9:.*]] = fir.box_addr %[[VAL_8]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:  %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (!fir.heap<!fir.array<?xi32>>) -> i64
! CHECK:  hlfir.assign %[[VAL_10]] to %[[VAL_1]]#0 : i64, !fir.ref<i64>
end

! CHECK-LABEL: func.func @_QPtest_external() {
subroutine test_external()
  integer(8) :: p
  integer, external :: f
  p = loc(x=f)
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ep
! CHECK:  %[[VAL_2:.*]] = fir.address_of(@_QPf) : () -> i32
! CHECK:  %[[VAL_3:.*]] = fir.emboxproc %[[VAL_2]] : (() -> i32) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_4:.*]] = fir.box_addr %[[VAL_3]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:  %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (() -> ()) -> i64
! CHECK:  hlfir.assign %[[VAL_5]] to %[[VAL_1]]#0 : i64, !fir.ref<i64>
end

! CHECK-LABEL: func.func @_QPtest_proc() {
subroutine test_proc()
  integer(8) :: p
  procedure() :: g
  p = loc(x=g)
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ep
! CHECK:  %[[VAL_2:.*]] = fir.address_of(@_QPg) : () -> ()
! CHECK:  %[[VAL_3:.*]] = fir.emboxproc %[[VAL_2]] : (() -> ()) -> !fir.boxproc<() -> ()>
! CHECK:  %[[VAL_4:.*]] = fir.box_addr %[[VAL_3]] : (!fir.boxproc<() -> ()>) -> (() -> ())
! CHECK:  %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (() -> ()) -> i64
! CHECK:  hlfir.assign %[[VAL_5]] to %[[VAL_1]]#0 : i64, !fir.ref<i64>
end

! CHECK-LABEL:   func.func @_QPtest_assumed_shape_optional(
subroutine test_assumed_shape_optional(x)
  integer(8) :: p
  real, optional :: x(:)
  p = loc(x=x)
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1:[a-z0-9]*]]  {{.*}}Ep
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ex
! CHECK:  %[[VAL_4:.*]] = fir.is_present %[[VAL_3]]#1 : (!fir.box<!fir.array<?xf32>>) -> i1
! CHECK:  %[[VAL_5:.*]] = fir.if %[[VAL_4]] -> (i64) {
! CHECK:    %[[VAL_6:.*]] = fir.box_addr %[[VAL_3]]#1 : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:    %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.array<?xf32>>) -> i64
! CHECK:    fir.result %[[VAL_7]] : i64
! CHECK:  } else {
! CHECK:    %[[VAL_8:.*]] = arith.constant 0 : i64
! CHECK:    fir.result %[[VAL_8]] : i64
! CHECK:  }
! CHECK:  hlfir.assign %[[VAL_5]] to %[[VAL_2]]#0 : i64, !fir.ref<i64>
end subroutine
