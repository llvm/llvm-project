! Test remapping of component references in data clauses.
! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

module mhdata_types
  type t_scalar
      integer :: x
      real :: y
  end type
  type t_array
      integer :: x
      real :: y(10)
  end type
  type t_character
      integer :: x
      character(5) :: y
  end type
  type t_pointer
      integer :: x
      real, pointer :: y(:)
  end type
  type t_nested
    type(t_array) :: comp(100)
  end type
end module

subroutine test_scalar_comp(obj)
  use mhdata_types, only : t_scalar
  type(t_scalar) :: obj
  !$acc host_data use_device(obj%y)
  call foo_scalar(obj%y)
  !$acc end host_data
end subroutine

subroutine test_array_comp(obj)
  use mhdata_types, only : t_array
  type(t_array) :: obj
  !$acc host_data use_device(obj%y)
  call foo_array(obj%y)
  !$acc end host_data
end subroutine

subroutine test_character_comp(obj)
  use mhdata_types, only : t_character
  type(t_character) :: obj
  !$acc host_data use_device(obj%y)
  call foo_character(obj%y)
  !$acc end host_data
end subroutine

subroutine test_pointer_comp(obj)
  use mhdata_types, only : t_pointer
  type(t_pointer) :: obj
  interface
    subroutine foo_pointer(x)
      real, pointer :: x(:)
    end subroutine
  end interface
  !$acc host_data use_device(obj%y)
  call foo_pointer(obj%y)
  !$acc end host_data
end subroutine

subroutine test_nested_comp(obj)
  use mhdata_types, only : t_nested
  type(t_nested) :: obj(:)
  !$acc host_data use_device(obj(10)%comp(2)%y(4))
  call foo_nested(obj(10)%comp(2)%y(4))
  !$acc end host_data
end subroutine

! CHECK-LABEL:   func.func @_QPtest_scalar_comp(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.type<_QMmhdata_typesTt_scalar{x:i32,y:f32}>> {fir.bindc_name = "obj"}) {
! CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[DUMMY_SCOPE_0]] arg 1 {uniq_name = "_QFtest_scalar_compEobj"} : (!fir.ref<!fir.type<_QMmhdata_typesTt_scalar{x:i32,y:f32}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QMmhdata_typesTt_scalar{x:i32,y:f32}>>, !fir.ref<!fir.type<_QMmhdata_typesTt_scalar{x:i32,y:f32}>>)
! CHECK:           %[[DESIGNATE_0:.*]] = hlfir.designate %[[DECLARE_0]]#0{"y"}   : (!fir.ref<!fir.type<_QMmhdata_typesTt_scalar{x:i32,y:f32}>>) -> !fir.ref<f32>
! CHECK:           %[[USE_DEVICE_0:.*]] = acc.use_device varPtr(%[[DESIGNATE_0]] : !fir.ref<f32>) -> !fir.ref<f32> {name = "obj%y"}
! CHECK:           acc.host_data dataOperands(%[[USE_DEVICE_0]] : !fir.ref<f32>) {
! CHECK:             %[[DECLARE_1:.*]]:2 = hlfir.declare %[[USE_DEVICE_0]] {uniq_name = "obj%y"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:             fir.call @_QPfoo_scalar(%[[DECLARE_1]]#0) fastmath<contract> : (!fir.ref<f32>) -> ()
! CHECK:             acc.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_array_comp(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.type<_QMmhdata_typesTt_array{x:i32,y:!fir.array<10xf32>}>> {fir.bindc_name = "obj"}) {
! CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[DUMMY_SCOPE_0]] arg 1 {uniq_name = "_QFtest_array_compEobj"} : (!fir.ref<!fir.type<_QMmhdata_typesTt_array{x:i32,y:!fir.array<10xf32>}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QMmhdata_typesTt_array{x:i32,y:!fir.array<10xf32>}>>, !fir.ref<!fir.type<_QMmhdata_typesTt_array{x:i32,y:!fir.array<10xf32>}>>)
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 10 : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_0]] : (index) -> !fir.shape<1>
! CHECK:           %[[DESIGNATE_0:.*]] = hlfir.designate %[[DECLARE_0]]#0{"y"}   shape %[[SHAPE_0]] : (!fir.ref<!fir.type<_QMmhdata_typesTt_array{x:i32,y:!fir.array<10xf32>}>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
! CHECK:           %[[USE_DEVICE_0:.*]] = acc.use_device varPtr(%[[DESIGNATE_0]] : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {name = "obj%y"}
! CHECK:           acc.host_data dataOperands(%[[USE_DEVICE_0]] : !fir.ref<!fir.array<10xf32>>) {
! CHECK:             %[[DECLARE_1:.*]]:2 = hlfir.declare %[[USE_DEVICE_0]](%[[SHAPE_0]]) {uniq_name = "obj%y"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
! CHECK:             fir.call @_QPfoo_array(%[[DECLARE_1]]#0) fastmath<contract> : (!fir.ref<!fir.array<10xf32>>) -> ()
! CHECK:             acc.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_character_comp(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.type<_QMmhdata_typesTt_character{x:i32,y:!fir.char<1,5>}>> {fir.bindc_name = "obj"}) {
! CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[DUMMY_SCOPE_0]] arg 1 {uniq_name = "_QFtest_character_compEobj"} : (!fir.ref<!fir.type<_QMmhdata_typesTt_character{x:i32,y:!fir.char<1,5>}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QMmhdata_typesTt_character{x:i32,y:!fir.char<1,5>}>>, !fir.ref<!fir.type<_QMmhdata_typesTt_character{x:i32,y:!fir.char<1,5>}>>)
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 5 : index
! CHECK:           %[[DESIGNATE_0:.*]] = hlfir.designate %[[DECLARE_0]]#0{"y"}   typeparams %[[CONSTANT_0]] : (!fir.ref<!fir.type<_QMmhdata_typesTt_character{x:i32,y:!fir.char<1,5>}>>, index) -> !fir.ref<!fir.char<1,5>>
! CHECK:           %[[USE_DEVICE_0:.*]] = acc.use_device varPtr(%[[DESIGNATE_0]] : !fir.ref<!fir.char<1,5>>) -> !fir.ref<!fir.char<1,5>> {name = "obj%y"}
! CHECK:           acc.host_data dataOperands(%[[USE_DEVICE_0]] : !fir.ref<!fir.char<1,5>>) {
! CHECK:             %[[DECLARE_1:.*]]:2 = hlfir.declare %[[USE_DEVICE_0]] typeparams %[[CONSTANT_0]] {uniq_name = "obj%y"} : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
! CHECK:             %[[EMBOXCHAR_0:.*]] = fir.emboxchar %[[DECLARE_1]]#0, %[[CONSTANT_0]] : (!fir.ref<!fir.char<1,5>>, index) -> !fir.boxchar<1>
! CHECK:             fir.call @_QPfoo_character(%[[EMBOXCHAR_0]]) fastmath<contract> : (!fir.boxchar<1>) -> ()
! CHECK:             acc.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_pointer_comp(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.type<_QMmhdata_typesTt_pointer{x:i32,y:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>> {fir.bindc_name = "obj"}) {
! CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[DUMMY_SCOPE_0]] arg 1 {uniq_name = "_QFtest_pointer_compEobj"} : (!fir.ref<!fir.type<_QMmhdata_typesTt_pointer{x:i32,y:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QMmhdata_typesTt_pointer{x:i32,y:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>, !fir.ref<!fir.type<_QMmhdata_typesTt_pointer{x:i32,y:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>)
! CHECK:           %[[DESIGNATE_0:.*]] = hlfir.designate %[[DECLARE_0]]#0{"y"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMmhdata_typesTt_pointer{x:i32,y:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:           %[[USE_DEVICE_0:.*]] = acc.use_device varPtr(%[[DESIGNATE_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {name = "obj%y"}
! CHECK:           acc.host_data dataOperands(%[[USE_DEVICE_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) {
! CHECK:             %[[DECLARE_1:.*]]:2 = hlfir.declare %[[USE_DEVICE_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "obj%y"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
! CHECK:             fir.call @_QPfoo_pointer(%[[DECLARE_1]]#0) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> ()
! CHECK:             acc.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest_nested_comp(
! CHECK-SAME:      %[[ARG0:.*]]: !fir.box<!fir.array<?x!fir.type<_QMmhdata_typesTt_nested{comp:!fir.array<100x!fir.type<_QMmhdata_typesTt_array{x:i32,y:!fir.array<10xf32>}>>}>>> {fir.bindc_name = "obj"}) {
! CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[DUMMY_SCOPE_0]] arg 1 {uniq_name = "_QFtest_nested_compEobj"} : (!fir.box<!fir.array<?x!fir.type<_QMmhdata_typesTt_nested{comp:!fir.array<100x!fir.type<_QMmhdata_typesTt_array{x:i32,y:!fir.array<10xf32>}>>}>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.type<_QMmhdata_typesTt_nested{comp:!fir.array<100x!fir.type<_QMmhdata_typesTt_array{x:i32,y:!fir.array<10xf32>}>>}>>>, !fir.box<!fir.array<?x!fir.type<_QMmhdata_typesTt_nested{comp:!fir.array<100x!fir.type<_QMmhdata_typesTt_array{x:i32,y:!fir.array<10xf32>}>>}>>>)
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 10 : index
! CHECK:           %[[DESIGNATE_0:.*]] = hlfir.designate %[[DECLARE_0]]#0 (%[[CONSTANT_0]])  : (!fir.box<!fir.array<?x!fir.type<_QMmhdata_typesTt_nested{comp:!fir.array<100x!fir.type<_QMmhdata_typesTt_array{x:i32,y:!fir.array<10xf32>}>>}>>>, index) -> !fir.ref<!fir.type<_QMmhdata_typesTt_nested{comp:!fir.array<100x!fir.type<_QMmhdata_typesTt_array{x:i32,y:!fir.array<10xf32>}>>}>>
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 100 : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_1]] : (index) -> !fir.shape<1>
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 2 : index
! CHECK:           %[[DESIGNATE_1:.*]] = hlfir.designate %[[DESIGNATE_0]]{"comp"} <%[[SHAPE_0]]> (%[[CONSTANT_2]])  : (!fir.ref<!fir.type<_QMmhdata_typesTt_nested{comp:!fir.array<100x!fir.type<_QMmhdata_typesTt_array{x:i32,y:!fir.array<10xf32>}>>}>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QMmhdata_typesTt_array{x:i32,y:!fir.array<10xf32>}>>
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 10 : index
! CHECK:           %[[SHAPE_1:.*]] = fir.shape %[[CONSTANT_3]] : (index) -> !fir.shape<1>
! CHECK:           %[[DESIGNATE_2:.*]] = hlfir.designate %[[DESIGNATE_1]]{"y"}   shape %[[SHAPE_1]] : (!fir.ref<!fir.type<_QMmhdata_typesTt_array{x:i32,y:!fir.array<10xf32>}>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
! CHECK:           %[[CONSTANT_4:.*]] = arith.constant 1 : index
! CHECK:           %[[CONSTANT_5:.*]] = arith.constant 3 : index
! CHECK:           %[[BOUNDS_0:.*]] = acc.bounds lowerbound(%[[CONSTANT_5]] : index) upperbound(%[[CONSTANT_5]] : index) extent(%[[CONSTANT_4]] : index) stride(%[[CONSTANT_4]] : index) startIdx(%[[CONSTANT_4]] : index)
! CHECK:           %[[USE_DEVICE_0:.*]] = acc.use_device varPtr(%[[DESIGNATE_2]] : !fir.ref<!fir.array<10xf32>>)
! bounds(%[[BOUNDS_0]]) -> !fir.ref<!fir.array<10xf32>> {name = "obj(10_8)%comp(2_8)%y(4)"}
! CHECK:           acc.host_data dataOperands(%[[USE_DEVICE_0]] : !fir.ref<!fir.array<10xf32>>) {
! CHECK:             %[[DECLARE_1:.*]]:2 = hlfir.declare %[[USE_DEVICE_0]](%[[SHAPE_1]]) {uniq_name = "obj(10_8)%comp(2_8)%y(4)"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
! CHECK:             %[[CONSTANT_6:.*]] = arith.constant 4 : index
! CHECK:             %[[DESIGNATE_3:.*]] = hlfir.designate %[[DECLARE_1]]#0 (%[[CONSTANT_6]])  : (!fir.ref<!fir.array<10xf32>>, index) -> !fir.ref<f32>
! CHECK:             fir.call @_QPfoo_nested(%[[DESIGNATE_3]]) fastmath<contract> : (!fir.ref<f32>) -> ()
! CHECK:             acc.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }
