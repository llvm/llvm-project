! RUN: bbc --use-desc-for-alloc=false --emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -mllvm --use-desc-for-alloc=false -emit-fir %s -o - | FileCheck %s

! Test intrinsic module procedure c_loc

! CHECK-LABEL: func.func @_QPc_loc_scalar() {
! CHECK:         %[[VAL_0:.*]] = fir.address_of(@_QFc_loc_scalarEi) : !fir.ref<i32>
! CHECK:         %[[VAL_1:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {bindc_name = "ptr", uniq_name = "_QFc_loc_scalarEptr"}
! CHECK:         %[[VAL_2:.*]] = fir.embox %[[VAL_0]] : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK-DAG:         %[[VAL_4:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<i32>) -> !fir.ref<i32>
! CHECK-DAG:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<i32>) -> i64
! CHECK-DAG:         %[[VAL_6:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK-DAG:         %[[VAL_7:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_6]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         fir.store %[[VAL_5]] to %[[VAL_7]] : !fir.ref<i64>
! CHECK:       }

subroutine c_loc_scalar()
  use iso_c_binding
  type(c_ptr) :: ptr
  integer, target :: i = 10
  ptr = c_loc(i)
end

! CHECK-LABEL: func.func @_QPc_loc_char() {
! CHECK:         %[[VAL_0:.*]] = fir.address_of(@_QFc_loc_charEichr) : !fir.ref<!fir.char<1,5>>
! CHECK:         %[[VAL_1:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {bindc_name = "ptr", uniq_name = "_QFc_loc_charEptr"}
! CHECK:         %[[VAL_2:.*]] = fir.embox %[[VAL_0]] : (!fir.ref<!fir.char<1,5>>) -> !fir.box<!fir.char<1,5>>
! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK-DAG:         %[[VAL_4:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.char<1,5>>) -> !fir.ref<!fir.char<1,5>>
! CHECK-DAG:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<!fir.char<1,5>>) -> i64
! CHECK-DAG:         %[[VAL_6:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK-DAG:         %[[VAL_7:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_6]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         fir.store %[[VAL_5]] to %[[VAL_7]] : !fir.ref<i64>
! CHECK:       }

subroutine c_loc_char()
  use iso_c_binding
  type(c_ptr) :: ptr
  character(5, kind=c_char), target :: ichr = "abcde"
  ptr = c_loc(ichr)
end

! CHECK-LABEL: func.func @_QPc_loc_substring() {
! CHECK:         %[[VAL_0:.*]] = fir.address_of(@_QFc_loc_substringEichr) : !fir.ref<!fir.char<1,5>>
! CHECK:         %[[VAL_1:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {bindc_name = "ptr", uniq_name = "_QFc_loc_substringEptr"}
! CHECK:         %[[VAL_2:.*]] = arith.constant 2 : i64
! CHECK:         %[[VAL_3:.*]] = arith.constant 5 : i64
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_2]] : (i64) -> index
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_3]] : (i64) -> index
! CHECK:         %[[VAL_6:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_7:.*]] = arith.subi %[[VAL_4]], %[[VAL_6]] : index
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<!fir.array<5x!fir.char<1>>>
! CHECK:         %[[VAL_9:.*]] = fir.coordinate_of %[[VAL_8]], %[[VAL_7]] : (!fir.ref<!fir.array<5x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:         %[[VAL_11:.*]] = arith.subi %[[VAL_5]], %[[VAL_4]] : index
! CHECK:         %[[VAL_12:.*]] = arith.addi %[[VAL_11]], %[[VAL_6]] : index
! CHECK:         %[[VAL_13:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_14:.*]] = arith.cmpi slt, %[[VAL_12]], %[[VAL_13]] : index
! CHECK:         %[[VAL_15:.*]] = arith.select %[[VAL_14]], %[[VAL_13]], %[[VAL_12]] : index
! CHECK:         %[[VAL_16:.*]] = fir.embox %[[VAL_10]] typeparams %[[VAL_15]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK:         %[[VAL_17:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK-DAG:         %[[VAL_18:.*]] = fir.box_addr %[[VAL_16]] : (!fir.box<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,?>>
! CHECK-DAG:         %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (!fir.ref<!fir.char<1,?>>) -> i64
! CHECK-DAG:         %[[VAL_20:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK-DAG:         %[[VAL_21:.*]] = fir.coordinate_of %[[VAL_17]], %[[VAL_20]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         fir.store %[[VAL_19]] to %[[VAL_21]] : !fir.ref<i64>
! CHECK:       }

subroutine c_loc_substring()
  use iso_c_binding
  type(c_ptr) :: ptr
  character(5, kind=c_char), target :: ichr = "abcde"
  ptr = c_loc(ichr(2:))
end

! CHECK-LABEL: func.func @_QPc_loc_array() {
! CHECK:         %[[VAL_0:.*]] = fir.address_of(@_QFc_loc_arrayEa) : !fir.ref<!fir.array<10xi32>>
! CHECK:         %[[VAL_1:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {bindc_name = "ptr", uniq_name = "_QFc_loc_arrayEptr"}
! CHECK:         %[[VAL_3:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_4:.*]] = fir.embox %[[VAL_0]](%[[VAL_3]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
! CHECK:         %[[VAL_5:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK-DAG:         %[[VAL_6:.*]] = fir.box_addr %[[VAL_4]] : (!fir.box<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>>
! CHECK-DAG:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.array<10xi32>>) -> i64
! CHECK-DAG:         %[[VAL_8:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK-DAG:         %[[VAL_9:.*]] = fir.coordinate_of %[[VAL_5]], %[[VAL_8]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         fir.store %[[VAL_7]] to %[[VAL_9]] : !fir.ref<i64>
! CHECK:       }

subroutine c_loc_array
  use iso_c_binding
  type(c_ptr) :: ptr
  integer, target :: a(10) = 10
  ptr = c_loc(a)
end

! CHECK-LABEL: func.func @_QPc_loc_chararray() {
! CHECK:         %[[VAL_0:.*]] = fir.address_of(@_QFc_loc_chararrayEichr) : !fir.ref<!fir.array<2x!fir.char<1,5>>>
! CHECK:         %[[VAL_1:.*]] = arith.constant 2 : index
! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {bindc_name = "ptr", uniq_name = "_QFc_loc_chararrayEptr"}
! CHECK:         %[[VAL_3:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_4:.*]] = fir.embox %[[VAL_0]](%[[VAL_3]]) : (!fir.ref<!fir.array<2x!fir.char<1,5>>>, !fir.shape<1>) -> !fir.box<!fir.array<2x!fir.char<1,5>>>
! CHECK:         %[[VAL_5:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK-DAG:         %[[VAL_6:.*]] = fir.box_addr %[[VAL_4]] : (!fir.box<!fir.array<2x!fir.char<1,5>>>) -> !fir.ref<!fir.array<2x!fir.char<1,5>>>
! CHECK-DAG:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.array<2x!fir.char<1,5>>>) -> i64
! CHECK-DAG:         %[[VAL_8:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK-DAG:         %[[VAL_9:.*]] = fir.coordinate_of %[[VAL_5]], %[[VAL_8]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         fir.store %[[VAL_7]] to %[[VAL_9]] : !fir.ref<i64>
! CHECK:       }

subroutine c_loc_chararray()
  use iso_c_binding
  type(c_ptr) :: ptr
  character(5, kind=c_char), target :: ichr(2) = "abcde"
  ptr = c_loc(ichr)
end

! CHECK-LABEL: func.func @_QPc_loc_arrayelement() {
! CHECK:         %[[VAL_0:.*]] = fir.address_of(@_QFc_loc_arrayelementEa) : !fir.ref<!fir.array<10xi32>>
! CHECK:         %[[VAL_1:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {bindc_name = "ptr", uniq_name = "_QFc_loc_arrayelementEptr"}
! CHECK:         %[[VAL_2:.*]] = arith.constant 2 : i64
! CHECK:         %[[VAL_3:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_4:.*]] = arith.subi %[[VAL_2]], %[[VAL_3]] : i64
! CHECK:         %[[VAL_5:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_4]] : (!fir.ref<!fir.array<10xi32>>, i64) -> !fir.ref<i32>
! CHECK:         %[[VAL_6:.*]] = fir.embox %[[VAL_5]] : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK:         %[[VAL_7:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK-DAG:         %[[VAL_8:.*]] = fir.box_addr %[[VAL_6]] : (!fir.box<i32>) -> !fir.ref<i32>
! CHECK-DAG:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (!fir.ref<i32>) -> i64
! CHECK-DAG:         %[[VAL_10:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK-DAG:         %[[VAL_11:.*]] = fir.coordinate_of %[[VAL_7]], %[[VAL_10]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         fir.store %[[VAL_9]] to %[[VAL_11]] : !fir.ref<i64>
! CHECK:       }

subroutine c_loc_arrayelement()
  use iso_c_binding
  type(c_ptr) :: ptr
  integer, target :: a(10) = 10
  ptr = c_loc(a(2))
end

! CHECK-LABEL: func.func @_QPc_loc_arraysection() {
! CHECK:         %[[VAL_0:.*]] = fir.address_of(@_QFc_loc_arraysectionEa) : !fir.ref<!fir.array<10xi32>>
! CHECK:         %[[VAL_1:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_2:.*]] = fir.address_of(@_QFc_loc_arraysectionEind) : !fir.ref<i32>
! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {bindc_name = "ptr", uniq_name = "_QFc_loc_arraysectionEptr"}
! CHECK:         %[[VAL_4:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_5:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> i64
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i64) -> index
! CHECK:         %[[VAL_8:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i64) -> index
! CHECK:         %[[VAL_10:.*]] = arith.addi %[[VAL_4]], %[[VAL_1]] : index
! CHECK:         %[[VAL_11:.*]] = arith.subi %[[VAL_10]], %[[VAL_4]] : index
! CHECK:         %[[VAL_12:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_13:.*]] = fir.slice %[[VAL_7]], %[[VAL_11]], %[[VAL_9]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_14:.*]] = fir.embox %[[VAL_0]](%[[VAL_12]]) {{\[}}%[[VAL_13]]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:         %[[VAL_15:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK-DAG:         %[[VAL_16:.*]] = fir.box_addr %[[VAL_14]] : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
! CHECK-DAG:         %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (!fir.ref<!fir.array<?xi32>>) -> i64
! CHECK-DAG:         %[[VAL_18:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK-DAG:         %[[VAL_19:.*]] = fir.coordinate_of %[[VAL_15]], %[[VAL_18]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         fir.store %[[VAL_17]] to %[[VAL_19]] : !fir.ref<i64>
! CHECK:       }

subroutine c_loc_arraysection()
  use iso_c_binding
  type(c_ptr) :: ptr
  integer :: ind = 3
  integer, target :: a(10) = 10
  ptr = c_loc(a(ind:))
end

! CHECK-LABEL: func.func @_QPc_loc_non_save_pointer_scalar() {
! CHECK:         %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "i", uniq_name = "_QFc_loc_non_save_pointer_scalarEi"}
! CHECK:         %[[VAL_1:.*]] = fir.alloca !fir.ptr<i32> {uniq_name = "_QFc_loc_non_save_pointer_scalarEi.addr"}
! CHECK:         %[[VAL_2:.*]] = fir.zero_bits !fir.ptr<i32>
! CHECK:         fir.store %[[VAL_2]] to %[[VAL_1]] : !fir.ref<!fir.ptr<i32>>
! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {bindc_name = "ptr", uniq_name = "_QFc_loc_non_save_pointer_scalarEptr"}
! CHECK:         %[[VAL_4:.*]] = fir.allocmem i32 {fir.must_be_heap = true, uniq_name = "_QFc_loc_non_save_pointer_scalarEi.alloc"}
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.heap<i32>) -> !fir.ptr<i32>
! CHECK:         fir.store %[[VAL_5]] to %[[VAL_1]] : !fir.ref<!fir.ptr<i32>>
! CHECK:         %[[VAL_6:.*]] = arith.constant 10 : i32
! CHECK:         %[[VAL_7:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.ptr<i32>>
! CHECK:         fir.store %[[VAL_6]] to %[[VAL_7]] : !fir.ptr<i32>
! CHECK:         %[[VAL_8:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.ptr<i32>>
! CHECK:         %[[VAL_9:.*]] = fir.embox %[[VAL_8]] : (!fir.ptr<i32>) -> !fir.box<i32>
! CHECK:         %[[VAL_10:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK-DAG:         %[[VAL_11:.*]] = fir.box_addr %[[VAL_9]] : (!fir.box<i32>) -> !fir.ref<i32>
! CHECK-DAG:         %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (!fir.ref<i32>) -> i64
! CHECK-DAG:         %[[VAL_13:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK-DAG:         %[[VAL_14:.*]] = fir.coordinate_of %[[VAL_10]], %[[VAL_13]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         fir.store %[[VAL_12]] to %[[VAL_14]] : !fir.ref<i64>
! CHECK:       }

subroutine c_loc_non_save_pointer_scalar()
  use iso_c_binding
  type(c_ptr) :: ptr
  integer, pointer :: i
  allocate(i)
  i = 10
  ptr = c_loc(i)
end

! CHECK-LABEL: func.func @_QPc_loc_save_pointer_scalar() {
! CHECK:         %[[VAL_0:.*]] = fir.address_of(@_QFc_loc_save_pointer_scalarEi) : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:         %[[VAL_7:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:         %[[VAL_8:.*]] = fir.box_addr %[[VAL_7]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:         %[[VAL_9:.*]] = fir.embox %[[VAL_8:.*]] : (!fir.ptr<i32>) -> !fir.box<i32>
! CHECK:         %[[VAL_10:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK-DAG:         %[[VAL_11:.*]] = fir.box_addr %[[VAL_9]] : (!fir.box<i32>) -> !fir.ref<i32>
! CHECK-DAG:         %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (!fir.ref<i32>) -> i64
! CHECK-DAG:         %[[VAL_13:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK-DAG:         %[[VAL_14:.*]] = fir.coordinate_of %[[VAL_10]], %[[VAL_13]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         fir.store %[[VAL_12]] to %[[VAL_14]] : !fir.ref<i64>
! CHECK:       }

subroutine c_loc_save_pointer_scalar()
  use iso_c_binding
  type(c_ptr) :: ptr
  integer, pointer, save :: i
  allocate(i)
  i = 10
  ptr = c_loc(i)
end

! CHECK-LABEL: func.func @_QPc_loc_derived_type() {
! CHECK:         %[[VAL_1:.*]] = fir.alloca !fir.type<_QFc_loc_derived_typeTt{i:i32}> {bindc_name = "tt", fir.target, uniq_name = "_QFc_loc_derived_typeEtt"}
! CHECK:         %[[VAL_8:.*]] = fir.embox %[[VAL_1]] : (!fir.ref<!fir.type<_QFc_loc_derived_typeTt{i:i32}>>) -> !fir.box<!fir.type<_QFc_loc_derived_typeTt{i:i32}>>
! CHECK:         %[[VAL_9:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK-DAG:         %[[VAL_10:.*]] = fir.box_addr %[[VAL_8:.*]] : (!fir.box<!fir.type<_QFc_loc_derived_typeTt{i:i32}>>) -> !fir.ref<!fir.type<_QFc_loc_derived_typeTt{i:i32}>>
! CHECK-DAG:         %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (!fir.ref<!fir.type<_QFc_loc_derived_typeTt{i:i32}>>) -> i64
! CHECK-DAG:         %[[VAL_12:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK-DAG:         %[[VAL_13:.*]] = fir.coordinate_of %[[VAL_9]], %[[VAL_12]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         fir.store %[[VAL_11]] to %[[VAL_13]] : !fir.ref<i64>
! CHECK:       }

subroutine c_loc_derived_type
  use iso_c_binding
  type(c_ptr) :: ptr
  type t
    integer :: i = 1
  end type
  type(t), target :: tt
  ptr = c_loc(tt)
end

! CHECK-LABEL: func.func @_QPc_loc_pointer_array() {
! CHECK:         %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>> {bindc_name = "a", uniq_name = "_QFc_loc_pointer_arrayEa"}
! CHECK:         %[[VAL_30:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_31:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK-DAG:         %[[VAL_32:.*]] = fir.box_addr %[[VAL_30:.*]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
! CHECK-DAG:         %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (!fir.ptr<!fir.array<?xi32>>) -> i64
! CHECK-DAG:         %[[VAL_34:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK-DAG:         %[[VAL_35:.*]] = fir.coordinate_of %[[VAL_31]], %[[VAL_34]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         fir.store %[[VAL_33]] to %[[VAL_35]] : !fir.ref<i64>
! CHECK:       }

subroutine c_loc_pointer_array
  use iso_c_binding
  type(c_ptr) :: ptr
  integer, pointer :: a(:)
  allocate(a(10))
  a = 10
  ptr = c_loc(a)
end
