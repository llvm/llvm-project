! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! Test intrinsic module procedure c_loc

! CHECK-LABEL: func.func @_QPc_loc_scalar() {
! CHECK:         %[[VAL_I_ADDR:.*]] = fir.address_of(@_QFc_loc_scalarEi) : !fir.ref<i32>
! CHECK:         %[[VAL_I:.*]]:2 = hlfir.declare %[[VAL_I_ADDR]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFc_loc_scalarEi"}
! CHECK:         %[[VAL_PTR_ALLOC:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {bindc_name = "ptr", uniq_name = "_QFc_loc_scalarEptr"}
! CHECK:         %[[VAL_PTR:.*]]:2 = hlfir.declare %[[VAL_PTR_ALLOC]] {uniq_name = "_QFc_loc_scalarEptr"}
! CHECK:         %[[VAL_BOX:.*]] = fir.embox %[[VAL_I]]#0 : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK:         %[[VAL_TMP:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_ADDR_FIELD:.*]] = fir.coordinate_of %[[VAL_TMP]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
! CHECK:         %[[VAL_BADDR:.*]] = fir.box_addr %[[VAL_BOX]] : (!fir.box<i32>) -> !fir.ref<i32>
! CHECK:         %[[VAL_INT:.*]] = fir.convert %[[VAL_BADDR]] : (!fir.ref<i32>) -> i64
! CHECK:         fir.store %[[VAL_INT]] to %[[VAL_ADDR_FIELD]] : !fir.ref<i64>
! CHECK:         %[[VAL_TMP_DECL:.*]]:2 = hlfir.declare %[[VAL_TMP]] {uniq_name = ".tmp.intrinsic_result"}
! CHECK:         %[[VAL_EXPR:.*]] = hlfir.as_expr %[[VAL_TMP_DECL]]#0
! CHECK:         hlfir.assign %[[VAL_EXPR]] to %[[VAL_PTR]]#0
! CHECK:         hlfir.destroy %[[VAL_EXPR]]
! CHECK:       }

subroutine c_loc_scalar()
  use iso_c_binding
  type(c_ptr) :: ptr
  integer, target :: i = 10
  ptr = c_loc(i)
end

! CHECK-LABEL: func.func @_QPc_loc_char() {
! CHECK:         %[[VAL_ICHR_ADDR:.*]] = fir.address_of(@_QFc_loc_charEichr) : !fir.ref<!fir.char<1,5>>
! CHECK:         %[[VAL_ICHR:.*]]:2 = hlfir.declare %[[VAL_ICHR_ADDR]] typeparams %{{.*}} {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFc_loc_charEichr"}
! CHECK:         %[[VAL_PTR_ALLOC:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {bindc_name = "ptr", uniq_name = "_QFc_loc_charEptr"}
! CHECK:         %[[VAL_PTR:.*]]:2 = hlfir.declare %[[VAL_PTR_ALLOC]] {uniq_name = "_QFc_loc_charEptr"}
! CHECK:         %[[VAL_BOX:.*]] = fir.embox %[[VAL_ICHR]]#0 : (!fir.ref<!fir.char<1,5>>) -> !fir.box<!fir.char<1,5>>
! CHECK:         %[[VAL_TMP:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_ADDR_FIELD:.*]] = fir.coordinate_of %[[VAL_TMP]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
! CHECK:         %[[VAL_BADDR:.*]] = fir.box_addr %[[VAL_BOX]] : (!fir.box<!fir.char<1,5>>) -> !fir.ref<!fir.char<1,5>>
! CHECK:         %[[VAL_INT:.*]] = fir.convert %[[VAL_BADDR]] : (!fir.ref<!fir.char<1,5>>) -> i64
! CHECK:         fir.store %[[VAL_INT]] to %[[VAL_ADDR_FIELD]] : !fir.ref<i64>
! CHECK:         %[[VAL_TMP_DECL:.*]]:2 = hlfir.declare %[[VAL_TMP]] {uniq_name = ".tmp.intrinsic_result"}
! CHECK:         hlfir.assign
! CHECK:       }

subroutine c_loc_char()
  use iso_c_binding
  type(c_ptr) :: ptr
  character(5, kind=c_char), target :: ichr = "abcde"
  ptr = c_loc(ichr)
end

! CHECK-LABEL: func.func @_QPc_loc_substring() {
! CHECK:         %[[VAL_ICHR_ADDR:.*]] = fir.address_of(@_QFc_loc_substringEichr) : !fir.ref<!fir.char<1,5>>
! CHECK:         %[[VAL_ICHR:.*]]:2 = hlfir.declare %[[VAL_ICHR_ADDR]] typeparams %{{.*}} {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFc_loc_substringEichr"}
! CHECK:         %[[VAL_PTR_ALLOC:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {bindc_name = "ptr", uniq_name = "_QFc_loc_substringEptr"}
! CHECK:         %[[VAL_PTR:.*]]:2 = hlfir.declare %[[VAL_PTR_ALLOC]] {uniq_name = "_QFc_loc_substringEptr"}
! CHECK:         %[[VAL_C2:.*]] = arith.constant 2 : index
! CHECK:         %[[VAL_C5:.*]] = arith.constant 5 : index
! CHECK:         %[[VAL_C4:.*]] = arith.constant 4 : index
! CHECK:         %[[VAL_SUBSTR:.*]] = hlfir.designate %[[VAL_ICHR]]#0  substr %[[VAL_C2]], %[[VAL_C5]]  typeparams %[[VAL_C4]] : (!fir.ref<!fir.char<1,5>>, index, index, index) -> !fir.ref<!fir.char<1,4>>
! CHECK:         %[[VAL_BOX:.*]] = fir.embox %[[VAL_SUBSTR]] : (!fir.ref<!fir.char<1,4>>) -> !fir.box<!fir.char<1,4>>
! CHECK:         %[[VAL_TMP:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_ADDR_FIELD:.*]] = fir.coordinate_of %[[VAL_TMP]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
! CHECK:         %[[VAL_BADDR:.*]] = fir.box_addr %[[VAL_BOX]] : (!fir.box<!fir.char<1,4>>) -> !fir.ref<!fir.char<1,4>>
! CHECK:         %[[VAL_INT:.*]] = fir.convert %[[VAL_BADDR]] : (!fir.ref<!fir.char<1,4>>) -> i64
! CHECK:         fir.store %[[VAL_INT]] to %[[VAL_ADDR_FIELD]] : !fir.ref<i64>
! CHECK:         %[[VAL_TMP_DECL:.*]]:2 = hlfir.declare %[[VAL_TMP]] {uniq_name = ".tmp.intrinsic_result"}
! CHECK:         hlfir.assign
! CHECK:       }

subroutine c_loc_substring()
  use iso_c_binding
  type(c_ptr) :: ptr
  character(5, kind=c_char), target :: ichr = "abcde"
  ptr = c_loc(ichr(2:))
end

! CHECK-LABEL: func.func @_QPc_loc_array() {
! CHECK:         %[[VAL_A_ADDR:.*]] = fir.address_of(@_QFc_loc_arrayEa) : !fir.ref<!fir.array<10xi32>>
! CHECK:         %[[VAL_C10:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_A:.*]]:2 = hlfir.declare %[[VAL_A_ADDR]](%{{.*}}) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFc_loc_arrayEa"}
! CHECK:         %[[VAL_PTR_ALLOC:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {bindc_name = "ptr", uniq_name = "_QFc_loc_arrayEptr"}
! CHECK:         %[[VAL_PTR:.*]]:2 = hlfir.declare %[[VAL_PTR_ALLOC]] {uniq_name = "_QFc_loc_arrayEptr"}
! CHECK:         %[[VAL_SHAPE2:.*]] = fir.shape %[[VAL_C10]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_BOX:.*]] = fir.embox %[[VAL_A]]#0(%[[VAL_SHAPE2]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
! CHECK:         %[[VAL_TMP:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_ADDR_FIELD:.*]] = fir.coordinate_of %[[VAL_TMP]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
! CHECK:         %[[VAL_BADDR:.*]] = fir.box_addr %[[VAL_BOX]] : (!fir.box<!fir.array<10xi32>>) -> !fir.ref<!fir.array<10xi32>>
! CHECK:         %[[VAL_INT:.*]] = fir.convert %[[VAL_BADDR]] : (!fir.ref<!fir.array<10xi32>>) -> i64
! CHECK:         fir.store %[[VAL_INT]] to %[[VAL_ADDR_FIELD]] : !fir.ref<i64>
! CHECK:         %[[VAL_TMP_DECL:.*]]:2 = hlfir.declare %[[VAL_TMP]] {uniq_name = ".tmp.intrinsic_result"}
! CHECK:         hlfir.assign
! CHECK:       }

subroutine c_loc_array
  use iso_c_binding
  type(c_ptr) :: ptr
  integer, target :: a(10) = 10
  ptr = c_loc(a)
end

! CHECK-LABEL: func.func @_QPc_loc_chararray() {
! CHECK:         %[[VAL_ICHR_ADDR:.*]] = fir.address_of(@_QFc_loc_chararrayEichr) : !fir.ref<!fir.array<2x!fir.char<1,5>>>
! CHECK:         %[[VAL_C2:.*]] = arith.constant 2 : index
! CHECK:         %[[VAL_ICHR:.*]]:2 = hlfir.declare %[[VAL_ICHR_ADDR]](%{{.*}}) typeparams %{{.*}} {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFc_loc_chararrayEichr"}
! CHECK:         %[[VAL_PTR_ALLOC:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {bindc_name = "ptr", uniq_name = "_QFc_loc_chararrayEptr"}
! CHECK:         %[[VAL_PTR:.*]]:2 = hlfir.declare %[[VAL_PTR_ALLOC]] {uniq_name = "_QFc_loc_chararrayEptr"}
! CHECK:         %[[VAL_SHAPE2:.*]] = fir.shape %[[VAL_C2]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_BOX:.*]] = fir.embox %[[VAL_ICHR]]#0(%[[VAL_SHAPE2]]) : (!fir.ref<!fir.array<2x!fir.char<1,5>>>, !fir.shape<1>) -> !fir.box<!fir.array<2x!fir.char<1,5>>>
! CHECK:         %[[VAL_TMP:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_ADDR_FIELD:.*]] = fir.coordinate_of %[[VAL_TMP]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
! CHECK:         %[[VAL_BADDR:.*]] = fir.box_addr %[[VAL_BOX]] : (!fir.box<!fir.array<2x!fir.char<1,5>>>) -> !fir.ref<!fir.array<2x!fir.char<1,5>>>
! CHECK:         %[[VAL_INT:.*]] = fir.convert %[[VAL_BADDR]] : (!fir.ref<!fir.array<2x!fir.char<1,5>>>) -> i64
! CHECK:         fir.store %[[VAL_INT]] to %[[VAL_ADDR_FIELD]] : !fir.ref<i64>
! CHECK:       }

subroutine c_loc_chararray()
  use iso_c_binding
  type(c_ptr) :: ptr
  character(5, kind=c_char), target :: ichr(2) = "abcde"
  ptr = c_loc(ichr)
end

! CHECK-LABEL: func.func @_QPc_loc_arrayelement() {
! CHECK:         %[[VAL_A_ADDR:.*]] = fir.address_of(@_QFc_loc_arrayelementEa) : !fir.ref<!fir.array<10xi32>>
! CHECK:         %[[VAL_C10:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_SHAPE:.*]] = fir.shape %[[VAL_C10]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_A:.*]]:2 = hlfir.declare %[[VAL_A_ADDR]](%[[VAL_SHAPE]]) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFc_loc_arrayelementEa"}
! CHECK:         %[[VAL_PTR_ALLOC:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {bindc_name = "ptr", uniq_name = "_QFc_loc_arrayelementEptr"}
! CHECK:         %[[VAL_PTR:.*]]:2 = hlfir.declare %[[VAL_PTR_ALLOC]] {uniq_name = "_QFc_loc_arrayelementEptr"}
! CHECK:         %[[VAL_C2:.*]] = arith.constant 2 : index
! CHECK:         %[[VAL_ELEM:.*]] = hlfir.designate %[[VAL_A]]#0 (%[[VAL_C2]])  : (!fir.ref<!fir.array<10xi32>>, index) -> !fir.ref<i32>
! CHECK:         %[[VAL_BOX:.*]] = fir.embox %[[VAL_ELEM]] : (!fir.ref<i32>) -> !fir.box<i32>
! CHECK:         %[[VAL_TMP:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_ADDR_FIELD:.*]] = fir.coordinate_of %[[VAL_TMP]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
! CHECK:         %[[VAL_BADDR:.*]] = fir.box_addr %[[VAL_BOX]] : (!fir.box<i32>) -> !fir.ref<i32>
! CHECK:         %[[VAL_INT:.*]] = fir.convert %[[VAL_BADDR]] : (!fir.ref<i32>) -> i64
! CHECK:         fir.store %[[VAL_INT]] to %[[VAL_ADDR_FIELD]] : !fir.ref<i64>
! CHECK:       }

subroutine c_loc_arrayelement()
  use iso_c_binding
  type(c_ptr) :: ptr
  integer, target :: a(10) = 10
  ptr = c_loc(a(2))
end

! CHECK-LABEL: func.func @_QPc_loc_arraysection() {
! CHECK:         %[[VAL_A_ADDR:.*]] = fir.address_of(@_QFc_loc_arraysectionEa) : !fir.ref<!fir.array<10xi32>>
! CHECK:         %[[VAL_C10:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_SHAPE_A:.*]] = fir.shape %[[VAL_C10]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_A:.*]]:2 = hlfir.declare %[[VAL_A_ADDR]](%[[VAL_SHAPE_A]]) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFc_loc_arraysectionEa"}
! CHECK:         %[[VAL_IND_ADDR:.*]] = fir.address_of(@_QFc_loc_arraysectionEind) : !fir.ref<i32>
! CHECK:         %[[VAL_IND:.*]]:2 = hlfir.declare %[[VAL_IND_ADDR]] {uniq_name = "_QFc_loc_arraysectionEind"}
! CHECK:         %[[VAL_PTR_ALLOC:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}> {bindc_name = "ptr", uniq_name = "_QFc_loc_arraysectionEptr"}
! CHECK:         %[[VAL_PTR:.*]]:2 = hlfir.declare %[[VAL_PTR_ALLOC]] {uniq_name = "_QFc_loc_arraysectionEptr"}
! CHECK:         %[[VAL_IND_VAL:.*]] = fir.load %[[VAL_IND]]#0 : !fir.ref<i32>
! CHECK:         %[[VAL_IND_64:.*]] = fir.convert %[[VAL_IND_VAL]] : (i32) -> i64
! CHECK:         %[[VAL_IND_IDX:.*]] = fir.convert %[[VAL_IND_64]] : (i64) -> index
! CHECK:         %[[VAL_SECTION:.*]] = hlfir.designate %[[VAL_A]]#0 (%[[VAL_IND_IDX]]:%[[VAL_C10]]:%{{.*}})  shape %{{.*}} : (!fir.ref<!fir.array<10xi32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:         %[[VAL_TMP:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_ADDR_FIELD:.*]] = fir.coordinate_of %[[VAL_TMP]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
! CHECK:         %[[VAL_BADDR:.*]] = fir.box_addr %[[VAL_SECTION]] : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
! CHECK:         %[[VAL_INT:.*]] = fir.convert %[[VAL_BADDR]] : (!fir.ref<!fir.array<?xi32>>) -> i64
! CHECK:         fir.store %[[VAL_INT]] to %[[VAL_ADDR_FIELD]] : !fir.ref<i64>
! CHECK:       }

subroutine c_loc_arraysection()
  use iso_c_binding
  type(c_ptr) :: ptr
  integer :: ind = 3
  integer, target :: a(10) = 10
  ptr = c_loc(a(ind:))
end

! CHECK-LABEL: func.func @_QPc_loc_non_save_pointer_scalar() {
! CHECK:         %[[VAL_I_ALLOC:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "i", uniq_name = "_QFc_loc_non_save_pointer_scalarEi"}
! CHECK:         %[[VAL_I:.*]]:2 = hlfir.declare %[[VAL_I_ALLOC]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFc_loc_non_save_pointer_scalarEi"}
! CHECK:         fir.call @_FortranAPointerAllocate
! CHECK:         hlfir.assign %{{.*}} to %{{.*}} : i32, !fir.ptr<i32>
! CHECK:         %[[VAL_I_LD:.*]] = fir.load %[[VAL_I]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:         %[[VAL_TMP:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_ADDR_FIELD:.*]] = fir.coordinate_of %[[VAL_TMP]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
! CHECK:         %[[VAL_BADDR:.*]] = fir.box_addr %[[VAL_I_LD]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:         %[[VAL_INT:.*]] = fir.convert %[[VAL_BADDR]] : (!fir.ptr<i32>) -> i64
! CHECK:         fir.store %[[VAL_INT]] to %[[VAL_ADDR_FIELD]] : !fir.ref<i64>
! CHECK:         return
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
! CHECK:         %[[VAL_I_ADDR:.*]] = fir.address_of(@_QFc_loc_save_pointer_scalarEi) : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:         %[[VAL_I:.*]]:2 = hlfir.declare %[[VAL_I_ADDR]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFc_loc_save_pointer_scalarEi"}
! CHECK:         fir.call @_FortranAPointerAllocate
! CHECK:         hlfir.assign %{{.*}} to %{{.*}} : i32, !fir.ptr<i32>
! CHECK:         %[[VAL_I_LD:.*]] = fir.load %[[VAL_I]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:         %[[VAL_TMP:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_ADDR_FIELD:.*]] = fir.coordinate_of %[[VAL_TMP]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
! CHECK:         %[[VAL_BADDR:.*]] = fir.box_addr %[[VAL_I_LD]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:         %[[VAL_INT:.*]] = fir.convert %[[VAL_BADDR]] : (!fir.ptr<i32>) -> i64
! CHECK:         fir.store %[[VAL_INT]] to %[[VAL_ADDR_FIELD]] : !fir.ref<i64>
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
! CHECK:         %[[VAL_TT_ALLOC:.*]] = fir.alloca !fir.type<_QFc_loc_derived_typeTt{i:i32}> {bindc_name = "tt", fir.target, uniq_name = "_QFc_loc_derived_typeEtt"}
! CHECK:         %[[VAL_TT:.*]]:2 = hlfir.declare %[[VAL_TT_ALLOC]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFc_loc_derived_typeEtt"}
! CHECK:         %[[VAL_BOX:.*]] = fir.embox %[[VAL_TT]]#0 : (!fir.ref<!fir.type<_QFc_loc_derived_typeTt{i:i32}>>) -> !fir.box<!fir.type<_QFc_loc_derived_typeTt{i:i32}>>
! CHECK:         %[[VAL_TMP:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_ADDR_FIELD:.*]] = fir.coordinate_of %[[VAL_TMP]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
! CHECK:         %[[VAL_BADDR:.*]] = fir.box_addr %[[VAL_BOX]] : (!fir.box<!fir.type<_QFc_loc_derived_typeTt{i:i32}>>) -> !fir.ref<!fir.type<_QFc_loc_derived_typeTt{i:i32}>>
! CHECK:         %[[VAL_INT:.*]] = fir.convert %[[VAL_BADDR]] : (!fir.ref<!fir.type<_QFc_loc_derived_typeTt{i:i32}>>) -> i64
! CHECK:         fir.store %[[VAL_INT]] to %[[VAL_ADDR_FIELD]] : !fir.ref<i64>
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
! CHECK:         %[[VAL_A_ALLOC:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>> {bindc_name = "a", uniq_name = "_QFc_loc_pointer_arrayEa"}
! CHECK:         %[[VAL_A:.*]]:2 = hlfir.declare %[[VAL_A_ALLOC]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFc_loc_pointer_arrayEa"}
! CHECK:         fir.call @_FortranAPointerAllocate
! CHECK:         hlfir.assign %{{.*}} to %{{.*}} : i32, !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:         %[[VAL_A_LD:.*]] = fir.load %[[VAL_A]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:         %[[VAL_TMP:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_ADDR_FIELD:.*]] = fir.coordinate_of %[[VAL_TMP]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
! CHECK:         %[[VAL_BADDR:.*]] = fir.box_addr %[[VAL_A_LD]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
! CHECK:         %[[VAL_INT:.*]] = fir.convert %[[VAL_BADDR]] : (!fir.ptr<!fir.array<?xi32>>) -> i64
! CHECK:         fir.store %[[VAL_INT]] to %[[VAL_ADDR_FIELD]] : !fir.ref<i64>
! CHECK:       }

subroutine c_loc_pointer_array
  use iso_c_binding
  type(c_ptr) :: ptr
  integer, pointer :: a(:)
  allocate(a(10))
  a = 10
  ptr = c_loc(a)
end
