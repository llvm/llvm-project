! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! Test intrinsic module procedure c_f_pointer

! CHECK-LABEL: func.func @_QPtest_scalar(
! CHECK-SAME:                            %[[VAL_0:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>> {fir.bindc_name = "cptr"},
! CHECK-SAME:                            %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.ptr<f32>>> {fir.bindc_name = "fptr"}) {
! CHECK:         %[[CPTR:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}} {uniq_name = "_QFtest_scalarEcptr"}
! CHECK:         %[[FPTR:.*]]:2 = hlfir.declare %[[VAL_1]] {{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_scalarEfptr"}
! CHECK:         %[[ADDR:.*]] = fir.coordinate_of %[[CPTR]]#0, __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
! CHECK:         %[[ADDR_VAL:.*]] = fir.load %[[ADDR]] : !fir.ref<i64>
! CHECK:         %[[PTR:.*]] = fir.convert %[[ADDR_VAL]] : (i64) -> !fir.ptr<f32>
! CHECK:         %[[BOX:.*]] = fir.embox %[[PTR]] : (!fir.ptr<f32>) -> !fir.box<!fir.ptr<f32>>
! CHECK:         fir.store %[[BOX]] to %[[FPTR]]#0 : !fir.ref<!fir.box<!fir.ptr<f32>>>
! CHECK:         return
! CHECK:       }

subroutine test_scalar(cptr, fptr)
  use iso_c_binding
  real, pointer :: fptr
  type(c_ptr) :: cptr

  call c_f_pointer(cptr, fptr)
end

! CHECK-LABEL: func.func @_QPtest_array(
! CHECK-SAME:                           %[[VAL_0:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>> {fir.bindc_name = "cptr"},
! CHECK-SAME:                           %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>> {fir.bindc_name = "fptr"}) {
! CHECK:         %[[CPTR:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}} {uniq_name = "_QFtest_arrayEcptr"}
! CHECK:         %[[FPTR:.*]]:2 = hlfir.declare %[[VAL_1]] {{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_arrayEfptr"}
! CHECK:         %[[ARR:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = ".tmp.arrayctor"} : (!fir.heap<!fir.array<2xi32>>, !fir.shape<1>) -> (!fir.heap<!fir.array<2xi32>>, !fir.heap<!fir.array<2xi32>>)
! CHECK:         %[[ASSOC:.*]]:3 = hlfir.associate %{{.*}}({{.*}}) {{.*}} : (!hlfir.expr<2xi32>, !fir.shape<1>) -> (!fir.ref<!fir.array<2xi32>>, !fir.ref<!fir.array<2xi32>>, i1)
! CHECK:         %[[ADDR:.*]] = fir.coordinate_of %[[CPTR]]#0, __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
! CHECK:         %[[ADDR_VAL:.*]] = fir.load %[[ADDR]] : !fir.ref<i64>
! CHECK:         %[[PTR:.*]] = fir.convert %[[ADDR_VAL]] : (i64) -> !fir.ptr<!fir.array<?x?xf32>>
! CHECK:         %[[CONST_0:.*]] = arith.constant 0 : index
! CHECK:         %[[E0:.*]] = fir.coordinate_of %[[ASSOC]]#0, %[[CONST_0]] : (!fir.ref<!fir.array<2xi32>>, index) -> !fir.ref<i32>
! CHECK:         %[[D1_VAL:.*]] = fir.load %[[E0]] : !fir.ref<i32>
! CHECK:         %[[D1:.*]] = fir.convert %[[D1_VAL]] : (i32) -> index
! CHECK:         %[[CONST_1:.*]] = arith.constant 1 : index
! CHECK:         %[[E1:.*]] = fir.coordinate_of %[[ASSOC]]#0, %[[CONST_1]] : (!fir.ref<!fir.array<2xi32>>, index) -> !fir.ref<i32>
! CHECK:         %[[D2_VAL:.*]] = fir.load %[[E1]] : !fir.ref<i32>
! CHECK:         %[[D2:.*]] = fir.convert %[[D2_VAL]] : (i32) -> index
! CHECK:         %[[SHAPE:.*]] = fir.shape %[[D1]], %[[D2]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[BOX:.*]] = fir.embox %[[PTR]](%[[SHAPE]]) : (!fir.ptr<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
! CHECK:         fir.store %[[BOX]] to %[[FPTR]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>

subroutine test_array(cptr, fptr)
  use iso_c_binding
  real, pointer :: fptr(:,:)
  type(c_ptr) :: cptr
  integer :: x = 3, y = 4

  call c_f_pointer(cptr, fptr, [x, y])
end

! CHECK-LABEL: func.func @_QPtest_char(
! CHECK-SAME:                          %[[VAL_0:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>> {fir.bindc_name = "cptr"},
! CHECK-SAME:                          %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.char<1,10>>>> {fir.bindc_name = "fptr"}) {
! CHECK:         %[[CPTR:.*]]:2 = hlfir.declare %[[VAL_0]] {{.*}} {uniq_name = "_QFtest_charEcptr"}
! CHECK:         %[[FPTR:.*]]:2 = hlfir.declare %[[VAL_1]] typeparams %{{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_charEfptr"}
! CHECK:         %[[ADDR:.*]] = fir.coordinate_of %[[CPTR]]#0, __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
! CHECK:         %[[ADDR_VAL:.*]] = fir.load %[[ADDR]] : !fir.ref<i64>
! CHECK:         %[[PTR:.*]] = fir.convert %[[ADDR_VAL]] : (i64) -> !fir.ptr<!fir.char<1,10>>
! CHECK:         %[[BOX:.*]] = fir.embox %[[PTR]] : (!fir.ptr<!fir.char<1,10>>) -> !fir.box<!fir.ptr<!fir.char<1,10>>>
! CHECK:         fir.store %[[BOX]] to %[[FPTR]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,10>>>>
! CHECK:         return
! CHECK:       }

subroutine test_char(cptr, fptr)
  use iso_c_binding
  character(10), pointer :: fptr
  type(c_ptr) :: cptr

  call c_f_pointer(cptr, fptr)
end

! CHECK-LABEL: func.func @_QPtest_chararray(
! CHECK-SAME:                               %[[VAL_0:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>> {fir.bindc_name = "cptr"},
! CHECK-SAME:                               %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x!fir.char<1,?>>>>> {fir.bindc_name = "fptr"},
! CHECK-SAME:                               %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:         %[[N:.*]]:2 = hlfir.declare %[[VAL_2]] {{.*}} {uniq_name = "_QFtest_chararrayEn"}
! CHECK:         %[[N_VAL:.*]] = fir.load %[[N]]#0 : !fir.ref<i32>
! CHECK:         %[[ZERO:.*]] = arith.constant 0 : i32
! CHECK:         %[[CMP:.*]] = arith.cmpi sgt, %[[N_VAL]], %[[ZERO]] : i32
! CHECK:         %[[LEN:.*]] = arith.select %[[CMP]], %[[N_VAL]], %[[ZERO]] : i32
! CHECK:         %[[FPTR:.*]]:2 = hlfir.declare %[[VAL_1]] typeparams %[[LEN]] {{.*}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest_chararrayEfptr"}
! CHECK:         %[[ASSOC:.*]]:3 = hlfir.associate %{{.*}}({{.*}}) {{.*}} : (!hlfir.expr<2xi32>, !fir.shape<1>) -> (!fir.ref<!fir.array<2xi32>>, !fir.ref<!fir.array<2xi32>>, i1)
! CHECK:         %[[ADDR:.*]] = fir.coordinate_of %{{.*}}#0, __address
! CHECK:         %[[ADDR_VAL:.*]] = fir.load %[[ADDR]] : !fir.ref<i64>
! CHECK:         %[[PTR:.*]] = fir.convert %[[ADDR_VAL]] : (i64) -> !fir.ptr<!fir.array<?x?x!fir.char<1,?>>>
! CHECK:         %[[CZERO:.*]] = arith.constant 0 : index
! CHECK:         %[[E0:.*]] = fir.coordinate_of %[[ASSOC]]#0, %[[CZERO]] : (!fir.ref<!fir.array<2xi32>>, index) -> !fir.ref<i32>
! CHECK:         %[[D1_VAL:.*]] = fir.load %[[E0]] : !fir.ref<i32>
! CHECK:         %[[D1:.*]] = fir.convert %[[D1_VAL]] : (i32) -> index
! CHECK:         %[[CONE:.*]] = arith.constant 1 : index
! CHECK:         %[[E1:.*]] = fir.coordinate_of %[[ASSOC]]#0, %[[CONE]] : (!fir.ref<!fir.array<2xi32>>, index) -> !fir.ref<i32>
! CHECK:         %[[D2_VAL:.*]] = fir.load %[[E1]] : !fir.ref<i32>
! CHECK:         %[[D2:.*]] = fir.convert %[[D2_VAL]] : (i32) -> index
! CHECK:         %[[SHAPE:.*]] = fir.shape %[[D1]], %[[D2]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[BOX:.*]] = fir.embox %[[PTR]](%[[SHAPE]]) typeparams %[[LEN]] : (!fir.ptr<!fir.array<?x?x!fir.char<1,?>>>, !fir.shape<2>, i32) -> !fir.box<!fir.ptr<!fir.array<?x?x!fir.char<1,?>>>>
! CHECK:         fir.store %[[BOX]] to %[[FPTR]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x!fir.char<1,?>>>>>

subroutine test_chararray(cptr, fptr, n)
  use iso_c_binding
  character(n), pointer :: fptr(:,:)
  type(c_ptr) :: cptr
  integer :: x = 3, y = 4

  call c_f_pointer(cptr, fptr, [x, y])
end

! CHECK-LABEL: func.func @_QPdynamic_shape_size(
subroutine dynamic_shape_size(cptr, fptr, shape)
  use iso_c_binding
  type(c_ptr)  :: cptr
  real, pointer :: fptr(:, :)
  integer :: shape(:)
! CHECK:         %[[SHAPE_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFdynamic_shape_sizeEshape"}
! CHECK:         %[[CZERO:.*]] = arith.constant 0 : index
! CHECK:         %[[E0:.*]] = fir.coordinate_of %[[SHAPE_DECL]]#1, %[[CZERO]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:         %[[D1_VAL:.*]] = fir.load %[[E0]] : !fir.ref<i32>
! CHECK:         %[[D1:.*]] = fir.convert %[[D1_VAL]] : (i32) -> index
! CHECK:         %[[CONE:.*]] = arith.constant 1 : index
! CHECK:         %[[E1:.*]] = fir.coordinate_of %[[SHAPE_DECL]]#1, %[[CONE]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:         %[[D2_VAL:.*]] = fir.load %[[E1]] : !fir.ref<i32>
! CHECK:         %[[D2:.*]] = fir.convert %[[D2_VAL]] : (i32) -> index
! CHECK:         %[[SHAPE_OP:.*]] = fir.shape %[[D1]], %[[D2]] : (index, index) -> !fir.shape<2>
  call c_f_pointer(cptr, fptr, shape)
end subroutine

! CHECK-LABEL: func.func @_QPdynamic_shape_size_2(
subroutine dynamic_shape_size_2(cptr, fptr, shape, n)
  use iso_c_binding
  type(c_ptr)  :: cptr
  real, pointer :: fptr(:, :)
  integer :: n
  integer :: shape(n)
! CHECK:         %[[SHAPE_DECL:.*]]:2 = hlfir.declare %{{.*}}({{.*}}) {{.*}} {uniq_name = "_QFdynamic_shape_size_2Eshape"}
! CHECK:         %[[CZERO:.*]] = arith.constant 0 : index
! CHECK:         %[[E0:.*]] = fir.coordinate_of %[[SHAPE_DECL]]#1, %[[CZERO]] : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:         %[[D1_VAL:.*]] = fir.load %[[E0]] : !fir.ref<i32>
! CHECK:         %[[D1:.*]] = fir.convert %[[D1_VAL]] : (i32) -> index
! CHECK:         %[[CONE:.*]] = arith.constant 1 : index
! CHECK:         %[[E1:.*]] = fir.coordinate_of %[[SHAPE_DECL]]#1, %[[CONE]] : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:         %[[D2_VAL:.*]] = fir.load %[[E1]] : !fir.ref<i32>
! CHECK:         %[[D2:.*]] = fir.convert %[[D2_VAL]] : (i32) -> index
! CHECK:         %[[SHAPE_OP:.*]] = fir.shape %[[D1]], %[[D2]] : (index, index) -> !fir.shape<2>
  call c_f_pointer(cptr, fptr, shape)
end subroutine

! CHECK-LABEL: func.func @_QPdynamic_shape_lower(
subroutine dynamic_shape_lower(cptr, fpr, shape, lower)
  use iso_c_binding
  type(c_ptr)  :: cptr
  real, pointer :: fptr(:, :)
  integer :: n
  integer :: shape(:)
  integer :: lower(:)
! CHECK:         %[[FPTR_ALLOCA:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
! CHECK:         %[[ZERO_BITS:.*]] = fir.zero_bits !fir.ptr<!fir.array<?x?xf32>>
! CHECK:         %[[ZERO_INDEX:.*]] = arith.constant 0 : index
! CHECK:         %[[ZERO_SHAPE:.*]] = fir.shape %[[ZERO_INDEX]], %[[ZERO_INDEX]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[INIT_BOX:.*]] = fir.embox %[[ZERO_BITS]](%[[ZERO_SHAPE]]) : (!fir.ptr<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
! CHECK:         fir.store %[[INIT_BOX]] to %[[FPTR_ALLOCA]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
! CHECK:         %[[FPTR_DECL:.*]]:2 = hlfir.declare %[[FPTR_ALLOCA]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFdynamic_shape_lowerEfptr"}
! CHECK:         %[[LOWER_DECL:.*]]:2 = hlfir.declare %{{.*}} {{.*}} {uniq_name = "_QFdynamic_shape_lowerElower"}
! CHECK:         %[[SHAPE_DECL:.*]]:2 = hlfir.declare %{{.*}} {{.*}} {uniq_name = "_QFdynamic_shape_lowerEshape"}
! CHECK:         %[[ADDR:.*]] = fir.coordinate_of %{{.*}}#0, __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
! CHECK:         %[[ADDR_VAL:.*]] = fir.load %[[ADDR]] : !fir.ref<i64>
! CHECK:         %[[PTR:.*]] = fir.convert %[[ADDR_VAL]] : (i64) -> !fir.ptr<!fir.array<?x?xf32>>
! CHECK:         %[[CZERO:.*]] = arith.constant 0 : index
! CHECK:         %[[SE0:.*]] = fir.coordinate_of %[[SHAPE_DECL]]#1, %[[CZERO]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:         %[[S1_VAL:.*]] = fir.load %[[SE0]] : !fir.ref<i32>
! CHECK:         %[[S1:.*]] = fir.convert %[[S1_VAL]] : (i32) -> index
! CHECK:         %[[CONE:.*]] = arith.constant 1 : index
! CHECK:         %[[SE1:.*]] = fir.coordinate_of %[[SHAPE_DECL]]#1, %[[CONE]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:         %[[S2_VAL:.*]] = fir.load %[[SE1]] : !fir.ref<i32>
! CHECK:         %[[S2:.*]] = fir.convert %[[S2_VAL]] : (i32) -> index
! CHECK:         %[[CZERO2:.*]] = arith.constant 0 : index
! CHECK:         %[[LE0:.*]] = fir.coordinate_of %[[LOWER_DECL]]#1, %[[CZERO2]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:         %[[L1_VAL:.*]] = fir.load %[[LE0]] : !fir.ref<i32>
! CHECK:         %[[L1:.*]] = fir.convert %[[L1_VAL]] : (i32) -> index
! CHECK:         %[[CONE2:.*]] = arith.constant 1 : index
! CHECK:         %[[LE1:.*]] = fir.coordinate_of %[[LOWER_DECL]]#1, %[[CONE2]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:         %[[L2_VAL:.*]] = fir.load %[[LE1]] : !fir.ref<i32>
! CHECK:         %[[L2:.*]] = fir.convert %[[L2_VAL]] : (i32) -> index
! CHECK:         %[[SS:.*]] = fir.shape_shift %[[L1]], %[[S1]], %[[L2]], %[[S2]] : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:         %[[BOX:.*]] = fir.embox %[[PTR]](%[[SS]]) : (!fir.ptr<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
! CHECK:         fir.store %[[BOX]] to %[[FPTR_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
  call c_f_pointer(cptr, fptr, shape, lower)
end subroutine dynamic_shape_lower

! CHECK-LABEL: func.func @_QPdynamic_shape_lower_2(
subroutine dynamic_shape_lower_2(cptr, fpr, shape, lower, n)
  use iso_c_binding
  type(c_ptr)  :: cptr
  real, pointer :: fptr(:, :)
  integer :: n
  integer :: shape(n)
  integer :: lower(n)
! CHECK:         %[[FPTR_ALLOCA:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
! CHECK:         %[[FPTR_DECL:.*]]:2 = hlfir.declare %[[FPTR_ALLOCA]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFdynamic_shape_lower_2Efptr"}
! CHECK:         %[[LOWER_DECL:.*]]:2 = hlfir.declare %{{.*}}({{.*}}) {{.*}} {uniq_name = "_QFdynamic_shape_lower_2Elower"}
! CHECK:         %[[SHAPE_DECL:.*]]:2 = hlfir.declare %{{.*}}({{.*}}) {{.*}} {uniq_name = "_QFdynamic_shape_lower_2Eshape"}
! CHECK:         %[[ADDR:.*]] = fir.coordinate_of %{{.*}}#0, __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
! CHECK:         %[[ADDR_VAL:.*]] = fir.load %[[ADDR]] : !fir.ref<i64>
! CHECK:         %[[PTR:.*]] = fir.convert %[[ADDR_VAL]] : (i64) -> !fir.ptr<!fir.array<?x?xf32>>
! CHECK:         %[[CZERO:.*]] = arith.constant 0 : index
! CHECK:         %[[SE0:.*]] = fir.coordinate_of %[[SHAPE_DECL]]#1, %[[CZERO]] : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:         %[[S1_VAL:.*]] = fir.load %[[SE0]] : !fir.ref<i32>
! CHECK:         %[[S1:.*]] = fir.convert %[[S1_VAL]] : (i32) -> index
! CHECK:         %[[CONE:.*]] = arith.constant 1 : index
! CHECK:         %[[SE1:.*]] = fir.coordinate_of %[[SHAPE_DECL]]#1, %[[CONE]] : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:         %[[S2_VAL:.*]] = fir.load %[[SE1]] : !fir.ref<i32>
! CHECK:         %[[S2:.*]] = fir.convert %[[S2_VAL]] : (i32) -> index
! CHECK:         %[[CZERO2:.*]] = arith.constant 0 : index
! CHECK:         %[[LE0:.*]] = fir.coordinate_of %[[LOWER_DECL]]#1, %[[CZERO2]] : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:         %[[L1_VAL:.*]] = fir.load %[[LE0]] : !fir.ref<i32>
! CHECK:         %[[L1:.*]] = fir.convert %[[L1_VAL]] : (i32) -> index
! CHECK:         %[[CONE2:.*]] = arith.constant 1 : index
! CHECK:         %[[LE1:.*]] = fir.coordinate_of %[[LOWER_DECL]]#1, %[[CONE2]] : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:         %[[L2_VAL:.*]] = fir.load %[[LE1]] : !fir.ref<i32>
! CHECK:         %[[L2:.*]] = fir.convert %[[L2_VAL]] : (i32) -> index
! CHECK:         %[[SS:.*]] = fir.shape_shift %[[L1]], %[[S1]], %[[L2]], %[[S2]] : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:         %[[BOX:.*]] = fir.embox %[[PTR]](%[[SS]]) : (!fir.ptr<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
! CHECK:         fir.store %[[BOX]] to %[[FPTR_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
  call c_f_pointer(cptr, fptr, shape, lower)
end subroutine dynamic_shape_lower_2
