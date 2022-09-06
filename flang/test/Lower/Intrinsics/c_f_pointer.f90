! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! Test intrinsic module procedure c_f_pointer

! CHECK-LABEL: func.func @_QPtest_scalar(
! CHECK-SAME:                            %[[VAL_0:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>> {fir.bindc_name = "cptr"},
! CHECK-SAME:                            %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.ptr<f32>>> {fir.bindc_name = "fptr"}) {
! CHECK:         %[[VAL_2:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_2]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_4:.*]] = fir.load %[[VAL_3]] : !fir.ref<i64>
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i64) -> !fir.ptr<f32>
! CHECK:         %[[VAL_6:.*]] = fir.embox %[[VAL_5]] : (!fir.ptr<f32>) -> !fir.box<!fir.ptr<f32>>
! CHECK:         fir.store %[[VAL_6]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<f32>>>
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
! CHECK:         %[[VAL_65:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_66:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_65]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_67:.*]] = fir.load %[[VAL_66]] : !fir.ref<i64>
! CHECK:         %[[VAL_68:.*]] = fir.convert %[[VAL_67]] : (i64) -> !fir.ptr<!fir.array<?x?xf32>>
! CHECK:         %[[VAL_69:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_70:.*]] = fir.coordinate_of %[[VAL_53:.*]], %[[VAL_69]] : (!fir.heap<!fir.array<2xi32>>, index) -> !fir.ref<i32>
! CHECK:         %[[VAL_71:.*]] = fir.load %[[VAL_70]] : !fir.ref<i32>
! CHECK:         %[[VAL_72:.*]] = fir.convert %[[VAL_71]] : (i32) -> index
! CHECK:         %[[VAL_73:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_74:.*]] = fir.coordinate_of %[[VAL_53]], %[[VAL_73]] : (!fir.heap<!fir.array<2xi32>>, index) -> !fir.ref<i32>
! CHECK:         %[[VAL_75:.*]] = fir.load %[[VAL_74]] : !fir.ref<i32>
! CHECK:         %[[VAL_76:.*]] = fir.convert %[[VAL_75]] : (i32) -> index
! CHECK:         %[[VAL_77:.*]] = fir.shape %[[VAL_72]], %[[VAL_76]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_78:.*]] = fir.embox %[[VAL_68]](%[[VAL_77]]) : (!fir.ptr<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
! CHECK:         fir.store %[[VAL_78]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
! CHECK:         return
! CHECK:       }

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
! CHECK:         %[[VAL_2:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_2]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_4:.*]] = fir.load %[[VAL_3]] : !fir.ref<i64>
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i64) -> !fir.ptr<!fir.char<1,10>>
! CHECK:         %[[VAL_6:.*]] = fir.embox %[[VAL_5]] : (!fir.ptr<!fir.char<1,10>>) -> !fir.box<!fir.ptr<!fir.char<1,10>>>
! CHECK:         fir.store %[[VAL_6]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,10>>>>
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
! CHECK:         %[[VAL_7:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:         %[[VAL_8:.*]] = arith.constant 0 : i32
! CHECK:         %[[VAL_9:.*]] = arith.cmpi sgt, %[[VAL_7]], %[[VAL_8]] : i32
! CHECK:         %[[VAL_10:.*]] = arith.select %[[VAL_9]], %[[VAL_7]], %[[VAL_8]] : i32
! CHECK:         %[[VAL_70:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
! CHECK:         %[[VAL_71:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_70]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.field) -> !fir.ref<i64>
! CHECK:         %[[VAL_72:.*]] = fir.load %[[VAL_71]] : !fir.ref<i64>
! CHECK:         %[[VAL_73:.*]] = fir.convert %[[VAL_72]] : (i64) -> !fir.ptr<!fir.array<?x?x!fir.char<1,?>>>
! CHECK:         %[[VAL_74:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_75:.*]] = fir.coordinate_of %[[VAL_58:.*]], %[[VAL_74]] : (!fir.heap<!fir.array<2xi32>>, index) -> !fir.ref<i32>
! CHECK:         %[[VAL_76:.*]] = fir.load %[[VAL_75]] : !fir.ref<i32>
! CHECK:         %[[VAL_77:.*]] = fir.convert %[[VAL_76]] : (i32) -> index
! CHECK:         %[[VAL_78:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_79:.*]] = fir.coordinate_of %[[VAL_58]], %[[VAL_78]] : (!fir.heap<!fir.array<2xi32>>, index) -> !fir.ref<i32>
! CHECK:         %[[VAL_80:.*]] = fir.load %[[VAL_79]] : !fir.ref<i32>
! CHECK:         %[[VAL_81:.*]] = fir.convert %[[VAL_80]] : (i32) -> index
! CHECK:         %[[VAL_82:.*]] = fir.shape %[[VAL_77]], %[[VAL_81]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_83:.*]] = fir.embox %[[VAL_73]](%[[VAL_82]]) typeparams %[[VAL_10]] : (!fir.ptr<!fir.array<?x?x!fir.char<1,?>>>, !fir.shape<2>, i32) -> !fir.box<!fir.ptr<!fir.array<?x?x!fir.char<1,?>>>>
! CHECK:         fir.store %[[VAL_83]] to %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x!fir.char<1,?>>>>>
! CHECK:         return
! CHECK:       }

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
! CHECK:         %[[VAL_7:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_8:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_7]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:         %[[VAL_9:.*]] = fir.load %[[VAL_8]] : !fir.ref<i32>
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
! CHECK:         %[[VAL_11:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_12:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_11]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:         %[[VAL_13:.*]] = fir.load %[[VAL_12]] : !fir.ref<i32>
! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i32) -> index
! CHECK:         %[[VAL_15:.*]] = fir.shape %[[VAL_10]], %[[VAL_14]] : (index, index) -> !fir.shape<2>
  call c_f_pointer(cptr, fptr, shape)
end subroutine

! CHECK-LABEL: func.func @_QPdynamic_shape_size_2(
subroutine dynamic_shape_size_2(cptr, fptr, shape, n)
  use iso_c_binding
  type(c_ptr)  :: cptr
  real, pointer :: fptr(:, :)
  integer :: n
  integer :: shape(n)
! CHECK:         %[[VAL_8:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_9:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_8]] : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:         %[[VAL_10:.*]] = fir.load %[[VAL_9]] : !fir.ref<i32>
! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i32) -> index
! CHECK:         %[[VAL_12:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_13:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_12]] : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:         %[[VAL_14:.*]] = fir.load %[[VAL_13]] : !fir.ref<i32>
! CHECK:         %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i32) -> index
! CHECK:         %[[VAL_16:.*]] = fir.shape %[[VAL_11]], %[[VAL_15]] : (index, index) -> !fir.shape<2>
  call c_f_pointer(cptr, fptr, shape)
end subroutine
