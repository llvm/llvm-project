! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir %s -o - | FileCheck %s

! Test intrinsic module procedure c_f_pointer

! CHECK-LABEL: func.func @_QPtest_scalar(
! CHECK-SAME:                            %[[VAL_0:.*]]: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>> {fir.bindc_name = "cptr"},
! CHECK-SAME:                            %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.ptr<f32>>> {fir.bindc_name = "fptr"}) {
! CHECK:         %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_0]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
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
! CHECK:         %[[VAL_66:.*]] = fir.coordinate_of %[[VAL_0]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
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
! CHECK:         %[[VAL_3:.*]] = fir.coordinate_of %[[VAL_0]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
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
! CHECK:         %[[VAL_71:.*]] = fir.coordinate_of %[[VAL_0]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
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

! CHECK-LABEL: func.func @_QPdynamic_shape_lower(
subroutine dynamic_shape_lower(cptr, fpr, shape, lower)
  use iso_c_binding
  type(c_ptr)  :: cptr
  real, pointer :: fptr(:, :)
  integer :: n
  integer :: shape(:)
  integer :: lower(:)
! CHECK: %[[C_0:.*]] = arith.constant 0 : index
! CHECK: %[[VAL_2:.*]] = fir.shape %[[C_0]], %[[C_0]] : (index, index) -> !fir.shape<2>
! CHECK: %[[VAL_3:.*]] = fir.embox %[[VAL_1:.*]](%[[VAL_2]]) : (!fir.ptr<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
! CHECK: fir.store %[[VAL_3]] to %[[VAL_0:.*]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
! CHECK: %[[VAL_5:.*]] = fir.coordinate_of %[[ARG_0:.*]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
! CHECK: %[[VAL_6:.*]] = fir.load %[[VAL_5]] : !fir.ref<i64>
! CHECK: %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i64) -> !fir.ptr<!fir.array<?x?xf32>>
! CHECK: %[[C_0:.*]]_0 = arith.constant 0 : index
! CHECK: %[[VAL_8:.*]] = fir.coordinate_of %[[ARG_2:.*]], %[[C_0]]_0 : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK: %[[VAL_9:.*]] = fir.load %[[VAL_8]] : !fir.ref<i32>
! CHECK: %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
! CHECK: %[[C_1:.*]] = arith.constant 1 : index
! CHECK: %[[VAL_11:.*]] = fir.coordinate_of %[[ARG_2:.*]], %[[C_1]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK: %[[VAL_12:.*]] = fir.load %[[VAL_11]] : !fir.ref<i32>
! CHECK: %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i32) -> index
! CHECK: %[[C_0:.*]]_1 = arith.constant 0 : index
! CHECK: %[[VAL_14:.*]] = fir.coordinate_of %[[ARG_3:.*]], %[[C_0]]_1 : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK: %[[VAL_15:.*]] = fir.load %[[VAL_14]] : !fir.ref<i32>
! CHECK: %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i32) -> index
! CHECK: %[[C_1:.*]]_2 = arith.constant 1 : index
! CHECK: %[[VAL_17:.*]] = fir.coordinate_of %[[ARG_3:.*]], %[[C_1]]_2 : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK: %[[VAL_18:.*]] = fir.load %[[VAL_17]] : !fir.ref<i32>
! CHECK: %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i32) -> index
! CHECK: %[[VAL_20:.*]] = fir.shape_shift %[[VAL_16]], %[[VAL_10]], %[[VAL_19]], %[[VAL_13]] : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK: %[[VAL_21:.*]] = fir.embox %[[VAL_7]](%[[VAL_20]]) : (!fir.ptr<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
! CHECK: fir.store %[[VAL_21:.*]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
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
!CHECK: %[[C_0:.*]] = arith.constant 0 : index
!CHECK: %[[VAL_2:.*]] = fir.shape %[[C_0]], %[[C_0]] : (index, index) -> !fir.shape<2>
!CHECK: %[[VAL_3:.*]] = fir.embox %[[ARG1:.*]](%[[VAL_2]]) : (!fir.ptr<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
!CHECK: fir.store %[[VAL_3]] to %[[VAL_0:.*]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
!CHECK: %[[VAL_4:.*]] = fir.coordinate_of %[[ARG_0:.*]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
!CHECK: %[[VAL_5:.*]] = fir.load %[[VAL_4]] : !fir.ref<i64>
!CHECK: %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i64) -> !fir.ptr<!fir.array<?x?xf32>>
!CHECK: %[[C_0:.*]]_0 = arith.constant 0 : index
!CHECK: %[[VAL_7:.*]] = fir.coordinate_of %[[ARG_2:.*]], %[[C_0]]_0 : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<i32>
!CHECK: %[[VAL_8:.*]] = fir.load %[[VAL_7]] : !fir.ref<i32>
!CHECK: %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i32) -> index
!CHECK: %[[C_1:.*]] = arith.constant 1 : index
!CHECK: %[[VAL_10:.*]] = fir.coordinate_of %[[ARG_2]], %[[C_1]] : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<i32>
!CHECK: %[[VAL_11:.*]] = fir.load %[[VAL_10]] : !fir.ref<i32>
!CHECK: %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i32) -> index
!CHECK: %[[C_0:.*]]_1 = arith.constant 0 : index
!CHECK: %[[VAL_13:.*]] = fir.coordinate_of %[[ARG_3:.*]], %[[C_0]]_1 : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<i32>
!CHECK: %[[VAL_14:.*]] = fir.load %[[VAL_13]] : !fir.ref<i32>
!CHECK: %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i32) -> index
!CHECK: %[[C_1:.*]]_2 = arith.constant 1 : index
!CHECK: %[[VAL_16:.*]] = fir.coordinate_of %[[ARG_3]], %[[C_1]]_2 : (!fir.ref<!fir.array<?xi32>>, index) -> !fir.ref<i32>
!CHECK: %[[VAL_17:.*]] = fir.load %[[VAL_16]] : !fir.ref<i32>
!CHECK: %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i32) -> index
!CHECK: %[[VAL_19:.*]] = fir.shape_shift %[[VAL_15]], %[[VAL_9]], %[[VAL_18]], %[[VAL_12]] : (index, index, index, index) -> !fir.shapeshift<2>
!CHECK: %[[VAL_20:.*]] = fir.embox %[[VAL_6]](%[[VAL_19]]) : (!fir.ptr<!fir.array<?x?xf32>>, !fir.shapeshift<2>)
!CHECK: fir.store %[[VAL_20]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
  call c_f_pointer(cptr, fptr, shape, lower)
end subroutine dynamic_shape_lower_2
