! Test lowering of intrinsic assignments to HLFIR
! RUN: bbc -emit-fir -hlfir -o - %s 2>&1 | FileCheck %s

! -----------------------------------------------------------------------------
!     Test assignments with scalar variable LHS and RHS
! -----------------------------------------------------------------------------

subroutine scalar_int(x, y)
  integer :: x, y
  x = y
end subroutine
! CHECK-LABEL: func.func @_QPscalar_int(
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare {{.*}}  {uniq_name = "_QFscalar_intEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare {{.*}}  {uniq_name = "_QFscalar_intEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  hlfir.assign %[[VAL_3]]#0 to %[[VAL_2]]#0 : !fir.ref<i32>, !fir.ref<i32>

subroutine scalar_logical(x, y)
  logical :: x, y
  x = y
end subroutine
! CHECK-LABEL: func.func @_QPscalar_logical(
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare {{.*}}  {uniq_name = "_QFscalar_logicalEx"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare {{.*}}  {uniq_name = "_QFscalar_logicalEy"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:  hlfir.assign %[[VAL_3]]#0 to %[[VAL_2]]#0 : !fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>

subroutine scalar_real(x, y)
  real :: x, y
  x = y
end subroutine
! CHECK-LABEL: func.func @_QPscalar_real(
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare {{.*}}  {uniq_name = "_QFscalar_realEx"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare {{.*}}  {uniq_name = "_QFscalar_realEy"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:  hlfir.assign %[[VAL_3]]#0 to %[[VAL_2]]#0 : !fir.ref<f32>, !fir.ref<f32>

subroutine scalar_complex(x, y)
  complex :: x, y
  x = y
end subroutine
! CHECK-LABEL: func.func @_QPscalar_complex(
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare {{.*}}  {uniq_name = "_QFscalar_complexEx"} : (!fir.ref<!fir.complex<4>>) -> (!fir.ref<!fir.complex<4>>, !fir.ref<!fir.complex<4>>)
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare {{.*}}  {uniq_name = "_QFscalar_complexEy"} : (!fir.ref<!fir.complex<4>>) -> (!fir.ref<!fir.complex<4>>, !fir.ref<!fir.complex<4>>)
! CHECK:  hlfir.assign %[[VAL_3]]#0 to %[[VAL_2]]#0 : !fir.ref<!fir.complex<4>>, !fir.ref<!fir.complex<4>>

subroutine scalar_character(x, y)
  character(*) :: x, y
  x = y
end subroutine
! CHECK-LABEL: func.func @_QPscalar_character(
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare {{.*}}  {uniq_name = "_QFscalar_characterEx"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare {{.*}}  {uniq_name = "_QFscalar_characterEy"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:  hlfir.assign %[[VAL_5]]#0 to %[[VAL_3]]#0 : !fir.boxchar<1>, !fir.boxchar<1>

! -----------------------------------------------------------------------------
!     Test assignments with scalar variable LHS and expression RHS
! -----------------------------------------------------------------------------

subroutine scalar_int_2(x)
  integer :: x
  x = 42
end subroutine
! CHECK-LABEL: func.func @_QPscalar_int_2(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare {{.*}}  {uniq_name = "_QFscalar_int_2Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  %[[VAL_2:.*]] = arith.constant 42 : i32
! CHECK:  hlfir.assign %[[VAL_2]] to %[[VAL_1]]#0 : i32, !fir.ref<i32>

subroutine scalar_logical_2(x)
  logical :: x
  x = .true.
end subroutine
! CHECK-LABEL: func.func @_QPscalar_logical_2(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare {{.*}}  {uniq_name = "_QFscalar_logical_2Ex"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:  %[[VAL_2:.*]] = arith.constant true
! CHECK:  hlfir.assign %[[VAL_2]] to %[[VAL_1]]#0 : i1, !fir.ref<!fir.logical<4>>

subroutine scalar_real_2(x)
  real :: x
  x = 3.14
end subroutine
! CHECK-LABEL: func.func @_QPscalar_real_2(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare {{.*}}  {uniq_name = "_QFscalar_real_2Ex"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:  %[[VAL_2:.*]] = arith.constant 3.140000e+00 : f32
! CHECK:  hlfir.assign %[[VAL_2]] to %[[VAL_1]]#0 : f32, !fir.ref<f32>

subroutine scalar_complex_2(x)
  complex :: x
  x = (1., -1.)
end subroutine
! CHECK-LABEL: func.func @_QPscalar_complex_2(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare {{.*}}  {uniq_name = "_QFscalar_complex_2Ex"} : (!fir.ref<!fir.complex<4>>) -> (!fir.ref<!fir.complex<4>>, !fir.ref<!fir.complex<4>>)
! CHECK:  %[[VAL_2:.*]] = arith.constant 1.000000e+00 : f32
! CHECK:  %[[VAL_3:.*]] = arith.constant -1.000000e+00 : f32
! CHECK:  %[[VAL_4:.*]] = fir.undefined !fir.complex<4>
! CHECK:  %[[VAL_5:.*]] = fir.insert_value %[[VAL_4]], %[[VAL_2]], [0 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
! CHECK:  %[[VAL_6:.*]] = fir.insert_value %[[VAL_5]], %[[VAL_3]], [1 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
! CHECK:  hlfir.assign %[[VAL_6]] to %[[VAL_1]]#0 : !fir.complex<4>, !fir.ref<!fir.complex<4>>

subroutine scalar_character_2(x)
  character(*) :: x
  x = "hello"
end subroutine
! CHECK-LABEL: func.func @_QPscalar_character_2(
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFscalar_character_2Ex"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare {{.*}} {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQcl.68656C6C6F"} : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
! CHECK:  hlfir.assign %[[VAL_5]]#0 to %[[VAL_2]]#0 : !fir.ref<!fir.char<1,5>>, !fir.boxchar<1>

! -----------------------------------------------------------------------------
!     Test assignments with array variable LHS and RHS
! -----------------------------------------------------------------------------

subroutine array(x, y)
  integer :: x(:), y(100)
  x = y
end subroutine
! CHECK-LABEL: func.func @_QParray(
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare {{.*}}  {uniq_name = "_QFarrayEx"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare {{.*}}  {uniq_name = "_QFarrayEy"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
! CHECK:  hlfir.assign %[[VAL_5]]#0 to %[[VAL_2]]#0 : !fir.ref<!fir.array<100xi32>>, !fir.box<!fir.array<?xi32>>

subroutine array_lbs(x, y)
  logical :: x(2:21), y(3:22)
  x = y
end subroutine
! CHECK-LABEL: func.func @_QParray_lbs(
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare {{.*}}  {uniq_name = "_QFarray_lbsEx"} : (!fir.ref<!fir.array<20x!fir.logical<4>>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<20x!fir.logical<4>>>, !fir.ref<!fir.array<20x!fir.logical<4>>>)
! CHECK:  %[[VAL_9:.*]]:2 = hlfir.declare {{.*}}  {uniq_name = "_QFarray_lbsEy"} : (!fir.ref<!fir.array<20x!fir.logical<4>>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<20x!fir.logical<4>>>, !fir.ref<!fir.array<20x!fir.logical<4>>>)
! CHECK:  hlfir.assign %[[VAL_9]]#0 to %[[VAL_5]]#0 : !fir.box<!fir.array<20x!fir.logical<4>>>, !fir.box<!fir.array<20x!fir.logical<4>>>


subroutine array_character(x, y)
  character(*) :: x(10), y(10)
  x = y
end subroutine
! CHECK-LABEL: func.func @_QParray_character(
! CHECK:  %[[VAL_6:.*]]:2 = hlfir.declare {{.*}}  {uniq_name = "_QFarray_characterEx"} : (!fir.ref<!fir.array<10x!fir.char<1,?>>>, !fir.shape<1>, index) -> (!fir.box<!fir.array<10x!fir.char<1,?>>>, !fir.ref<!fir.array<10x!fir.char<1,?>>>)
! CHECK:  %[[VAL_11:.*]]:2 = hlfir.declare {{.*}}  {uniq_name = "_QFarray_characterEy"} : (!fir.ref<!fir.array<10x!fir.char<1,?>>>, !fir.shape<1>, index) -> (!fir.box<!fir.array<10x!fir.char<1,?>>>, !fir.ref<!fir.array<10x!fir.char<1,?>>>)
! CHECK:  hlfir.assign %[[VAL_11]]#0 to %[[VAL_6]]#0 : !fir.box<!fir.array<10x!fir.char<1,?>>>, !fir.box<!fir.array<10x!fir.char<1,?>>>

subroutine array_pointer(x, y)
  real, pointer :: x(:), y(:)
  x = y
end subroutine
! CHECK-LABEL: func.func @_QParray_pointer(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %{{.*}}Ex
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %{{.*}}Ey
! CHECK:  %[[VAL_3:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:  hlfir.assign %[[VAL_3]] to %[[VAL_4]] : !fir.box<!fir.ptr<!fir.array<?xf32>>>, !fir.box<!fir.ptr<!fir.array<?xf32>>>

! -----------------------------------------------------------------------------
!     Test assignments with array LHS and scalar RHS
! -----------------------------------------------------------------------------

subroutine array_scalar(x, y)
  integer :: x(100), y
  x = y
end subroutine
! CHECK-LABEL: func.func @_QParray_scalar(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare {{.*}}  {uniq_name = "_QFarray_scalarEx"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare {{.*}}  {uniq_name = "_QFarray_scalarEy"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  hlfir.assign %[[VAL_5]]#0 to %[[VAL_4]]#0 : !fir.ref<i32>, !fir.ref<!fir.array<100xi32>>
