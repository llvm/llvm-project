! Test lowering of binary intrinsic operations to HLFIR
! RUN: bbc -emit-hlfir -o - %s 2>&1 | FileCheck %s

subroutine int_add(x, y, z)
 integer :: x, y, z
 x = y + z
end subroutine
! CHECK-LABEL: func.func @_QPint_add(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}}y"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %{{.*}}z"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i32>
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
! CHECK:  %[[VAL_8:.*]] = arith.addi %[[VAL_6]], %[[VAL_7]] : i32

subroutine real_add(x, y, z)
 real :: x, y, z
 x = y + z
end subroutine
! CHECK-LABEL: func.func @_QPreal_add(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}}y"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %{{.*}}z"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<f32>
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<f32>
! CHECK:  %[[VAL_8:.*]] = arith.addf %[[VAL_6]], %[[VAL_7]] fastmath<contract> : f32

subroutine complex_add(x, y, z)
 complex :: x, y, z
 x = y + z
end subroutine
! CHECK-LABEL: func.func @_QPcomplex_add(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}}y"} : (!fir.ref<!fir.complex<4>>) -> (!fir.ref<!fir.complex<4>>, !fir.ref<!fir.complex<4>>)
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %{{.*}}z"} : (!fir.ref<!fir.complex<4>>) -> (!fir.ref<!fir.complex<4>>, !fir.ref<!fir.complex<4>>)
! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<!fir.complex<4>>
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<!fir.complex<4>>
! CHECK:  %[[VAL_8:.*]] = fir.addc %[[VAL_6]], %[[VAL_7]] : !fir.complex<4>

subroutine int_sub(x, y, z)
 integer :: x, y, z
 x = y - z
end subroutine
! CHECK-LABEL: func.func @_QPint_sub(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}}y"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %{{.*}}z"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i32>
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
! CHECK:  %[[VAL_8:.*]] = arith.subi %[[VAL_6]], %[[VAL_7]] : i32

subroutine real_sub(x, y, z)
 real :: x, y, z
 x = y - z
end subroutine
! CHECK-LABEL: func.func @_QPreal_sub(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}}y"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %{{.*}}z"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<f32>
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<f32>
! CHECK:  %[[VAL_8:.*]] = arith.subf %[[VAL_6]], %[[VAL_7]] fastmath<contract> : f32

subroutine complex_sub(x, y, z)
 complex :: x, y, z
 x = y - z
end subroutine
! CHECK-LABEL: func.func @_QPcomplex_sub(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}}y"} : (!fir.ref<!fir.complex<4>>) -> (!fir.ref<!fir.complex<4>>, !fir.ref<!fir.complex<4>>)
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %{{.*}}z"} : (!fir.ref<!fir.complex<4>>) -> (!fir.ref<!fir.complex<4>>, !fir.ref<!fir.complex<4>>)
! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<!fir.complex<4>>
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<!fir.complex<4>>
! CHECK:  %[[VAL_8:.*]] = fir.subc %[[VAL_6]], %[[VAL_7]] : !fir.complex<4>

subroutine int_mul(x, y, z)
 integer :: x, y, z
 x = y * z
end subroutine
! CHECK-LABEL: func.func @_QPint_mul(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}}y"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %{{.*}}z"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i32>
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
! CHECK:  %[[VAL_8:.*]] = arith.muli %[[VAL_6]], %[[VAL_7]] : i32

subroutine real_mul(x, y, z)
 real :: x, y, z
 x = y * z
end subroutine
! CHECK-LABEL: func.func @_QPreal_mul(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}}y"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %{{.*}}z"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<f32>
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<f32>
! CHECK:  %[[VAL_8:.*]] = arith.mulf %[[VAL_6]], %[[VAL_7]] fastmath<contract> : f32

subroutine complex_mul(x, y, z)
 complex :: x, y, z
 x = y * z
end subroutine
! CHECK-LABEL: func.func @_QPcomplex_mul(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}}y"} : (!fir.ref<!fir.complex<4>>) -> (!fir.ref<!fir.complex<4>>, !fir.ref<!fir.complex<4>>)
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %{{.*}}z"} : (!fir.ref<!fir.complex<4>>) -> (!fir.ref<!fir.complex<4>>, !fir.ref<!fir.complex<4>>)
! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<!fir.complex<4>>
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<!fir.complex<4>>
! CHECK:  %[[VAL_8:.*]] = fir.mulc %[[VAL_6]], %[[VAL_7]] : !fir.complex<4>

subroutine int_div(x, y, z)
 integer :: x, y, z
 x = y / z
end subroutine
! CHECK-LABEL: func.func @_QPint_div(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}}y"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %{{.*}}z"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i32>
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
! CHECK:  %[[VAL_8:.*]] = arith.divsi %[[VAL_6]], %[[VAL_7]] : i32

subroutine real_div(x, y, z)
 real :: x, y, z
 x = y / z
end subroutine
! CHECK-LABEL: func.func @_QPreal_div(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}}y"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %{{.*}}z"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<f32>
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<f32>
! CHECK:  %[[VAL_8:.*]] = arith.divf %[[VAL_6]], %[[VAL_7]] fastmath<contract> : f32

subroutine complex_div(x, y, z)
 complex :: x, y, z
 x = y / z
end subroutine
! CHECK-LABEL: func.func @_QPcomplex_div(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}}y"} : (!fir.ref<!fir.complex<4>>) -> (!fir.ref<!fir.complex<4>>, !fir.ref<!fir.complex<4>>)
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %{{.*}}z"} : (!fir.ref<!fir.complex<4>>) -> (!fir.ref<!fir.complex<4>>, !fir.ref<!fir.complex<4>>)
! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<!fir.complex<4>>
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<!fir.complex<4>>
! CHECK:  %[[VAL_8:.*]] = fir.extract_value %[[VAL_6]], [0 : index] : (!fir.complex<4>) -> f32
! CHECK:  %[[VAL_9:.*]] = fir.extract_value %[[VAL_6]], [1 : index] : (!fir.complex<4>) -> f32
! CHECK:  %[[VAL_10:.*]] = fir.extract_value %[[VAL_7]], [0 : index] : (!fir.complex<4>) -> f32
! CHECK:  %[[VAL_11:.*]] = fir.extract_value %[[VAL_7]], [1 : index] : (!fir.complex<4>) -> f32
! CHECK:  %[[VAL_12:.*]] = fir.call @__divsc3(%[[VAL_8]], %[[VAL_9]], %[[VAL_10]], %[[VAL_11]]) fastmath<contract> : (f32, f32, f32, f32) -> !fir.complex<4>

subroutine int_power(x, y, z)
  integer :: x, y, z
  x = y**z
end subroutine
! CHECK-LABEL: func.func @_QPint_power(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}}y"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %{{.*}}z"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i32>
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
! CHECK:  %[[VAL_8:.*]] = math.ipowi %[[VAL_6]], %[[VAL_7]] : i32

subroutine real_power(x, y, z)
  real :: x, y, z
  x = y**z
end subroutine
! CHECK-LABEL: func.func @_QPreal_power(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}}y"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %{{.*}}z"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<f32>
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<f32>
! CHECK:  %[[VAL_8:.*]] = math.powf %[[VAL_6]], %[[VAL_7]] fastmath<contract> : f32

subroutine complex_power(x, y, z)
  complex :: x, y, z
  x = y**z
end subroutine
! CHECK-LABEL: func.func @_QPcomplex_power(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}}y"} : (!fir.ref<!fir.complex<4>>) -> (!fir.ref<!fir.complex<4>>, !fir.ref<!fir.complex<4>>)
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %{{.*}}z"} : (!fir.ref<!fir.complex<4>>) -> (!fir.ref<!fir.complex<4>>, !fir.ref<!fir.complex<4>>)
! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<!fir.complex<4>>
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<!fir.complex<4>>
! CHECK:  %[[VAL_8:.*]] = fir.call @cpowf(%[[VAL_6]], %[[VAL_7]]) fastmath<contract> : (!fir.complex<4>, !fir.complex<4>) -> !fir.complex<4>


subroutine real_to_int_power(x, y, z)
  real :: x, y
  integer :: z
  x = y**z
end subroutine
! CHECK-LABEL: func.func @_QPreal_to_int_power(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}}y"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %{{.*}}z"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<f32>
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
! CHECK:  %[[VAL_8:.*]] = math.fpowi %[[VAL_6]], %[[VAL_7]] fastmath<contract> : f32, i32

subroutine complex_to_int_power(x, y, z)
  complex :: x, y
  integer :: z
  x = y**z
end subroutine
! CHECK-LABEL: func.func @_QPcomplex_to_int_power(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}}y"} : (!fir.ref<!fir.complex<4>>) -> (!fir.ref<!fir.complex<4>>, !fir.ref<!fir.complex<4>>)
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %{{.*}}z"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<!fir.complex<4>>
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
! CHECK:  %[[VAL_8:.*]] = fir.call @_FortranAcpowi(%[[VAL_6]], %[[VAL_7]]) fastmath<contract> : (!fir.complex<4>, i32) -> !fir.complex<4>

subroutine extremum(c, n, l)
  integer(8), intent(in) :: l
  integer(8) :: n
  character(l) :: c
  ! evaluate::Extremum is created by semantics while analyzing LEN().
  n = len(c, 8)
end subroutine
! CHECK-LABEL: func.func @_QPextremum(
! CHECK:  hlfir.declare {{.*}}c
! CHECK:  %[[VAL_11:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_12:.*]] = fir.load %{{.*}} : !fir.ref<i64>
! CHECK:  %[[VAL_13:.*]] = arith.cmpi sgt, %[[VAL_11]], %[[VAL_12]] : i64
! CHECK:  arith.select %[[VAL_13]], %[[VAL_11]], %[[VAL_12]] : i64

subroutine cmp_int(l, x, y)
  logical :: l
  integer :: x, y
  l = x .eq. y
end subroutine
! CHECK-LABEL: func.func @_QPcmp_int(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare {{.*}}x"
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare {{.*}}y"
! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i32>
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
! CHECK:  %[[VAL_8:.*]] = arith.cmpi eq, %[[VAL_6]], %[[VAL_7]] : i32

subroutine cmp_int_2(l, x, y)
  logical :: l
  integer :: x, y
  l = x .ne. y
! CHECK:  arith.cmpi ne
  l = x .gt. y
! CHECK:  arith.cmpi sgt
  l = x .ge. y
! CHECK:  arith.cmpi sge
  l = x .lt. y
! CHECK:  arith.cmpi slt
  l = x .le. y
! CHECK:  arith.cmpi sle
end subroutine

subroutine cmp_real(l, x, y)
  logical :: l
  real :: x, y
  l = x .eq. y
end subroutine
! CHECK-LABEL: func.func @_QPcmp_real(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare {{.*}}x"
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare {{.*}}y"
! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<f32>
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<f32>
! CHECK:  %[[VAL_8:.*]] = arith.cmpf oeq, %[[VAL_6]], %[[VAL_7]] : f32

subroutine cmp_real_2(l, x, y)
  logical :: l
  real :: x, y
  l = x .ne. y
! CHECK:  arith.cmpf une
  l = x .gt. y
! CHECK:  arith.cmpf ogt
  l = x .ge. y
! CHECK:  arith.cmpf oge
  l = x .lt. y
! CHECK:  arith.cmpf olt
  l = x .le. y
! CHECK:  arith.cmpf ole
end subroutine

subroutine cmp_cmplx(l, x, y)
  logical :: l
  complex :: x, y
  l = x .eq. y
end subroutine
! CHECK-LABEL: func.func @_QPcmp_cmplx(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare {{.*}}x"
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare {{.*}}y"
! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<!fir.complex<4>>
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<!fir.complex<4>>
! CHECK:  %[[VAL_8:.*]] = fir.cmpc "oeq", %[[VAL_6]], %[[VAL_7]] : !fir.complex<4>

subroutine cmp_char(l, x, y)
  logical :: l
  character(*) :: x, y
  l = x .eq. y
end subroutine
! CHECK-LABEL: func.func @_QPcmp_char(
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %{{.*}} typeparams %[[VAL_4:.*]]#1 {uniq_name = "_QFcmp_charEx"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:  %[[VAL_7:.*]]:2 = hlfir.declare %{{.*}} typeparams %[[VAL_6:.*]]#1 {uniq_name = "_QFcmp_charEy"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:  %[[VAL_8:.*]] = fir.convert %[[VAL_5]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:  %[[VAL_9:.*]] = fir.convert %[[VAL_7]]#1 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:  %[[VAL_10:.*]] = fir.convert %[[VAL_4]]#1 : (index) -> i64
! CHECK:  %[[VAL_11:.*]] = fir.convert %[[VAL_6]]#1 : (index) -> i64
! CHECK:  %[[VAL_12:.*]] = fir.call @_FortranACharacterCompareScalar1(%[[VAL_8]], %[[VAL_9]], %[[VAL_10]], %[[VAL_11]]) fastmath<contract> : (!fir.ref<i8>, !fir.ref<i8>, i64, i64) -> i32
! CHECK:  %[[VAL_13:.*]] = arith.constant 0 : i32
! CHECK:  %[[VAL_14:.*]] = arith.cmpi eq, %[[VAL_12]], %[[VAL_13]] : i32

subroutine logical_and(x, y, z)
  logical :: x, y, z
  x = y.and.z
end subroutine
! CHECK-LABEL: func.func @_QPlogical_and(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}}y"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %{{.*}}z"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:  %[[VAL_8:.*]] = fir.convert %[[VAL_6]] : (!fir.logical<4>) -> i1
! CHECK:  %[[VAL_9:.*]] = fir.convert %[[VAL_7]] : (!fir.logical<4>) -> i1
! CHECK:  %[[VAL_10:.*]] = arith.andi %[[VAL_8]], %[[VAL_9]] : i1

subroutine logical_or(x, y, z)
  logical :: x, y, z
  x = y.or.z
end subroutine
! CHECK-LABEL: func.func @_QPlogical_or(
! CHECK:  %[[VAL_10:.*]] = arith.ori

subroutine logical_eqv(x, y, z)
  logical :: x, y, z
  x = y.eqv.z
end subroutine
! CHECK-LABEL: func.func @_QPlogical_eqv(
! CHECK:  %[[VAL_10:.*]] = arith.cmpi eq

subroutine logical_neqv(x, y, z)
  logical :: x, y, z
  x = y.neqv.z
end subroutine
! CHECK-LABEL: func.func @_QPlogical_neqv(
! CHECK:  %[[VAL_10:.*]] = arith.cmpi ne

subroutine cmplx_ctor(z, x, y)
  complex :: z
  real :: x, y
  z = cmplx(x, y)
end subroutine
! CHECK-LABEL: func.func @_QPcmplx_ctor(
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %{{.*}}x"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}}y"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<f32>
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<f32>
! CHECK:  %[[VAL_8:.*]] = fir.undefined !fir.complex<4>
! CHECK:  %[[VAL_9:.*]] = fir.insert_value %[[VAL_8]], %[[VAL_6]], [0 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
! CHECK:  %[[VAL_10:.*]] = fir.insert_value %[[VAL_9]], %[[VAL_7]], [1 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>

subroutine cmplx_ctor_2(z, x)
  complex(8) :: z
  real(8) :: x
  z = cmplx(x, 1._8, kind=8)
end subroutine
! CHECK-LABEL: func.func @_QPcmplx_ctor_2(
! CHECK:  %[[VAL_2:.*]]:2 = hlfir.declare %{{.*}}x"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<f64>
! CHECK:  %[[VAL_5:.*]] = arith.constant 1.000000e+00 : f64
! CHECK:  %[[VAL_6:.*]] = fir.undefined !fir.complex<8>
! CHECK:  %[[VAL_7:.*]] = fir.insert_value %[[VAL_6]], %[[VAL_4]], [0 : index] : (!fir.complex<8>, f64) -> !fir.complex<8>
! CHECK:  %[[VAL_8:.*]] = fir.insert_value %[[VAL_7]], %[[VAL_5]], [1 : index] : (!fir.complex<8>, f64) -> !fir.complex<8>
