! Test lowering of binary intrinsic operations to HLFIR
! RUN: bbc -emit-fir -hlfir -o - %s 2>&1 | FileCheck %s

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
! CHECK:  %[[VAL_8:.*]] = fir.divc %[[VAL_6]], %[[VAL_7]] : !fir.complex<4>


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
! CHECK:  %[[VAL_8:.*]] = fir.convert %[[VAL_6]] : (!fir.complex<4>) -> complex<f32>
! CHECK:  %[[VAL_9:.*]] = fir.convert %[[VAL_7]] : (!fir.complex<4>) -> complex<f32>
! CHECK:  %[[VAL_10:.*]] = complex.pow %[[VAL_8]], %[[VAL_9]] : complex<f32>
! CHECK:  %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (complex<f32>) -> !fir.complex<4>

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
! CHECK:  %[[VAL_8:.*]] = fir.call @llvm.powi.f32.i32(%[[VAL_6]], %[[VAL_7]]) fastmath<contract> : (f32, i32) -> f32

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
