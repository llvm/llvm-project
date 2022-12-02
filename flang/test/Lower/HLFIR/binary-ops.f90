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
