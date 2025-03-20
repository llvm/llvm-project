! RUN: bbc %s -o - | FileCheck %s

program p
  print*,a(),b(),c()
contains
  function a()
    integer,volatile::a
    a=1
  end function
  function b() result(r)
    integer,volatile::r
    r=2
  end function
  function c() result(r)
    volatile::r
    r=3
  end function
end program


! CHECK-LABEL:   func.func @_QQmain() attributes {{.+}} {
! CHECK:           %[[VAL_0:.*]] = arith.constant 4 : i32
! CHECK:           %[[VAL_1:.*]] = arith.constant 6 : i32
! CHECK:           %[[VAL_2:.*]] = fir.address_of
! CHECK:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.char<1,{{.+}}>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_4:.*]] = fir.call @_QFPa() fastmath<contract> : () -> i32
! CHECK:           %[[VAL_5:.*]] = fir.call @_QFPb() fastmath<contract> : () -> i32
! CHECK:           %[[VAL_6:.*]] = fir.call @_QFPc() fastmath<contract> : () -> f32
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFPa() -> i32 attributes {{.+}} {
! CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "a", uniq_name = "_QFFaEa"}
! CHECK:           %[[VAL_2:.*]] = fir.volatile_cast %[[VAL_1]] : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {{.+}} : (!fir.ref<i32, volatile>) -> (!fir.ref<i32, volatile>, !fir.ref<i32, volatile>)
! CHECK:           hlfir.assign %[[VAL_0]] to %[[VAL_3]]#0 : i32, !fir.ref<i32, volatile>
! CHECK:           %[[VAL_4:.*]] = fir.volatile_cast %[[VAL_3]]#0 : (!fir.ref<i32, volatile>) -> !fir.ref<i32>
! CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
! CHECK:           return %[[VAL_5]] : i32
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFPb() -> i32 attributes {{.+}} {
! CHECK:           %[[VAL_0:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "r", uniq_name = "_QFFbEr"}
! CHECK:           %[[VAL_2:.*]] = fir.volatile_cast %[[VAL_1]] : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {{.+}} : (!fir.ref<i32, volatile>) -> (!fir.ref<i32, volatile>, !fir.ref<i32, volatile>)
! CHECK:           hlfir.assign %[[VAL_0]] to %[[VAL_3]]#0 : i32, !fir.ref<i32, volatile>
! CHECK:           %[[VAL_4:.*]] = fir.volatile_cast %[[VAL_3]]#0 : (!fir.ref<i32, volatile>) -> !fir.ref<i32>
! CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
! CHECK:           return %[[VAL_5]] : i32
! CHECK:         }

! CHECK-LABEL:   func.func private @_QFPc() -> f32 attributes {{.+}} {
! CHECK:           %[[VAL_0:.*]] = arith.constant 3.000000e+00 : f32
! CHECK:           %[[VAL_1:.*]] = fir.alloca f32 {bindc_name = "r", uniq_name = "_QFFcEr"}
! CHECK:           %[[VAL_2:.*]] = fir.volatile_cast %[[VAL_1]] : (!fir.ref<f32>) -> !fir.ref<f32, volatile>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {{.+}} : (!fir.ref<f32, volatile>) -> (!fir.ref<f32, volatile>, !fir.ref<f32, volatile>)
! CHECK:           hlfir.assign %[[VAL_0]] to %[[VAL_3]]#0 : f32, !fir.ref<f32, volatile>
! CHECK:           %[[VAL_4:.*]] = fir.volatile_cast %[[VAL_3]]#0 : (!fir.ref<f32, volatile>) -> !fir.ref<f32>
! CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_4]] : !fir.ref<f32>
! CHECK:           return %[[VAL_5]] : f32
! CHECK:         }
