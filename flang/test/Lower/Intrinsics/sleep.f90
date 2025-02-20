! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

subroutine test_sleep()
! CHECK-LABEL:   func.func @_QPtest_sleep() {

  call sleep(1_2)
! CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i16
! CHECK:           %[[VAL_1:.*]] = fir.convert %[[VAL_0]] : (i16) -> i64
! CHECK:           fir.call @_FortranASleep(%[[VAL_1]]) fastmath<contract> : (i64) -> ()

  call sleep(1_4)
! CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (i32) -> i64
! CHECK:           fir.call @_FortranASleep(%[[VAL_4]]) fastmath<contract> : (i64) -> ()

  call sleep(1_8)
! CHECK:           %[[VAL_6:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i64) -> i64
! CHECK:           fir.call @_FortranASleep(%[[VAL_7]]) fastmath<contract> : (i64) -> ()

  call sleep(1_16)
! CHECK:           %[[VAL_9:.*]] = arith.constant 1 : i128
! CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i128) -> i64
! CHECK:           fir.call @_FortranASleep(%[[VAL_10]]) fastmath<contract> : (i64) -> ()
end
! CHECK:           return
! CHECK:         }
