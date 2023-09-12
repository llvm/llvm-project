! This test checks lowering of OpenACC set directive.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s --check-prefixes=CHECK,FIR
! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s --check-prefixes=CHECK,HLFIR

program test_acc_set
  logical :: l

!$acc set default_async(1)

!$acc set default_async(1) if(l)

!$acc set device_num(0)

!$acc set device_type(*)

!$acc set device_type(0)

end

! CHECK-LABEL: func.func @_QQmain()
! CHECK: %[[L:.*]] = fir.alloca !fir.logical<4> {bindc_name = "l", uniq_name = "_QFEl"}
! HLFIR: %[[DECLL:.*]]:2 = hlfir.declare %[[L]] {uniq_name = "_QFEl"}

! CHECK: %[[C1:.*]] = arith.constant 1 : i32
! CHECK: acc.set default_async(%[[C1]] : i32)

! CHECK: %[[C1:.*]] = arith.constant 1 : i32
! FIR: %[[LOAD_L:.*]] = fir.load %[[L]] : !fir.ref<!fir.logical<4>>
! HLFIR: %[[LOAD_L:.*]] = fir.load %[[DECLL]]#0 : !fir.ref<!fir.logical<4>>
! CHECK: %[[CONV_L:.*]] = fir.convert %[[LOAD_L]] : (!fir.logical<4>) -> i1
! CHECK: acc.set default_async(%[[C1]] : i32) if(%[[CONV_L]])

! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: acc.set device_num(%[[C0]] : i32)

! CHECK: %[[C_1:.*]] = arith.constant -1 : index
! CHECK: acc.set device_type(%[[C_1]] : index)

! CHECK: %[[C0:.*]] = arith.constant 0 : i32
! CHECK: acc.set device_type(%[[C0]] : i32)


