! RUN: %flang_fc1 -ffast-real-mod -emit-mlir -o - %s | FileCheck %s --check-prefixes=CHECK-FRM
! RUN: %flang_fc1 -fno-fast-real-mod -ffast-real-mod -emit-mlir -o - %s | FileCheck %s --check-prefixes=CHECK-FRM
! RUN: %flang_fc1 -ffast-real-mod -mllvm -fast-real-mod-trunc=math-trunc -emit-mlir -o - %s | FileCheck %s --check-prefixes=CHECK-FRM
! RUN: %flang_fc1 -ffast-math -emit-mlir -o - %s | FileCheck %s --check-prefixes=CHECK-FM
! RUN: %flang_fc1 -ffast-real-mod -mllvm -fast-real-mod-trunc=emulate-trunc -emit-mlir -o - %s | FileCheck %s --check-prefixes=CHECK-EMU
! RUN: %flang_fc1 -ffast-math -mllvm -fast-real-mod-trunc=emulate-trunc -emit-mlir -o - %s | FileCheck %s --check-prefixes=CHECK-EMU

! REAL(2) MOD is only supported inline, so it is not tested with
! -fno-fast-real-mod.

! CHECK-LABEL: @_QPmod_real2
subroutine mod_real2(r, a, p)
    implicit none
    real(kind=2) :: r, a, p
! CHECK-FRM: %[[A:.*]] = fir.declare{{.*}}a"
! CHECK-FRM: %[[P:.*]] = fir.declare{{.*}}p"
! CHECK-FRM: %[[R:.*]] = fir.declare{{.*}}r"
! CHECK-FRM: %[[A_LOAD:.*]] = fir.load %[[A]]
! CHECK-FRM: %[[P_LOAD:.*]] = fir.load %[[P]]
! CHECK-FRM: %[[DIV:.*]] = arith.divf %[[A_LOAD]], %[[P_LOAD]] fastmath<contract> : f16
! CHECK-FRM: %[[TRUNC:.*]] = math.trunc %[[DIV]] fastmath<contract> : f16
! CHECK-FRM: %[[MUL:.*]] = arith.mulf %[[TRUNC]], %[[P_LOAD]] fastmath<contract> : f16
! CHECK-FRM: %[[SUB:.*]] = arith.subf %[[A_LOAD]], %[[MUL]] fastmath<contract> : f16
! CHECK-FRM: fir.store %[[SUB]] to %[[R]] : !fir.ref<f16>
! CHECK-FM: %[[A:.*]] = fir.declare{{.*}}a"
! CHECK-FM: %[[P:.*]] = fir.declare{{.*}}p"
! CHECK-FM: %[[R:.*]] = fir.declare{{.*}}r"
! CHECK-FM: %[[A_LOAD:.*]] = fir.load %[[A]]
! CHECK-FM: %[[P_LOAD:.*]] = fir.load %[[P]]
! CHECK-FM: %[[DIV:.*]] = arith.divf %[[A_LOAD]], %[[P_LOAD]] fastmath<fast> : f16
! CHECK-FM: %[[TRUNC:.*]] = math.trunc %[[DIV]] fastmath<fast> : f16
! CHECK-FM: %[[MUL:.*]] = arith.mulf %[[TRUNC]], %[[P_LOAD]] fastmath<fast> : f16
! CHECK-FM: %[[SUB:.*]] = arith.subf %[[A_LOAD]], %[[MUL]] fastmath<fast> : f16
! CHECK-FM: fir.store %[[SUB]] to %[[R]] : !fir.ref<f16>
! CHECK-EMU: %[[BOUND:.*]] = arith.constant 1.024000e+03 : f16
! CHECK-EMU: %[[A:.*]] = fir.declare{{.*}}a"
! CHECK-EMU: %[[P:.*]] = fir.declare{{.*}}p"
! CHECK-EMU: %[[R:.*]] = fir.declare{{.*}}r"
! CHECK-EMU: %[[A_LOAD:.*]] = fir.load %[[A]]
! CHECK-EMU: %[[P_LOAD:.*]] = fir.load %[[P]]
! CHECK-EMU: %[[DIV:.*]] = arith.divf %[[A_LOAD]], %[[P_LOAD]] fastmath<{{contract|fast}}> : f16
! CHECK-EMU: %[[ABS:.*]] = math.absf %[[DIV]] fastmath<{{contract|fast}}> : f16
! CHECK-EMU: %[[CMP:.*]] = arith.cmpf olt, %[[ABS]], %[[BOUND]] fastmath<{{contract|fast}}> : f16
! CHECK-EMU: %[[INT:.*]] = arith.fptosi %[[DIV]] : f16 to i16
! CHECK-EMU: %[[FLT:.*]] = arith.sitofp %[[INT]] : i16 to f16
! CHECK-EMU: %[[TRUNC:.*]] = arith.select %[[CMP]], %[[FLT]], %[[DIV]] : f16
! CHECK-EMU: %[[MUL:.*]] = arith.mulf %[[TRUNC]], %[[P_LOAD]] fastmath<{{contract|fast}}> : f16
! CHECK-EMU: %[[SUB:.*]] = arith.subf %[[A_LOAD]], %[[MUL]] fastmath<{{contract|fast}}> : f16
! CHECK-EMU: fir.store %[[SUB]] to %[[R]] : !fir.ref<f16>
    r = mod(a, p)
end subroutine mod_real2
