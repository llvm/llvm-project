! RUN: %flang_fc1 -ffast-real-mod -emit-mlir -o - %s | FileCheck %s -DFMF=contract --check-prefixes=CHECK,CHECK-MT
! RUN: %flang_fc1 -fno-fast-real-mod -ffast-real-mod -emit-mlir -o - %s | FileCheck %s -DFMF=contract --check-prefixes=CHECK,CHECK-MT
! RUN: %flang_fc1 -ffast-real-mod -mllvm -fast-real-mod-trunc=math-trunc -emit-mlir -o - %s | FileCheck %s -DFMF=contract --check-prefixes=CHECK,CHECK-MT
! RUN: %flang_fc1 -ffast-math -emit-mlir -o - %s | FileCheck %s -DFMF=fast --check-prefixes=CHECK,CHECK-MT
! RUN: %flang_fc1 -ffast-real-mod -mllvm -fast-real-mod-trunc=emulate-trunc -emit-mlir -o - %s | FileCheck %s -DFMF=contract --check-prefixes=CHECK,CHECK-EMU
! RUN: %flang_fc1 -ffast-math -mllvm -fast-real-mod-trunc=emulate-trunc -emit-mlir -o - %s | FileCheck %s -DFMF=fast --check-prefixes=CHECK,CHECK-EMU

! REAL(2) MOD is only supported inline, so it is not tested with
! -fno-fast-real-mod.

! CHECK-LABEL: @_QPmod_real2
subroutine mod_real2(r, a, p)
    implicit none
    real(kind=2) :: r, a, p
! CHECK-EMU: %[[BOUND:.*]] = arith.constant 1.024000e+03 : f16
! CHECK: %[[A:.*]] = fir.declare{{.*}}a"
! CHECK: %[[P:.*]] = fir.declare{{.*}}p"
! CHECK: %[[R:.*]] = fir.declare{{.*}}r"
! CHECK: %[[A_LOAD:.*]] = fir.load %[[A]]
! CHECK: %[[P_LOAD:.*]] = fir.load %[[P]]
! CHECK: %[[DIV:.*]] = arith.divf %[[A_LOAD]], %[[P_LOAD]] fastmath<[[FMF]]> : f16
! CHECK-MT: %[[TRUNC:.*]] = math.trunc %[[DIV]] fastmath<[[FMF]]> : f16
! CHECK-EMU: %[[ABS:.*]] = math.absf %[[DIV]] fastmath<[[FMF]]> : f16
! CHECK-EMU: %[[CMP:.*]] = arith.cmpf olt, %[[ABS]], %[[BOUND]] fastmath<[[FMF]]> : f16
! CHECK-EMU: %[[INT:.*]] = arith.fptosi %[[DIV]] : f16 to i16
! CHECK-EMU: %[[FLT:.*]] = arith.sitofp %[[INT]] : i16 to f16
! CHECK-EMU: %[[TRUNC:.*]] = arith.select %[[CMP]], %[[FLT]], %[[DIV]] : f16
! CHECK: %[[MUL:.*]] = arith.mulf %[[TRUNC]], %[[P_LOAD]] fastmath<[[FMF]]> : f16
! CHECK: %[[RES:.*]] = arith.subf %[[A_LOAD]], %[[MUL]] fastmath<[[FMF]]> : f16
! CHECK: fir.store %[[RES]] to %[[R]] : !fir.ref<f16>
    r = mod(a, p)
end subroutine mod_real2
