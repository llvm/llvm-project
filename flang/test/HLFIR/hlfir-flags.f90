! Test -flang-deprecated-hlfir, -flang-experimental-hlfir (flang-new), and
! -hlfir (bbc), -emit-hlfir, -emit-fir flags
! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck --check-prefix HLFIR --check-prefix ALL %s
! RUN: bbc -emit-hlfir -o - %s | FileCheck --check-prefix HLFIR --check-prefix ALL %s
! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck --check-prefix HLFIR --check-prefix ALL %s
! RUN: bbc -emit-hlfir -hlfir -o - %s | FileCheck --check-prefix HLFIR --check-prefix ALL %s
! RUN: %flang_fc1 -emit-fir -o - %s | FileCheck --check-prefix FIR --check-prefix ALL %s
! RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir -o - %s | FileCheck %s --check-prefix NO-HLFIR --check-prefix ALL
! RUN: %flang_fc1 -emit-fir -flang-experimental-hlfir -o - %s | FileCheck %s --check-prefix FIR --check-prefix ALL
! RUN: bbc -emit-fir -o - %s | FileCheck --check-prefix FIR --check-prefix ALL %s
! RUN: bbc -emit-fir -hlfir=false -o - %s | FileCheck %s --check-prefix NO-HLFIR --check-prefix ALL

! | Action      | -flang-deprecated-no-hlfir  | Result                          |
! |             | / -hlfir=false?             |                                 |
! | =========== | =========================== | =============================== |
! | -emit-hlfir | N                           | Outputs HLFIR                   |
! | -emit-hlfir | Y                           | Outputs HLFIR                   |
! | -emit-fir   | N                           | Outputs FIR, lowering via HLFIR |
! | -emit-fir   | Y                           | Outputs FIR, using old lowering |

subroutine test(a, res)
  real :: a(:), res
  res = SUM(a)
end subroutine
! ALL-LABEL: func.func @_QPtest
! ALL:             %[[A:.*]]: !fir.box<!fir.array<?xf32>>
! ALL:             %[[RES:.*]]: !fir.ref<f32>

! HLFIR:         %[[A_VAR:.*]]:2 = hlfir.declare %[[A]]
! fir.declare is only generated via the hlfir -> fir lowering
! FIR:           %[[A_VAR:.*]] = fir.declare %[[A]]
! NO-HLFIR-NOT:  fir.declare

! HLFIR-DAG:     %[[RES_VAR:.*]]:2 = hlfir.declare %[[RES]]
! FIR:           %[[RES_VAR:.*]] = fir.declare %[[RES]]
! NO-HLFIR-NOT:  fir.declare

! HLFIR-NEXT:    %[[SUM_RES:.*]] = hlfir.sum %[[A_VAR]]#0
! HLFIR-NEXT:    hlfir.assign %[[SUM_RES]] to %[[RES_VAR]]#0
! FIR-NOT:       hlfir.
! NO-HLFIR-NOT:  hlfir.

! ALL:           return
! ALL-NEXT:  }

