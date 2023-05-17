! Test -flang-experimental-hlfir flag
! RUN: %flang_fc1 -flang-experimental-hlfir -emit-fir -o - %s | FileCheck %s
! RUN: %flang_fc1 -emit-fir -o - %s | FileCheck %s --check-prefix NO-HLFIR

subroutine test(a, res)
  real :: a(:), res
  res = SUM(a)
end subroutine
! CHECK-LABEL: func.func @_QPtest
! CHECK:           %[[A:.*]]: !fir.box<!fir.array<?xf32>>
! CHECK:           %[[RES:.*]]: !fir.ref<f32>
! CHECK-DAG:     %[[A_VAR:.*]]:2 = hlfir.declare %[[A]]
! CHECK-DAG:     %[[RES_VAR:.*]]:2 = hlfir.declare %[[RES]]
! CHECK-NEXT:    %[[SUM_RES:.*]] = hlfir.sum %[[A_VAR]]#0
! CHECK-NEXT:    hlfir.assign %[[SUM_RES]] to %[[RES_VAR]]#0
! CHECK-NEXT:    hlfir.destroy %[[SUM_RES]]
! CHECK-NEXT:    return
! CHECK-NEXT:  }

! NO-HLFIR-NOT: hlfir.
