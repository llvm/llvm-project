! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPall_args() {
! CHECK:         %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "gid", uniq_name = "_QFall_argsEgid"}
! CHECK:         %[[VAL_1:.*]]:2 = hlfir.declare %0 {uniq_name = "_QFall_argsEgid"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:         %[[VAL_2:.*]] = fir.call @_FortranAGetGID() fastmath<contract> : () -> i32
! CHECK:         hlfir.assign %[[VAL_2:.*]] to %[[VAL_1:.*]]#0 : i32, !fir.ref<i32>
! CHECK:         return
! CHECK:       }

subroutine all_args()
  integer :: gid
  gid = getgid()
end
