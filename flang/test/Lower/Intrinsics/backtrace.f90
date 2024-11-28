! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPbacktrace_test() {
! CHECK:         %[[VAL_0:.*]] = fir.call @_FortranABacktrace() {{.*}}: () -> none
! CHECK:         return
! CHECK:       }

subroutine backtrace_test()
  call backtrace
end subroutine
