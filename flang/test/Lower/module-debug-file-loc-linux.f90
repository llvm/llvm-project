! Test that the module has the location information

! RUN: %flang_fc1 -mmlir --mlir-print-debuginfo -emit-fir -o - %s | FileCheck %s

! REQUIRES: system-linux

subroutine sb1()
end subroutine

! CHECK: module attributes
! CHECK:   func.func @_QPsb1() {
! CHECK:   }
! CHECK: } loc(#[[MODULE_LOC:.*]])
! CHECK: #[[MODULE_LOC]] = loc("/{{.*}}/test/Lower/module-debug-file-loc-linux.f90":0:0)
