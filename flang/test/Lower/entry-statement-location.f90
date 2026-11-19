! Check the location on the generated func.func: the SUBROUTINE or FUNCTION
! line for the host subprogram, the ENTRY line for each alternate entry.

! RUN: %flang_fc1 -emit-fir -debug-info-kind=standalone %s -o - \
! RUN:   -mmlir -mlir-print-debuginfo -mmlir -mlir-print-local-scope \
! RUN:   | FileCheck %s

subroutine host_sub(i)
    integer :: i
    i = 1
    return
entry first_entry(i)
    i = 2
    return
entry second_entry(i)
    i = 3
end subroutine

! CHECK-LABEL: func.func @_QPhost_sub(
! CHECK: } loc("{{.*}}entry-statement-location.f90":8:1)

! CHECK-LABEL: func.func @_QPfirst_entry(
! CHECK: } loc("{{.*}}entry-statement-location.f90":12:1)

! CHECK-LABEL: func.func @_QPsecond_entry(
! CHECK: } loc("{{.*}}entry-statement-location.f90":15:1)

function host_func() result(r)
    integer :: r
    r = 1
    return
entry func_entry() result(r)
    r = 2
end function

! CHECK-LABEL: func.func @_QPhost_func()
! CHECK: } loc("{{.*}}entry-statement-location.f90":28:1)

! CHECK-LABEL: func.func @_QPfunc_entry()
! CHECK: } loc("{{.*}}entry-statement-location.f90":32:1)

module test_mod
contains
    subroutine module_sub()
        entry module_entry()
    end subroutine
end module

! CHECK-LABEL: func.func @_QMtest_modPmodule_sub()
! CHECK: } loc("{{.*}}entry-statement-location.f90":44:5)

! CHECK-LABEL: func.func @_QMtest_modPmodule_entry()
! CHECK: } loc("{{.*}}entry-statement-location.f90":45:9)
