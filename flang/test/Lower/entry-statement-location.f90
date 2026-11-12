! Check the location on the generated func.func and on its fir.dummy_scope: the
! SUBROUTINE or FUNCTION line for the host subprogram, the ENTRY line for each
! alternate entry.

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
! CHECK: fir.dummy_scope : !fir.dscope loc("{{.*}}entry-statement-location.f90":9:1)
! CHECK: } loc("{{.*}}entry-statement-location.f90":9:1)

! CHECK-LABEL: func.func @_QPfirst_entry(
! CHECK: fir.dummy_scope : !fir.dscope loc("{{.*}}entry-statement-location.f90":13:1)
! CHECK: } loc("{{.*}}entry-statement-location.f90":13:1)

! CHECK-LABEL: func.func @_QPsecond_entry(
! CHECK: fir.dummy_scope : !fir.dscope loc("{{.*}}entry-statement-location.f90":16:1)
! CHECK: } loc("{{.*}}entry-statement-location.f90":16:1)

function host_func() result(r)
    integer :: r
    r = 1
    return
entry func_entry() result(r)
    r = 2
end function

! CHECK-LABEL: func.func @_QPhost_func()
! CHECK: fir.dummy_scope : !fir.dscope loc("{{.*}}entry-statement-location.f90":32:1)
! CHECK: } loc("{{.*}}entry-statement-location.f90":32:1)

! CHECK-LABEL: func.func @_QPfunc_entry()
! CHECK: fir.dummy_scope : !fir.dscope loc("{{.*}}entry-statement-location.f90":36:1)
! CHECK: } loc("{{.*}}entry-statement-location.f90":36:1)

module test_mod
contains
    subroutine module_sub()
        entry module_entry()
    end subroutine
end module

! CHECK-LABEL: func.func @_QMtest_modPmodule_sub()
! CHECK: fir.dummy_scope : !fir.dscope loc("{{.*}}entry-statement-location.f90":50:5)
! CHECK: } loc("{{.*}}entry-statement-location.f90":50:5)

! CHECK-LABEL: func.func @_QMtest_modPmodule_entry()
! CHECK: fir.dummy_scope : !fir.dscope loc("{{.*}}entry-statement-location.f90":51:9)
! CHECK: } loc("{{.*}}entry-statement-location.f90":51:9)
