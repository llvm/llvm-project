! Cross-TU propagation of `groupprivate` and its device_type via .mod files.

! RUN: rm -rf %t && split-file %s %t
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 -module-dir %t %t/m.f90 -o - > /dev/null
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 -J %t %t/use.f90 -o - | FileCheck %s

! First RUN builds gp_mod.mod.
! Second RUN lowers a consumer that only USE-associates the module
! and checks that the omp.groupprivate op picks up device_type recovered
! from the .mod file rather than the original source.

!--- m.f90
module gp_mod
  implicit none
  integer, save :: gp_x
  !$omp groupprivate(gp_x) device_type(host)
end module gp_mod

!--- use.f90
program use_gp_mod
  use gp_mod
  implicit none
  !$omp teams
    gp_x = 42
  !$omp end teams
end program use_gp_mod

! CHECK-LABEL: func.func @_QQmain
! CHECK:         fir.address_of(@_QMgp_modEgp_x) : !fir.ref<i32>
! CHECK:         omp.teams {
! CHECK:           %[[GP:.*]] = omp.groupprivate @_QMgp_modEgp_x device_type (host) : !fir.ref<i32>
! CHECK:           %[[DECL:.*]]:2 = hlfir.declare %[[GP]] {uniq_name = "_QMgp_modEgp_x"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[C42:.*]] = arith.constant 42 : i32
! CHECK:           hlfir.assign %[[C42]] to %[[DECL]]#0 : i32, !fir.ref<i32>
! CHECK:           omp.terminator
! CHECK:         }
