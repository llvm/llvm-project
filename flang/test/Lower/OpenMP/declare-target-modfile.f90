! Cross-TU propagation of `declare target` on a module variable via .mod files.

! RUN: rm -rf %t && split-file %s %t
! RUN: %flang_fc1 -emit-hlfir -fopenmp -module-dir %t %t/m.f90 -o - > /dev/null
! RUN: %flang_fc1 -emit-hlfir -fopenmp -J %t %t/use.f90 -o - | FileCheck %s

! The consumer only USE-associates the module. Its declaration must still carry
! omp.declare_target, recovered from the .mod file, or it is internalized for
! the device and the definition is lost.

!--- m.f90
module dt_mod
  implicit none
  integer :: dt_x
  !$omp declare target(dt_x)
end module dt_mod

!--- use.f90
subroutine use_dt_mod(out)
  use dt_mod
  implicit none
  integer, intent(out) :: out
  !$omp target map(tofrom: out)
    out = dt_x
  !$omp end target
end subroutine use_dt_mod

! CHECK: fir.global @_QMdt_modEdt_x {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to), automap = false>} : i32
