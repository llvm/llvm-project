! RUN: rm -rf %t && split-file %s %t
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -module-dir %t %t/declare_target_module.f90 -o - > /dev/null
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -J %t %t/use_declare_target_module.f90 -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -fopenmp-is-target-device -J %t %t/use_declare_target_module.f90 -o - | FileCheck %s

!--- declare_target_module.f90
module declare_target_module
  implicit none
  integer, dimension(10) :: global_arr
  !$omp declare target (global_arr)
  real :: global_real
  !$omp declare target link(global_real)
  integer :: global_integer
  !$omp declare target to(global_integer)
  integer :: global_device_integer
  !$omp declare target enter(global_device_integer) device_type(nohost)
  contains
  subroutine module_s()
    !$omp declare target
  end subroutine
end module

!--- use_declare_target_module.f90
module use_declare_target_module
use declare_target_module
implicit none
contains

subroutine s()
  !$omp declare target
  global_arr(1) = 1
  global_real = 1.0
  global_integer = 1
end subroutine
!CHECK-DAG: fir.global @_QMdeclare_target_moduleEglobal_arr {alignment = 64 : i64, omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>} : !fir.array<10xi32>
!CHECK-DAG: fir.global @_QMdeclare_target_moduleEglobal_real {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} : f32
!CHECK-DAG: fir.global @_QMdeclare_target_moduleEglobal_integer {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>} : i32

subroutine device_s()
  !$omp declare target enter(device_s) device_type(nohost)
  global_device_integer = 1
end subroutine
!CHECK-DAG: fir.global @_QMdeclare_target_moduleEglobal_device_integer {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (enter)>} : i32

subroutine call_module_s()
call module_s()
end subroutine
!CHECK-DAG: func.func private @_QMdeclare_target_modulePmodule_s() attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>}
end module
