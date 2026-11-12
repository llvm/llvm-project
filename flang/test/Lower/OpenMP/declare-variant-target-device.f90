! RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-version=51 \
! RUN:   -fopenmp-is-target-device %s -o - | FileCheck %s

module m
contains
  subroutine base
    !$omp declare variant (base:vsub) match (device={kind(nohost)})
  end subroutine base

  subroutine vsub
  end subroutine vsub

  subroutine device_caller
    !$omp declare target to(device_caller) device_type(nohost)
    call base()
  end subroutine device_caller

  subroutine target_region_caller
    !$omp target
    call base()
    !$omp end target
  end subroutine target_region_caller
end module m

! CHECK-LABEL: func.func @_QMmPdevice_caller
! CHECK: fir.call @_QMmPvsub(){{.*}}: () -> ()
! CHECK-NOT: fir.call @_QMmPbase

! CHECK-LABEL: func.func @_QMmPtarget_region_caller
! CHECK: omp.target
! CHECK: fir.call @_QMmPvsub(){{.*}}: () -> ()
