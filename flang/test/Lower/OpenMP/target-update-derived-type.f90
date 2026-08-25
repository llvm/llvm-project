! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s --check-prefix=HOST
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda %s -o - | FileCheck %s --check-prefix=NONAMD
! RUN: %flang_fc1 -triple amdgcn-amd-amdhsa -emit-hlfir -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s --check-prefix=DEVICE

module target_update_derived_type
  type :: wavefun
    real(8) :: ferwe
    real(8) :: aux
    complex(8) :: celen
    integer :: nb
    integer :: isp
    logical :: ldo
    integer, pointer :: ptr
  end type
contains

! CHECK-LABEL: func.func @_QMtarget_update_derived_typePupdate_with_if(
! DEVICE-LABEL: func.func @_QMtarget_update_derived_typePupdate_with_if(
! DEVICE: omp.target kernel_type(generic)
! DEVICE-NOT: omp.target_update
subroutine update_with_if(w, enabled)
  type(wavefun) :: w
  logical :: enabled

  ! CHECK: %[[SOURCE:.*]] = fir.alloca tuple<f64, complex<f64>, i32, i32, !fir.logical<4>>
  ! CHECK: %[[COND:.*]] = fir.convert %{{.*}} : (!fir.logical<4>) -> i1
  ! CHECK: %[[FERWE:.*]] = hlfir.designate %{{.*}}{"ferwe"}
  ! CHECK: %[[CELEN:.*]] = hlfir.designate %{{.*}}{"celen"}
  ! CHECK: fir.if %[[COND]] {
  ! CHECK: fir.store {{.*}} to {{.*}} : !fir.ref<f64>
  ! CHECK: %[[FERWE_MAP:.*]] = omp.map.info var_ptr(%[[FERWE]] : !fir.ref<f64>, f64) map_clauses(storage) capture(ByRef)
  ! CHECK: fir.store {{.*}} to {{.*}} : !fir.ref<complex<f64>>
  ! CHECK: %[[CELEN_MAP:.*]] = omp.map.info var_ptr(%[[CELEN]] : !fir.ref<complex<f64>>, complex<f64>) map_clauses(storage) capture(ByRef)
  ! CHECK: %[[SOURCE_MAP:.*]] = omp.map.info var_ptr(%[[SOURCE]] {{.*}}) map_clauses(to) capture(ByRef) name(".omp.target.update.source")
  ! CHECK: omp.target kernel_type(generic) map_entries(%[[SOURCE_MAP]] -> [[SOURCE_ARG:%[^, ]+]], %[[FERWE_MAP]] -> [[FERWE_ARG:%[^, ]+]], %[[CELEN_MAP]] -> [[CELEN_ARG:%[^, ]+]]
  ! CHECK: %[[FERWE_SOURCE:.*]] = fir.coordinate_of [[SOURCE_ARG]], {{.*}} -> !fir.ref<f64>
  ! CHECK: %[[FERWE_VALUE:.*]] = fir.load %[[FERWE_SOURCE]] : !fir.ref<f64>
  ! CHECK: fir.store %[[FERWE_VALUE]] to [[FERWE_ARG]] : !fir.ref<f64>
  ! CHECK: %[[CELEN_SOURCE:.*]] = fir.coordinate_of [[SOURCE_ARG]], {{.*}} -> !fir.ref<complex<f64>>
  ! CHECK: %[[CELEN_VALUE:.*]] = fir.load %[[CELEN_SOURCE]] : !fir.ref<complex<f64>>
  ! CHECK: fir.store %[[CELEN_VALUE]] to [[CELEN_ARG]] : !fir.ref<complex<f64>>
  ! CHECK-NOT: omp.target_update
  ! CHECK: return
  ! HOST: omp.target_update
  ! NONAMD: omp.target_update
  !$omp target update to(w%ferwe, w%celen, w%nb, w%isp, w%ldo) if(enabled)
end subroutine

! CHECK-LABEL: func.func @_QMtarget_update_derived_typePupdate_without_if(
! DEVICE-LABEL: func.func @_QMtarget_update_derived_typePupdate_without_if(
subroutine update_without_if(w)
  type(wavefun) :: w

  ! CHECK: %[[SOURCE:.*]] = fir.alloca tuple<complex<f64>, i32, i32, !fir.logical<4>>
  ! CHECK: %[[CELEN:.*]] = hlfir.designate %{{.*}}{"celen"}
  ! CHECK: %[[CELEN_MAP:.*]] = omp.map.info var_ptr(%[[CELEN]] : !fir.ref<complex<f64>>, complex<f64>) map_clauses(storage) capture(ByRef)
  ! CHECK: %[[SOURCE_MAP:.*]] = omp.map.info var_ptr(%[[SOURCE]] {{.*}}) map_clauses(to) capture(ByRef) name(".omp.target.update.source")
  ! CHECK: omp.target kernel_type(generic) map_entries(%[[SOURCE_MAP]] -> [[SOURCE_ARG:%[^, ]+]], %[[CELEN_MAP]] -> [[CELEN_ARG:%[^, ]+]]
  ! CHECK: %[[CELEN_SOURCE:.*]] = fir.coordinate_of [[SOURCE_ARG]], {{.*}} -> !fir.ref<complex<f64>>
  ! CHECK: %[[CELEN_VALUE:.*]] = fir.load %[[CELEN_SOURCE]] : !fir.ref<complex<f64>>
  ! CHECK: fir.store %[[CELEN_VALUE]] to [[CELEN_ARG]] : !fir.ref<complex<f64>>
  ! CHECK-NOT: omp.target_update
  ! CHECK: return
  !$omp target update to(w%celen, w%nb, w%isp, w%ldo)
end subroutine

! CHECK-LABEL: func.func @_QMtarget_update_derived_typePupdate_pointer(
subroutine update_pointer(w)
  type(wavefun) :: w

  ! CHECK: omp.map.info {{.*}} map_clauses(to)
  ! CHECK: omp.target_update map_entries(
  !$omp target update to(w%ptr)
end subroutine

! CHECK-LABEL: func.func @_QMtarget_update_derived_typePupdate_device(
subroutine update_device(w)
  type(wavefun) :: w

  ! CHECK: %[[MAP:.*]] = omp.map.info {{.*}} map_clauses(to)
  ! CHECK: omp.target_update device({{.*}}) map_entries(%[[MAP]]
  !$omp target update to(w%ferwe) device(0)
end subroutine

! CHECK-LABEL: func.func @_QMtarget_update_derived_typePupdate_array_element(
subroutine update_array_element(w)
  type(wavefun) :: w(2)

  ! CHECK: fir.alloca tuple<f64, i32>
  ! CHECK: omp.target kernel_type(generic)
  ! CHECK-NOT: omp.target_update
  !$omp target update to(w(2)%ferwe, w(2)%nb)
end subroutine

! CHECK-LABEL: func.func @_QMtarget_update_derived_typePupdate_from(
subroutine update_from(w)
  type(wavefun) :: w

  ! CHECK: %[[MAP:.*]] = omp.map.info {{.*}} map_clauses(from)
  ! CHECK: omp.target_update map_entries(%[[MAP]]
  !$omp target update from(w%ferwe)
end subroutine

! CHECK-LABEL: func.func @_QMtarget_update_derived_typePupdate_nowait(
subroutine update_nowait(w)
  type(wavefun) :: w

  ! CHECK: %[[MAP:.*]] = omp.map.info {{.*}} map_clauses(to)
  ! CHECK: omp.target_update map_entries(%[[MAP]]{{.*}}) nowait
  !$omp target update to(w%ferwe) nowait
end subroutine

end module
