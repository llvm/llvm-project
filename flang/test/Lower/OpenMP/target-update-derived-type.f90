! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s --check-prefix=HOST
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda %s -o - | FileCheck %s --check-prefix=NONAMD
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa,nvptx64-nvidia-cuda %s -o - | FileCheck %s --check-prefix=MIXED
! RUN: %flang_fc1 -triple amdgcn-amd-amdhsa -emit-hlfir -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s --check-prefix=DEVICE

!REQUIRES: amdgpu-registered-target

module target_update_derived_type
  type :: wavefun
    real(8) :: ferwe
    real(8) :: aux
    complex(8) :: celen
    integer :: pad1
    integer :: nb
    integer :: pad2
    integer :: isp
    integer :: pad3
    logical :: ldo
    integer, pointer :: ptr
  end type
contains

! CHECK-LABEL: func.func @_QMtarget_update_derived_typePupdate_with_if(
! DEVICE-LABEL: func.func @_QMtarget_update_derived_typePupdate_with_if(
! DEVICE: omp.target kernel_type(generic)
! DEVICE-NOT: omp.target_update
! HOST-LABEL: func.func @_QMtarget_update_derived_typePupdate_with_if(
! HOST-NOT: omp.target kernel_type(generic)
! HOST: omp.target_update
! HOST-NOT: omp.target kernel_type(generic)
! HOST-LABEL: func.func @_QMtarget_update_derived_typePupdate_without_if(
! NONAMD-LABEL: func.func @_QMtarget_update_derived_typePupdate_with_if(
! NONAMD-NOT: omp.target kernel_type(generic)
! NONAMD: omp.target_update
! NONAMD-NOT: omp.target kernel_type(generic)
! NONAMD-LABEL: func.func @_QMtarget_update_derived_typePupdate_without_if(
! MIXED-LABEL: func.func @_QMtarget_update_derived_typePupdate_with_if(
! MIXED-NOT: omp.target kernel_type(generic)
! MIXED: omp.target_update
! MIXED-NOT: omp.target kernel_type(generic)
! MIXED-LABEL: func.func @_QMtarget_update_derived_typePupdate_without_if(
subroutine update_with_if(w, enabled)
  type(wavefun) :: w
  logical :: enabled

  ! CHECK: %[[SOURCE:.*]] = fir.alloca tuple<f64, complex<f64>, i32, i32, !fir.logical<4>>
  ! CHECK: %[[COND:.*]] = fir.convert %{{.*}} : (!fir.logical<4>) -> i1
  ! CHECK: %[[FERWE:.*]] = hlfir.designate %{{.*}}{"ferwe"}
  ! CHECK: %[[CELEN:.*]] = hlfir.designate %{{.*}}{"celen"}
  ! CHECK: %[[NB:.*]] = hlfir.designate %{{.*}}{"nb"}
  ! CHECK: %[[ISP:.*]] = hlfir.designate %{{.*}}{"isp"}
  ! CHECK: %[[LDO:.*]] = hlfir.designate %{{.*}}{"ldo"}
  ! CHECK: fir.if %[[COND]] {
  ! CHECK: %[[FERWE_HOST:.*]] = fir.load %[[FERWE]] : !fir.ref<f64>
  ! CHECK: %[[PACK0:.*]] = fir.coordinate_of %[[SOURCE]], {{.*}} -> !fir.ref<f64>
  ! CHECK: fir.store %[[FERWE_HOST]] to %[[PACK0]] : !fir.ref<f64>
  ! CHECK: %[[FERWE_MAP:.*]] = omp.map.info var_ptr(%[[FERWE]] : !fir.ref<f64>, f64) map_clauses(storage) capture(ByRef)
  ! CHECK: %[[CELEN_HOST:.*]] = fir.load %[[CELEN]] : !fir.ref<complex<f64>>
  ! CHECK: %[[PACK1:.*]] = fir.coordinate_of %[[SOURCE]], {{.*}} -> !fir.ref<complex<f64>>
  ! CHECK: fir.store %[[CELEN_HOST]] to %[[PACK1]] : !fir.ref<complex<f64>>
  ! CHECK: %[[CELEN_MAP:.*]] = omp.map.info var_ptr(%[[CELEN]] : !fir.ref<complex<f64>>, complex<f64>) map_clauses(storage) capture(ByRef)
  ! CHECK: %[[NB_HOST:.*]] = fir.load %[[NB]] : !fir.ref<i32>
  ! CHECK: %[[PACK2:.*]] = fir.coordinate_of %[[SOURCE]], {{.*}} -> !fir.ref<i32>
  ! CHECK: fir.store %[[NB_HOST]] to %[[PACK2]] : !fir.ref<i32>
  ! CHECK: %[[NB_MAP:.*]] = omp.map.info var_ptr(%[[NB]] : !fir.ref<i32>, i32) map_clauses(storage) capture(ByRef)
  ! CHECK: %[[ISP_HOST:.*]] = fir.load %[[ISP]] : !fir.ref<i32>
  ! CHECK: %[[PACK3:.*]] = fir.coordinate_of %[[SOURCE]], {{.*}} -> !fir.ref<i32>
  ! CHECK: fir.store %[[ISP_HOST]] to %[[PACK3]] : !fir.ref<i32>
  ! CHECK: %[[ISP_MAP:.*]] = omp.map.info var_ptr(%[[ISP]] : !fir.ref<i32>, i32) map_clauses(storage) capture(ByRef)
  ! CHECK: %[[LDO_HOST:.*]] = fir.load %[[LDO]] : !fir.ref<!fir.logical<4>>
  ! CHECK: %[[PACK4:.*]] = fir.coordinate_of %[[SOURCE]], {{.*}} -> !fir.ref<!fir.logical<4>>
  ! CHECK: fir.store %[[LDO_HOST]] to %[[PACK4]] : !fir.ref<!fir.logical<4>>
  ! CHECK: %[[LDO_MAP:.*]] = omp.map.info var_ptr(%[[LDO]] : !fir.ref<!fir.logical<4>>, !fir.logical<4>) map_clauses(storage) capture(ByRef)
  ! CHECK: %[[SOURCE_MAP:.*]] = omp.map.info var_ptr(%[[SOURCE]] {{.*}}) map_clauses(to) capture(ByRef) name(".omp.target.update.source")
  ! CHECK: omp.target kernel_type(generic) map_entries(%[[SOURCE_MAP]] -> [[SOURCE_ARG:%[^, ]+]], %[[FERWE_MAP]] -> [[FERWE_ARG:%[^, ]+]], %[[CELEN_MAP]] -> [[CELEN_ARG:%[^, ]+]], %[[NB_MAP]] -> [[NB_ARG:%[^, ]+]], %[[ISP_MAP]] -> [[ISP_ARG:%[^, ]+]], %[[LDO_MAP]] -> [[LDO_ARG:%[^, ]+]]
  ! CHECK: %[[FERWE_SOURCE:.*]] = fir.coordinate_of [[SOURCE_ARG]], {{.*}} -> !fir.ref<f64>
  ! CHECK: %[[FERWE_VALUE:.*]] = fir.load %[[FERWE_SOURCE]] : !fir.ref<f64>
  ! CHECK: fir.store %[[FERWE_VALUE]] to [[FERWE_ARG]] : !fir.ref<f64>
  ! CHECK: %[[CELEN_SOURCE:.*]] = fir.coordinate_of [[SOURCE_ARG]], {{.*}} -> !fir.ref<complex<f64>>
  ! CHECK: %[[CELEN_VALUE:.*]] = fir.load %[[CELEN_SOURCE]] : !fir.ref<complex<f64>>
  ! CHECK: fir.store %[[CELEN_VALUE]] to [[CELEN_ARG]] : !fir.ref<complex<f64>>
  ! CHECK: %[[NB_SOURCE:.*]] = fir.coordinate_of [[SOURCE_ARG]], {{.*}} -> !fir.ref<i32>
  ! CHECK: %[[NB_VALUE:.*]] = fir.load %[[NB_SOURCE]] : !fir.ref<i32>
  ! CHECK: fir.store %[[NB_VALUE]] to [[NB_ARG]] : !fir.ref<i32>
  ! CHECK: %[[ISP_SOURCE:.*]] = fir.coordinate_of [[SOURCE_ARG]], {{.*}} -> !fir.ref<i32>
  ! CHECK: %[[ISP_VALUE:.*]] = fir.load %[[ISP_SOURCE]] : !fir.ref<i32>
  ! CHECK: fir.store %[[ISP_VALUE]] to [[ISP_ARG]] : !fir.ref<i32>
  ! CHECK: %[[LDO_SOURCE:.*]] = fir.coordinate_of [[SOURCE_ARG]], {{.*}} -> !fir.ref<!fir.logical<4>>
  ! CHECK: %[[LDO_VALUE:.*]] = fir.load %[[LDO_SOURCE]] : !fir.ref<!fir.logical<4>>
  ! CHECK: fir.store %[[LDO_VALUE]] to [[LDO_ARG]] : !fir.ref<!fir.logical<4>>
  ! CHECK-NEXT: omp.terminator
  ! CHECK-NOT: omp.target_update
  ! CHECK: return
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

  ! CHECK: %[[FERWE_MAP:.*]] = omp.map.info {{.*}} map_clauses(to)
  ! CHECK: %[[PTR_MAP:.*]] = omp.map.info {{.*}} map_clauses(to) {{.*}}name("w%ptr")
  ! CHECK: omp.target_update map_entries(%[[FERWE_MAP]], %[[PTR_MAP]],
  !$omp target update to(w%ferwe, w%ptr)
end subroutine

! CHECK-LABEL: func.func @_QMtarget_update_derived_typePupdate_device(
subroutine update_device(w)
  type(wavefun) :: w

  ! CHECK: %[[FERWE_MAP:.*]] = omp.map.info {{.*}} map_clauses(to)
  ! CHECK: %[[NB_MAP:.*]] = omp.map.info {{.*}} map_clauses(to)
  ! CHECK: omp.target_update device({{.*}}) map_entries(%[[FERWE_MAP]], %[[NB_MAP]]
  !$omp target update to(w%ferwe, w%nb) device(0)
end subroutine

! CHECK-LABEL: func.func @_QMtarget_update_derived_typePupdate_single(
subroutine update_single(w)
  type(wavefun) :: w

  ! CHECK: %[[MAP:.*]] = omp.map.info {{.*}} map_clauses(to)
  ! CHECK-NOT: omp.target kernel_type(generic)
  ! CHECK: omp.target_update map_entries(%[[MAP]]
  !$omp target update to(w%ferwe)
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

  ! CHECK: %[[FERWE_MAP:.*]] = omp.map.info {{.*}} map_clauses(from)
  ! CHECK: %[[NB_MAP:.*]] = omp.map.info {{.*}} map_clauses(from)
  ! CHECK: omp.target_update map_entries(%[[FERWE_MAP]], %[[NB_MAP]]
  !$omp target update from(w%ferwe, w%nb)
end subroutine

! CHECK-LABEL: func.func @_QMtarget_update_derived_typePupdate_nowait(
subroutine update_nowait(w)
  type(wavefun) :: w

  ! CHECK: %[[FERWE_MAP:.*]] = omp.map.info {{.*}} map_clauses(to)
  ! CHECK: %[[NB_MAP:.*]] = omp.map.info {{.*}} map_clauses(to)
  ! CHECK: omp.target_update map_entries(%[[FERWE_MAP]], %[[NB_MAP]]{{.*}}) nowait
  !$omp target update to(w%ferwe, w%nb) nowait
end subroutine

end module
