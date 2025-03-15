! Simple test for lowering of OpenMP Threadprivate Directive with HLFIR.

!RUN: %flang_fc1 -flang-experimental-hlfir -emit-hlfir -fopenmp %s -o - | FileCheck %s
!RUN: bbc -hlfir -emit-hlfir -fopenmp %s -o - | FileCheck %s

!CHECK-LABEL: @_QPsub
!CHECK:    %[[ADDR:.*]] = fir.address_of(@_QFsubEa) : !fir.ref<i32>
!CHECK:    %[[DECL:.*]]:2 = hlfir.declare %[[ADDR]] {uniq_name = "_QFsubEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[TP:.*]] = omp.threadprivate %[[DECL]]#1 : !fir.ref<i32> -> !fir.ref<i32>
!CHECK:    %[[TP_DECL:.*]]:2 = hlfir.declare %[[TP:.*]] {uniq_name = "_QFsubEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    omp.parallel   {
!CHECK:      %[[TP_PARALLEL:.*]] = omp.threadprivate %[[DECL]]#1 : !fir.ref<i32> -> !fir.ref<i32>
!CHECK:      %[[TP_PARALLEL_DECL:.*]]:2 = hlfir.declare %[[TP_PARALLEL]] {uniq_name = "_QFsubEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:      %[[TP_VAL:.*]] = fir.load %[[TP_PARALLEL_DECL]]#0 : !fir.ref<i32>
!CHECK:      %{{.*}} = fir.call @_FortranAioOutputInteger32(%{{.*}}, %[[TP_VAL]]) fastmath<contract> : (!fir.ref<i8>, i32) -> i1
!CHECK:      omp.terminator

subroutine sub()
  integer, save:: a
  !$omp threadprivate(a)
  !$omp parallel
    print *, a
  !$omp end parallel
end subroutine

!CHECK-LABEL: func.func @_QPsub_02()
subroutine sub_02()
  integer, save :: a
  !$omp threadprivate(a)
  !CHECK:   %[[ADDR_02:.*]] = fir.address_of(@_QFsub_02Ea) : !fir.ref<i32>
  !CHECK:   %[[DECL_02:.*]]:2 = hlfir.declare %[[ADDR_02]] {{{.*}} uniq_name = "_QFsub_02Ea"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  !CHECK:   %[[TP_02:.*]] = omp.threadprivate %[[DECL_02]]#1 : !fir.ref<i32> -> !fir.ref<i32>
  !CHECK:   %[[TP_DECL_02:.*]]:2 = hlfir.declare %[[TP_02]] {{{.*}} uniq_name = "_QFsub_02Ea"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  call sub_03
  !CHECK:   fir.call @_QFsub_02Psub_03() fastmath<contract> : () -> ()
  !CHECK:   return

contains

  !CHECK-LABEL: func.func private @_QFsub_02Psub_03()
  subroutine sub_03()
    !CHECK:   %[[ADDR_03:.*]] = fir.address_of(@_QFsub_02Ea) : !fir.ref<i32>
    !CHECK:   %[[DECL_03:.*]]:2 = hlfir.declare %[[ADDR_03]] {uniq_name = "_QFsub_02Ea"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    !CHECK:   %[[TP_03:.*]] = omp.threadprivate %[[DECL_03]]#1 : !fir.ref<i32> -> !fir.ref<i32>
    !CHECK:   %[[TP_DECL_03:.*]]:2 = hlfir.declare %[[TP_03]] {uniq_name = "_QFsub_02Ea"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    !$omp parallel default(private)
      !CHECK:   omp.parallel
      !CHECK:     %[[TP_04:.*]] = omp.threadprivate %[[DECL_03]]#1 : !fir.ref<i32> -> !fir.ref<i32>
      !CHECK:     %[[TP_DECL_04:.*]]:2 = hlfir.declare %[[TP_04]] {uniq_name = "_QFsub_02Ea"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
      print *, a
      !CHECK:     omp.terminator
    !$omp end parallel
  end subroutine
end subroutine

module mod_01
  integer, save :: a
  !CHECK: fir.global @_QMmod_01Ea : i32
  !$omp threadprivate(a)
end module

!CHECK-LABEL: func.func @_QPsub_05()
subroutine sub_05()
  use mod_01, only: a
  !$omp parallel default(private)
    !CHECK:   omp.parallel {
    !CHECK:     %[[TP_05:.*]] = omp.threadprivate %{{.*}} : !fir.ref<i32> -> !fir.ref<i32>
    !CHECK:     %{{.*}} = hlfir.declare %[[TP_05]] {uniq_name = "_QMmod_01Ea"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
      print *, a
    !CHECK:     omp.terminator
  !$omp end parallel
end subroutine


!CHECK:  fir.global internal @_QFsubEa : i32

!CHECK: fir.global internal @_QFsub_02Ea : i32
