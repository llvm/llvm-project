! RUN: %flang_fc1 -fopenmp -fopenmp-version=60 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPp_scalar
! CHECK: omp.parallel private(@_QFp_scalarEs_private_i32 %{{.*}} -> %{{.*}} : !fir.ref<{{.*}}>) {
subroutine p_scalar(s)
  integer :: s
  !$omp parallel default(private: scalar)
    s = 1
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPp_aggregate
! CHECK: omp.parallel private(@_QFp_aggregateEarr_private_10xi32 %{{.*}} -> %{{.*}}, @_QFp_aggregateEchar_private_c8 %{{.*}} -> %{{.*}} : !fir.ref<{{.*}}>) {
subroutine p_aggregate(arr, char)
  integer :: arr(10)
  character :: char
  !$omp parallel default(private: aggregate)
    arr = 1
    char = 'A'
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPp_pointer
! CHECK: omp.parallel private(@_QFp_pointerEp_private_box_ptr_i32 %{{.*}} -> %{{.*}} : !fir.ref<{{.*}}>) {
subroutine p_pointer(p)
  integer, pointer :: p
  !$omp parallel default(private: pointer)
    p = 1
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPp_allocatable
! CHECK: omp.parallel private(@_QFp_allocatableEal_private_box_heap_Uxi32 %{{.*}} -> %{{.*}} : !fir.ref<{{.*}}>) {
subroutine p_allocatable(s, al)
  integer :: s
  integer, allocatable :: al(:)
  !$omp parallel default(private: allocatable)
    s = size(al)
  !$omp end parallel
end subroutine
