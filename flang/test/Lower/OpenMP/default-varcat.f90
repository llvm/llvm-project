! RUN: %flang_fc1 -fopenmp -fopenmp-version=60 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPp_scalar
! CHECK: omp.parallel private(@_QFp_scalarEs_private_i32 %{{.*}} -> %{{.*}} : !fir.ref<i32>) {
subroutine p_scalar( s, arr, char, p, al )
  integer :: s
  integer :: arr(10)
  character :: char
  integer, pointer :: p
  integer, allocatable :: al(:)
  !$omp parallel default(private: scalar)
    s = size(al)
    arr = 1
    char = 'A'
    p = 1
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPp_scalar_task
! CHECK: omp.task private(@_QFp_scalar_taskEs_private_i32 %{{.*}} -> %{{.*}}, @_QFp_scalar_taskEal_firstprivate_box_heap_Uxi32 %{{.*}} -> %{{.*}}, @_QFp_scalar_taskEarr_firstprivate_box_10xi32 %{{.*}} -> %{{.*}}, @_QFp_scalar_taskEchar_firstprivate_c8 %{{.*}} -> %{{.*}}, @_QFp_scalar_taskEp_firstprivate_box_ptr_i32 %{{.*}} -> %{{.*}} : !fir.ref<i32>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.array<10xi32>>>, !fir.ref<!fir.char<1>>, !fir.ref<!fir.box<!fir.ptr<i32>>>) {
subroutine p_scalar_task( s, arr, char, p, al )
  integer :: s
  integer :: arr(10)
  character :: char
  integer, pointer :: p
  integer, allocatable :: al(:)
  !$omp task default(private: scalar)
    s = size(al)
    arr = 1
    char = 'A'
    p = 1
  !$omp end task
end subroutine

! CHECK-LABEL: func.func @_QPp_aggregate
! CHECK: omp.parallel private(@_QFp_aggregateEarr_private_10xi32 %{{.*}} -> %{{.*}}, @_QFp_aggregateEchar_private_c8 %{{.*}} -> %{{.*}} : !fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.char<1>>) {
subroutine p_aggregate( s, arr, char, p, al )
  integer :: s
  integer :: arr(10)
  character :: char
  integer, pointer :: p
  integer, allocatable :: al(:)
  !$omp parallel default(private: aggregate)
    s = size(al)
    arr = 1
    char = 'A'
    p = 1
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPp_pointer
! CHECK: omp.parallel private(@_QFp_pointerEp_private_box_ptr_i32 %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.ptr<i32>>>) {
subroutine p_pointer( s, arr, char, p, al )

  integer :: s
  integer :: arr(10)
  character :: char
  integer, pointer :: p
  integer, allocatable :: al(:)
  !$omp parallel default(private: pointer)
    s = size(al)
    arr = 1
    char = 'A'
    p = 1
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPp_allocatable
! CHECK: omp.parallel private(@_QFp_allocatableEal_private_box_heap_Uxi32 %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) {
subroutine p_allocatable( s, arr, char, p, al )
  integer :: s
  integer :: arr(10)
  character :: char
  integer, pointer :: p
  integer, allocatable :: al(:)
  !$omp parallel default(private: allocatable)
    s = size(al)
    arr = 1
    char = 'A'
    p = 1
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPfp_scalar
! CHECK: omp.parallel private(@_QFfp_scalarEs_firstprivate_i32 %{{.*}} -> %{{.*}} : !fir.ref<i32>) {
subroutine fp_scalar( s, arr, char, p, al )

  integer :: s
  integer :: arr(10)
  character :: char
  integer, pointer :: p
  integer, allocatable :: al(:)
  !$omp parallel default(firstprivate: scalar)
    s = size(al)
    arr = 1
    char = 'A'
    p = 1
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPfp_aggregate
! CHECK: omp.parallel private(@_QFfp_aggregateEarr_firstprivate_box_10xi32 %{{.*}} -> %{{.*}}, @_QFfp_aggregateEchar_firstprivate_c8 %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.array<10xi32>>>, !fir.ref<!fir.char<1>>) {
subroutine fp_aggregate( s, arr, char, p, al )
  integer :: s
  integer :: arr(10)
  character :: char
  integer, pointer :: p
  integer, allocatable :: al(:)
  !$omp parallel default(firstprivate: aggregate)
    s = size(al)
    arr = 1
    char = 'A'
    p = 1
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPfp_pointer
! CHECK: omp.parallel private(@_QFfp_pointerEp_firstprivate_box_ptr_i32 %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.ptr<i32>>>) {
subroutine fp_pointer( s, arr, char, p, al )
  integer :: s
  integer :: arr(10)
  character :: char
  integer, pointer :: p
  integer, allocatable :: al(:)
  !$omp parallel default(firstprivate: pointer)
    s = size(al)
    arr = 1
    char = 'A'
    p = 1
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPfp_allocatable
! CHECK: omp.parallel private(@_QFfp_allocatableEal_firstprivate_box_heap_Uxi32 %{{.*}} -> %{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) {
subroutine fp_allocatable( s, arr, char, p, al )
  integer :: s
  integer :: arr(10)
  character :: char
  integer, pointer :: p
  integer, allocatable :: al(:)
  !$omp parallel default(firstprivate: allocatable)
    s = size(al)
    arr = 1
    char = 'A'
    p = 1
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPpar_do_aggregate
! CHECK: omp.parallel private(@_QFpar_do_aggregateEarr_private_10xi32 %{{.*}} -> %{{.*}} : !fir.ref<!fir.array<10xi32>>) {
! CHECK: omp.wsloop private(@_QFpar_do_aggregateEs_private_i32 %{{.*}} -> %{{.*}}, @_QFpar_do_aggregateEi_private_i32 %{{.*}} -> %{{.*}} : !fir.ref<i32>, !fir.ref<i32>) {
subroutine par_do_aggregate( s, arr )
  integer :: s
  integer :: arr(10)
  !$omp parallel do default(private: aggregate) private(s)
  do i = 1, 10
    s = i
    arr(i) = s
  end do
end subroutine
