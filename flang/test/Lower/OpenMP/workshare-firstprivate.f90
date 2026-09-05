! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s --check-prefix HLFIR
! RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s --check-prefix FIR

! firstprivate on "parallel workshare" is applied to the parallel leaf, so it
! uses the standard omp.private delayed privatization. After the LowerWorkshare
! pass the private clause is preserved on omp.parallel; array assignments are
! workshared (omp.wsloop) while FORALL runs in omp.single.

! HLFIR: omp.private {type = firstprivate} @{{.*}}_firstprivate_box_ptr_Uxi32 : !fir.box<!fir.ptr<!fir.array<?xi32>>>
! HLFIR: omp.private {type = firstprivate} @{{.*}}_firstprivate_box_heap_Uxi32 : !fir.box<!fir.heap<!fir.array<?xi32>>>
! HLFIR: hlfir.assign %{{.*}} to %{{.*}} realloc

! Pointer firstprivate, array assignment -> workshared loop.
subroutine test_ptr(p)
  integer, pointer, intent(in) :: p(:)
  integer :: a(4)
  !$omp parallel workshare firstprivate(p)
    a = p + 1
  !$omp end parallel workshare
end subroutine

! FIR-LABEL: func.func @{{.*}}test_ptr(
! FIR:         omp.parallel private(@{{.*}}test_ptrEp_firstprivate{{.*}} -> %{{.*}} :
! FIR-SAME:        !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) {
! FIR:           omp.wsloop

! Allocatable firstprivate, array assignment -> workshared loop.
subroutine test_alloc(p)
  integer, allocatable :: p(:)
  integer :: a(4)
  !$omp parallel workshare firstprivate(p)
    a = p + 1
  !$omp end parallel workshare
end subroutine

! FIR-LABEL: func.func @{{.*}}test_alloc(
! FIR:         omp.parallel private(@{{.*}}test_allocEp_firstprivate{{.*}} -> %{{.*}} :
! FIR-SAME:        !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) {
! FIR:           omp.wsloop

! FORALL body runs in omp.single (not workshared).
subroutine test_forall(p)
  integer, pointer, intent(in) :: p(:)
  integer :: i
  !$omp parallel workshare firstprivate(p)
    forall (i=1:size(p)) p(i) = i*i
  !$omp end parallel workshare
end subroutine

! FIR-LABEL: func.func @{{.*}}test_forall(
! FIR:         omp.parallel private(@{{.*}}test_forallEp_firstprivate{{.*}} -> %{{.*}} :
! FIR-SAME:        !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) {
! FIR:           omp.single
