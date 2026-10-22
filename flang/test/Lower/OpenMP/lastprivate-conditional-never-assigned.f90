! Test lowering of `lastprivate(conditional:)` where one list item (z) is never
! assigned in the region.  The lowering must statically guarantee such a
! variable is not copied back: its index field is initialized to -1 and no
! iteration-index store is emitted for it, so the guarded copy-back can never
! fire.  The assigned variable (x) behaves normally.

! RUN: bbc -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s

subroutine test_conditional_lp_never_assigned(n, x, z)
  implicit none
  integer, intent(in) :: n
  integer :: x, z
  integer :: k

  !$omp parallel do lastprivate(conditional: x, z)
  do k = 1, n
    if (mod(k, 2) == 0) x = k     ! x is assigned; z is never assigned
  end do
  !$omp end parallel do
end subroutine

! CHECK-LABEL: omp.declare_reduction @lp_cond_byref_rec__lp_cond_t
! CHECK-SAME:    {x:i32,z:i32,$x:i64,$z:i64}
! Both index fields start at the -1 sentinel.
! CHECK:       init {
! CHECK-DAG:     arith.constant -1 : i64
! CHECK:       }

! CHECK-LABEL: func.func @_QPtest_conditional_lp_never_assigned
! CHECK:         omp.loop_nest
! The assigned variable x records its canonical iteration index; the
! never-assigned variable z records no index store at all.
! CHECK:           fir.coordinate_of %{{.*}}, $x
! CHECK-NOT:       fir.coordinate_of %{{.*}}, $z
! CHECK:         omp.single {
! Both variables still get a guarded copy-back; z's guard can never fire
! because its index remains -1.
! CHECK:           arith.cmpi sge, %{{.*}}, %{{.*}} : i64
! CHECK:           arith.cmpi sge, %{{.*}}, %{{.*}} : i64
! CHECK:           omp.terminator
! CHECK:         }
