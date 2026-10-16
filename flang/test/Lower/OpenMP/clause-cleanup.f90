!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

subroutine test1(a)
integer :: a(:)

!$omp parallel num_threads(count(a .eq. 1))
print *, "don't optimize me"
!$omp end parallel
end subroutine

! CHECK:     %[[EXPR:.*]] = hlfir.elemental {{.*}} -> !hlfir.expr<?x!fir.logical<4>>
! CHECK:     %[[COUNT:.*]] = hlfir.count %[[EXPR]]
! CHECK:     omp.parallel num_threads(%[[COUNT]] : i32) {
! CHECK-NOT:   hlfir.destroy %[[EXPR]]
! CHECK:     omp.terminator
! CHECK:    }
! CHECK:    hlfir.destroy %[[EXPR]]
