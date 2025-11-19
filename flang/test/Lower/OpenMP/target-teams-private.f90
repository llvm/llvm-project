! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --enable-delayed-privatization \
! RUN:   -o - %s 2>&1 | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp --enable-delayed-privatization -o - %s 2>&1 |\
! RUN:   FileCheck %s

!===============================================================================
! `private` clause on `target teams`
!===============================================================================

subroutine target_teams_private()
integer, dimension(3) :: i
!$omp target teams private(i)
!$omp end target teams
end subroutine

! CHECK: omp.target {{.*}} {
! CHECK:   omp.teams {
! CHECK:     %{{.*}} = fir.alloca !fir.array<3xi32> {bindc_name = "i", {{.*}}}
! CHECK:   }
! CHECK: }
