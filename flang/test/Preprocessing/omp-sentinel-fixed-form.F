! RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=50 %s | FileCheck %s

#define OMP_TARGET        .true.
#define OMP_SIMD    .false.
      program test
      implicit none
      integer     i,j,n
      n = 100
! CHECK: !$OMP METADIRECTIVE  WHEN(USER={CONDITION(.true._4)}: TARGET TEAMS DISTRIBUTE PARALLEL&
! CHECK: !$OMP& DO) DEFAULT(TARGET TEAMS LOOP)
!$omp  metadirective
!$omp& when(user={condition(OMP_TARGET.or.OMP_SIMD)}:
!$omp&      target teams distribute parallel do )
!$omp& default(target teams loop)
      do i=0,n
        do j=0,n
          write(*,*) "Test"
        enddo
      enddo
      return
      end program
