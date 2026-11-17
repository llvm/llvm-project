! RUN: %flang_fc1 -fopenmp -ffixed-form -fdebug-unparse %s 2>&1 | FileCheck %s --ignore-case
! RUN: %flang_fc1 -fopenmp -ffixed-form -fdebug-dump-parse-tree %s 2>&1 | FileCheck %s --check-prefix=TREE
! Exercise fixed-form OpenMP sentinels with continuation and column-72 padding.
      subroutine sub1
!23456789012345678901234567890123456789012345678901234567890123456789012
*$omp   paral
c$ompxlel s
c$ompyections n
!$ompz u m _ t h r e a d s ( 2 )
!$omp   section
*$      print *,'in section'
!$omp   end parallel sections
      end subroutine
      subroutine sub2
!$omp sections
!$omp sect
!$omp  ! should be ignored, and make the continuation correct
!$omp+ion
      print *,'ok'
!$omp end sections
      end subroutine
      program main
      call sub1
      call sub2
      end program

!CHECK: !$omp parallel sections num_threads(2_4)
!CHECK: !$omp section
!CHECK: !$omp end parallel sections
!CHECK: !$omp sections{{$}}
!CHECK: !$omp section{{$}}
!CHECK: !$omp end sections{{$}}

!TREE: OpenMPSectionsConstruct
!TREE: OmpSectionDirective
!TREE: OpenMPSectionsConstruct
!TREE: OmpSectionDirective
