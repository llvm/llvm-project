!RUN: %flang_fc1 -fdebug-unparse-with-symbols -fopenmp -fopenmp-version=60 %s | FileCheck %s

! Shortened version of Fujitsu/Fortran/0160/0160_0000.f90
! Make sure that j is privatized.

!CHECK-LABEL: !DEF: /MAIN MainProgram
!CHECK-NEXT:  program MAIN
!CHECK-NEXT:   implicit none
!CHECK-NEXT:   !DEF: /MAIN/j ObjectEntity INTEGER(4)
!CHECK-NEXT:   !DEF: /MAIN/k ObjectEntity INTEGER(4)
!CHECK-NEXT:   !DEF: /MAIN/ndim ObjectEntity INTEGER(4)
!CHECK-NEXT:   integer j, k, ndim
!CHECK-NEXT:   !DEF: /MAIN/flux (Subroutine) Subprogram
!CHECK-NEXT:   call flux
!CHECK-NEXT:  contains
!CHECK-NEXT:   !REF: /MAIN/flux
!CHECK-NEXT:   subroutine flux
!CHECK-NEXT:  !$omp parallel
!CHECK-NEXT:  !$omp do
!CHECK-NEXT:    !DEF: /MAIN/flux/OtherConstruct1/OtherConstruct1/k (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
!CHECK-NEXT:    !DEF: /MAIN/flux/OtherConstruct1/OtherConstruct1/ndim HostAssoc INTEGER(4)
!CHECK-NEXT:    do k=-1,ndim+1
!CHECK-NEXT:     !DEF: /MAIN/flux/OtherConstruct1/OtherConstruct1/j (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
!CHECK-NEXT:     !REF: /MAIN/flux/OtherConstruct1/OtherConstruct1/ndim
!CHECK-NEXT:     do j=-1,ndim+1
!CHECK-NEXT:     end do
!CHECK-NEXT:    end do
!CHECK-NEXT:  !$omp end do
!CHECK-NEXT:  !$omp end parallel
!CHECK-NEXT:   end subroutine flux
!CHECK-NEXT:  end program MAIN

program main
  implicit none
  integer :: j, k, ndim

  call flux()

  contains

    subroutine flux
      !$omp parallel
      !$omp do
      do k = -1, ndim + 1
        do j = -1, ndim + 1
        enddo
      enddo
      !$omp end do
      !$omp end parallel
    end subroutine flux

end program main
