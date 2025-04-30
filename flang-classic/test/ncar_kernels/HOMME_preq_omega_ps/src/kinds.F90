
! KGEN-generated Fortran source file
!
! Filename    : kinds.F90
! Generated at: 2015-03-16 09:25:32
! KGEN version: 0.4.5



    MODULE kinds
        IMPLICIT NONE
        PRIVATE
        !
        !  most floating point variables should be of type real_kind = real*8
        !  For higher precision, we also have quad_kind = real*16, but this
        !  is only supported on IBM systems
        !
        INTEGER(KIND=4), public, parameter :: real_kind    = 8
        ! stderr file handle

    ! write subroutines
    ! No subroutines
    ! No module extern variables
    END MODULE kinds
