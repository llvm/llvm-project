
! KGEN-generated Fortran source file
!
! Filename    : kinds.F90
! Generated at: 2015-04-12 19:37:49
! KGEN version: 0.4.9



    MODULE kinds
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: shr_kind_i4
        USE shr_kind_mod, ONLY: shr_kind_i8
        USE shr_kind_mod, ONLY: shr_kind_r8
        ! _EXTERNAL
        IMPLICIT NONE
        PRIVATE
        !
        !  most floating point variables should be of type real_kind = real*8
        !  For higher precision, we also have quad_kind = real*16, but this
        !  is only supported on IBM systems
        !
        INTEGER(KIND=4), public, parameter :: real_kind    = shr_kind_r8
        INTEGER(KIND=4), public, parameter :: int_kind     = shr_kind_i4
        INTEGER(KIND=4), public, parameter :: log_kind     = kind(.true.)
        INTEGER(KIND=4), public, parameter :: long_kind    = shr_kind_i8

    ! write subroutines
    ! No subroutines
    ! No module extern variables
    END MODULE kinds
