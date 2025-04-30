
! KGEN-generated Fortran source file
!
! Filename    : physical_constants.F90
! Generated at: 2015-04-12 19:17:34
! KGEN version: 0.4.9



    MODULE physical_constants
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        ! ------------------------------
        USE physconst, ONLY: rrearth => ra ! _EXTERNAL
        ! -----------------------------
        IMPLICIT NONE
        PRIVATE
        ! m s^-2
        ! m
        ! s^-1
        ! Pa
        PUBLIC rrearth ! m

    ! write subroutines
    ! No subroutines
    ! No module extern variables
    END MODULE physical_constants
