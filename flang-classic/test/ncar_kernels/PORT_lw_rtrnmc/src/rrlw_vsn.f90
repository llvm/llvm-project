
! KGEN-generated Fortran source file
!
! Filename    : rrlw_vsn.f90
! Generated at: 2015-07-26 20:37:04
! KGEN version: 0.4.13



    MODULE rrlw_vsn
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        !      use parkind, only : jpim, jprb
        IMPLICIT NONE
        !------------------------------------------------------------------
        ! rrtmg_lw version information
        ! Initial version:  JJMorcrette, ECMWF, jul1998
        ! Revised: MJIacono, AER, jun2006
        !------------------------------------------------------------------
        !  name     type     purpose
        ! -----  :  ----   : ----------------------------------------------
        !hnamrtm :character:
        !hnamini :character:
        !hnamcld :character:
        !hnamclc :character:
        !hnamrtr :character:
        !hnamrtx :character:
        !hnamrtc :character:
        !hnamset :character:
        !hnamtau :character:
        !hnamatm :character:
        !hnamutl :character:
        !hnamext :character:
        !hnamkg  :character:
        !
        ! hvrrtm :character:
        ! hvrini :character:
        ! hvrcld :character:
        ! hvrclc :character:
        ! hvrrtr :character:
        ! hvrrtx :character:
        ! hvrrtc :character:
        ! hvrset :character:
        ! hvrtau :character:
        ! hvratm :character:
        ! hvrutl :character:
        ! hvrext :character:
        ! hvrkg  :character:
        !------------------------------------------------------------------
        CHARACTER(LEN=18) :: hvrrtc
        PUBLIC kgen_read_externs_rrlw_vsn
    CONTAINS

    ! write subroutines
    ! No subroutines

    ! module extern variables

    SUBROUTINE kgen_read_externs_rrlw_vsn(kgen_unit)
        INTEGER, INTENT(IN) :: kgen_unit
        READ(UNIT=kgen_unit) hvrrtc
    END SUBROUTINE kgen_read_externs_rrlw_vsn

    END MODULE rrlw_vsn
