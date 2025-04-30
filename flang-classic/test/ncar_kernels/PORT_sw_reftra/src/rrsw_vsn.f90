
! KGEN-generated Fortran source file
!
! Filename    : rrsw_vsn.f90
! Generated at: 2015-07-31 20:52:25
! KGEN version: 0.4.13



    MODULE rrsw_vsn
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        !      use parkind, only : jpim, jprb
        IMPLICIT NONE
        !------------------------------------------------------------------
        ! rrtmg_sw version information
        ! Initial version:  JJMorcrette, ECMWF, jul1998
        ! Revised: MJIacono, AER, jul2006
        !------------------------------------------------------------------
        !  name     type     purpose
        ! -----  :  ----   : ----------------------------------------------
        !hnamrtm :character:
        !hnamini :character:
        !hnamcld :character:
        !hnamclc :character:
        !hnamrft :character:
        !hnamspv :character:
        !hnamspc :character:
        !hnamset :character:
        !hnamtau :character:
        !hnamvqd :character:
        !hnamatm :character:
        !hnamutl :character:
        !hnamext :character:
        !hnamkg  :character:
        !
        ! hvrrtm :character:
        ! hvrini :character:
        ! hvrcld :character:
        ! hvrclc :character:
        ! hvrrft :character:
        ! hvrspv :character:
        ! hvrspc :character:
        ! hvrset :character:
        ! hvrtau :character:
        ! hvrvqd :character:
        ! hvratm :character:
        ! hvrutl :character:
        ! hvrext :character:
        ! hvrkg  :character:
        !------------------------------------------------------------------
        CHARACTER(LEN=18) :: hvrrft
        PUBLIC kgen_read_externs_rrsw_vsn
    CONTAINS

    ! write subroutines
    ! No subroutines

    ! module extern variables

    SUBROUTINE kgen_read_externs_rrsw_vsn(kgen_unit)
        INTEGER, INTENT(IN) :: kgen_unit
        READ(UNIT=kgen_unit) hvrrft
    END SUBROUTINE kgen_read_externs_rrsw_vsn

    END MODULE rrsw_vsn
