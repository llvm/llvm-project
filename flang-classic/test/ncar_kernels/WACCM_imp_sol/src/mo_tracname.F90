
! KGEN-generated Fortran source file
!
! Filename    : mo_tracname.F90
! Generated at: 2015-05-13 11:02:21
! KGEN version: 0.4.10



    MODULE mo_tracname
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        !-----------------------------------------------------------
        !       ... List of advected and non-advected trace species, and
        !           surface fluxes for the advected species.
        !-----------------------------------------------------------
        USE chem_mods, ONLY: gas_pcnst
        IMPLICIT NONE
        CHARACTER(LEN=16) :: solsym(gas_pcnst) ! species names
        PUBLIC kgen_read_externs_mo_tracname
    CONTAINS

    ! write subroutines

    ! module extern variables

    SUBROUTINE kgen_read_externs_mo_tracname(kgen_unit)
        INTEGER, INTENT(IN) :: kgen_unit
        READ(UNIT=kgen_unit) solsym
    END SUBROUTINE kgen_read_externs_mo_tracname

    END MODULE mo_tracname
