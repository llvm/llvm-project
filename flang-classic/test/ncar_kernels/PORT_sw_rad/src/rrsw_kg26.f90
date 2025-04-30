
! KGEN-generated Fortran source file
!
! Filename    : rrsw_kg26.f90
! Generated at: 2015-07-07 00:48:25
! KGEN version: 0.4.13



    MODULE rrsw_kg26
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !      use parkind ,only : jpim, jprb
        USE parrrsw, ONLY: ng26
        IMPLICIT NONE
        !-----------------------------------------------------------------
        ! rrtmg_sw ORIGINAL abs. coefficients for interval 26
        ! band 26: 22650-29000 cm-1 (low - nothing; high - nothing)
        !
        ! Initial version:  JJMorcrette, ECMWF, oct1999
        ! Revised: MJIacono, AER, jul2006
        !-----------------------------------------------------------------
        !
        !  name     type     purpose
        !  ----   : ----   : ---------------------------------------------
        !sfluxrefo: real
        ! raylo   : real
        !-----------------------------------------------------------------
        !-----------------------------------------------------------------
        ! rrtmg_sw COMBINED abs. coefficients for interval 26
        ! band 26: 22650-29000 cm-1 (low - nothing; high - nothing)
        !
        ! Initial version:  JJMorcrette, ECMWF, oct1999
        ! Revised: MJIacono, AER, jul2006
        !-----------------------------------------------------------------
        !
        !  name     type     purpose
        !  ----   : ----   : ---------------------------------------------
        ! sfluxref: real
        ! rayl    : real
        !-----------------------------------------------------------------
        REAL(KIND=r8) :: sfluxref(ng26)
        REAL(KIND=r8) :: rayl(ng26)
        PUBLIC kgen_read_externs_rrsw_kg26
    CONTAINS

    ! write subroutines

    ! module extern variables

    SUBROUTINE kgen_read_externs_rrsw_kg26(kgen_unit)
        INTEGER, INTENT(IN) :: kgen_unit
        READ(UNIT=kgen_unit) sfluxref
        READ(UNIT=kgen_unit) rayl
    END SUBROUTINE kgen_read_externs_rrsw_kg26

    END MODULE rrsw_kg26
