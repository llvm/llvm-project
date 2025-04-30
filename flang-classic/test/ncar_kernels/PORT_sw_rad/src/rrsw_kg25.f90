
! KGEN-generated Fortran source file
!
! Filename    : rrsw_kg25.f90
! Generated at: 2015-07-07 00:48:23
! KGEN version: 0.4.13



    MODULE rrsw_kg25
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !      use parkind ,only : jpim, jprb
        USE parrrsw, ONLY: ng25
        IMPLICIT NONE
        !-----------------------------------------------------------------
        ! rrtmg_sw ORIGINAL abs. coefficients for interval 25
        ! band 25: 16000-22650 cm-1 (low - h2o; high - nothing)
        !
        ! Initial version:  JJMorcrette, ECMWF, oct1999
        ! Revised: MJIacono, AER, jul2006
        !-----------------------------------------------------------------
        !
        !  name     type     purpose
        !  ----   : ----   : ---------------------------------------------
        ! kao     : real
        !sfluxrefo: real
        ! abso3ao : real
        ! abso3bo : real
        ! raylo   : real
        !-----------------------------------------------------------------
        INTEGER :: layreffr
        !-----------------------------------------------------------------
        ! rrtmg_sw COMBINED abs. coefficients for interval 25
        ! band 25: 16000-22650 cm-1 (low - h2o; high - nothing)
        !
        ! Initial version:  JJMorcrette, ECMWF, oct1999
        ! Revised: MJIacono, AER, jul2006
        !-----------------------------------------------------------------
        !
        !  name     type     purpose
        !  ----   : ----   : ---------------------------------------------
        ! ka      : real
        ! absa    : real
        ! sfluxref: real
        ! abso3a  : real
        ! abso3b  : real
        ! rayl    : real
        !-----------------------------------------------------------------
        REAL(KIND=r8) :: absa(65,ng25)
        REAL(KIND=r8) :: sfluxref(ng25)
        REAL(KIND=r8) :: abso3a(ng25)
        REAL(KIND=r8) :: abso3b(ng25)
        REAL(KIND=r8) :: rayl(ng25)
        PUBLIC kgen_read_externs_rrsw_kg25
    CONTAINS

    ! write subroutines

    ! module extern variables

    SUBROUTINE kgen_read_externs_rrsw_kg25(kgen_unit)
        INTEGER, INTENT(IN) :: kgen_unit
        READ(UNIT=kgen_unit) layreffr
        READ(UNIT=kgen_unit) absa
        READ(UNIT=kgen_unit) sfluxref
        READ(UNIT=kgen_unit) abso3a
        READ(UNIT=kgen_unit) abso3b
        READ(UNIT=kgen_unit) rayl
    END SUBROUTINE kgen_read_externs_rrsw_kg25

    END MODULE rrsw_kg25
