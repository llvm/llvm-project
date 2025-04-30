
! KGEN-generated Fortran source file
!
! Filename    : rrsw_kg29.f90
! Generated at: 2015-07-31 20:35:44
! KGEN version: 0.4.13



    MODULE rrsw_kg29
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !      use parkind ,only : jpim, jprb
        USE parrrsw, ONLY: ng29
        IMPLICIT NONE
        !-----------------------------------------------------------------
        ! rrtmg_sw ORIGINAL abs. coefficients for interval 29
        ! band 29:  820-2600 cm-1 (low - h2o; high - co2)
        !
        ! Initial version:  JJMorcrette, ECMWF, oct1999
        ! Revised: MJIacono, AER, jul2006
        !-----------------------------------------------------------------
        !
        !  name     type     purpose
        !  ----   : ----   : ---------------------------------------------
        ! kao     : real
        ! kbo     : real
        ! selfrefo: real
        ! forrefo : real
        !sfluxrefo: real
        ! absh2oo : real
        ! absco2o : real
        !-----------------------------------------------------------------
        INTEGER :: layreffr
        REAL(KIND=r8) :: rayl
        !-----------------------------------------------------------------
        ! rrtmg_sw COMBINED abs. coefficients for interval 29
        ! band 29:  820-2600 cm-1 (low - h2o; high - co2)
        !
        ! Initial version:  JJMorcrette, ECMWF, oct1999
        ! Revised: MJIacono, AER, jul2006
        !-----------------------------------------------------------------
        !
        !  name     type     purpose
        !  ----   : ----   : ---------------------------------------------
        ! ka      : real
        ! kb      : real
        ! selfref : real
        ! forref  : real
        ! sfluxref: real
        ! absh2o  : real
        ! absco2  : real
        !-----------------------------------------------------------------
        REAL(KIND=r8) :: absa(65,ng29)
        REAL(KIND=r8) :: absb(235,ng29)
        REAL(KIND=r8) :: forref(4,ng29)
        REAL(KIND=r8) :: selfref(10,ng29)
        REAL(KIND=r8) :: sfluxref(ng29)
        REAL(KIND=r8) :: absco2(ng29)
        REAL(KIND=r8) :: absh2o(ng29)
        PUBLIC kgen_read_externs_rrsw_kg29
    CONTAINS

    ! write subroutines

    ! module extern variables

    SUBROUTINE kgen_read_externs_rrsw_kg29(kgen_unit)
        INTEGER, INTENT(IN) :: kgen_unit
        READ(UNIT=kgen_unit) layreffr
        READ(UNIT=kgen_unit) rayl
        READ(UNIT=kgen_unit) absa
        READ(UNIT=kgen_unit) absb
        READ(UNIT=kgen_unit) forref
        READ(UNIT=kgen_unit) selfref
        READ(UNIT=kgen_unit) sfluxref
        READ(UNIT=kgen_unit) absco2
        READ(UNIT=kgen_unit) absh2o
    END SUBROUTINE kgen_read_externs_rrsw_kg29

    END MODULE rrsw_kg29
