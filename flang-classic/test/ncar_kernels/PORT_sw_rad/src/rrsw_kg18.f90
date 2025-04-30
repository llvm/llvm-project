
! KGEN-generated Fortran source file
!
! Filename    : rrsw_kg18.f90
! Generated at: 2015-07-07 00:48:25
! KGEN version: 0.4.13



    MODULE rrsw_kg18
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !      use parkind ,only : jpim, jprb
        USE parrrsw, ONLY: ng18
        IMPLICIT NONE
        !-----------------------------------------------------------------
        ! rrtmg_sw ORIGINAL abs. coefficients for interval 18
        ! band 18:  4000-4650 cm-1 (low - h2o,ch4; high - ch4)
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
        !-----------------------------------------------------------------
        INTEGER :: layreffr
        REAL(KIND=r8) :: strrat
        REAL(KIND=r8) :: rayl
        !-----------------------------------------------------------------
        ! rrtmg_sw COMBINED abs. coefficients for interval 18
        ! band 18:  4000-4650 cm-1 (low - h2o,ch4; high - ch4)
        !
        ! Initial version:  JJMorcrette, ECMWF, oct1999
        ! Revised: MJIacono, AER, jul2006
        !-----------------------------------------------------------------
        !
        !  name     type     purpose
        !  ----   : ----   : ---------------------------------------------
        ! ka      : real
        ! kb      : real
        ! absa    : real
        ! absb    : real
        ! selfref : real
        ! forref  : real
        ! sfluxref: real
        !-----------------------------------------------------------------
        REAL(KIND=r8) :: absa(585,ng18)
        REAL(KIND=r8) :: absb(235,ng18)
        REAL(KIND=r8) :: forref(3,ng18)
        REAL(KIND=r8) :: selfref(10,ng18)
        REAL(KIND=r8) :: sfluxref(ng18,9)
        PUBLIC kgen_read_externs_rrsw_kg18
    CONTAINS

    ! write subroutines

    ! module extern variables

    SUBROUTINE kgen_read_externs_rrsw_kg18(kgen_unit)
        INTEGER, INTENT(IN) :: kgen_unit
        READ(UNIT=kgen_unit) layreffr
        READ(UNIT=kgen_unit) strrat
        READ(UNIT=kgen_unit) rayl
        READ(UNIT=kgen_unit) absa
        READ(UNIT=kgen_unit) absb
        READ(UNIT=kgen_unit) forref
        READ(UNIT=kgen_unit) selfref
        READ(UNIT=kgen_unit) sfluxref
    END SUBROUTINE kgen_read_externs_rrsw_kg18

    END MODULE rrsw_kg18
