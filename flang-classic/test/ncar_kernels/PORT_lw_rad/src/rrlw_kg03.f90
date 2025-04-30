
! KGEN-generated Fortran source file
!
! Filename    : rrlw_kg03.f90
! Generated at: 2015-07-06 23:28:44
! KGEN version: 0.4.13



    MODULE rrlw_kg03
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !      use parkind ,only : jpim, jprb
        IMPLICIT NONE
        !-----------------------------------------------------------------
        ! rrtmg_lw ORIGINAL abs. coefficients for interval 3
        ! band 3:  500-630 cm-1 (low - h2o,co2; high - h2o,co2)
        !
        ! Initial version:  JJMorcrette, ECMWF, jul1998
        ! Revised: MJIacono, AER, jun2006
        !-----------------------------------------------------------------
        !
        !  name     type     purpose
        !  ----   : ----   : ---------------------------------------------
        !fracrefao: real
        !fracrefbo: real
        ! kao     : real
        ! kbo     : real
        ! kao_mn2o: real
        ! kbo_mn2o: real
        ! selfrefo: real
        ! forrefo : real
        !-----------------------------------------------------------------
        !-----------------------------------------------------------------
        ! rrtmg_lw COMBINED abs. coefficients for interval 3
        ! band 3:  500-630 cm-1 (low - h2o,co2; high - h2o,co2)
        !
        ! Initial version:  JJMorcrette, ECMWF, jul1998
        ! Revised: MJIacono, AER, jun2006
        !-----------------------------------------------------------------
        !
        !  name     type     purpose
        !  ----   : ----   : ---------------------------------------------
        !fracrefa : real
        !fracrefb : real
        ! ka      : real
        ! kb      : real
        ! ka_mn2o : real
        ! kb_mn2o : real
        ! selfref : real
        ! forref  : real
        !
        ! absa    : real
        ! absb    : real
        !-----------------------------------------------------------------
        INTEGER, parameter :: ng3  = 16
        REAL(KIND=r8) :: fracrefa(ng3,10)
        REAL(KIND=r8) :: fracrefb(ng3,5)
        REAL(KIND=r8) :: absa(585,ng3)
        REAL(KIND=r8) :: absb(1175,ng3)
        REAL(KIND=r8) :: ka_mn2o(9,19,ng3)
        REAL(KIND=r8) :: kb_mn2o(5,19,ng3)
        REAL(KIND=r8) :: selfref(10,ng3)
        REAL(KIND=r8) :: forref(4,ng3)
        PUBLIC kgen_read_externs_rrlw_kg03
    CONTAINS

    ! write subroutines

    ! module extern variables

    SUBROUTINE kgen_read_externs_rrlw_kg03(kgen_unit)
        INTEGER, INTENT(IN) :: kgen_unit
        READ(UNIT=kgen_unit) fracrefa
        READ(UNIT=kgen_unit) fracrefb
        READ(UNIT=kgen_unit) absa
        READ(UNIT=kgen_unit) absb
        READ(UNIT=kgen_unit) ka_mn2o
        READ(UNIT=kgen_unit) kb_mn2o
        READ(UNIT=kgen_unit) selfref
        READ(UNIT=kgen_unit) forref
    END SUBROUTINE kgen_read_externs_rrlw_kg03

    END MODULE rrlw_kg03
