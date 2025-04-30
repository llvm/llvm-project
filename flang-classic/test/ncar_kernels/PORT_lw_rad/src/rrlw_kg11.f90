
! KGEN-generated Fortran source file
!
! Filename    : rrlw_kg11.f90
! Generated at: 2015-07-06 23:28:45
! KGEN version: 0.4.13



    MODULE rrlw_kg11
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !      use parkind ,only : jpim, jprb
        IMPLICIT NONE
        !-----------------------------------------------------------------
        ! rrtmg_lw ORIGINAL abs. coefficients for interval 11
        ! band 11:  1480-1800 cm-1 (low - h2o; high - h2o)
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
        ! kao_mo2 : real
        ! kbo_mo2 : real
        ! selfrefo: real
        ! forrefo : real
        !-----------------------------------------------------------------
        !-----------------------------------------------------------------
        ! rrtmg_lw COMBINED abs. coefficients for interval 11
        ! band 11:  1480-1800 cm-1 (low - h2o; high - h2o)
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
        ! ka_mo2  : real
        ! kb_mo2  : real
        ! selfref : real
        ! forref  : real
        !
        ! absa    : real
        ! absb    : real
        !-----------------------------------------------------------------
        INTEGER, parameter :: ng11 = 8
        REAL(KIND=r8), dimension(ng11) :: fracrefa
        REAL(KIND=r8), dimension(ng11) :: fracrefb
        REAL(KIND=r8) :: absa(65,ng11)
        REAL(KIND=r8) :: absb(235,ng11)
        REAL(KIND=r8) :: ka_mo2(19,ng11)
        REAL(KIND=r8) :: kb_mo2(19,ng11)
        REAL(KIND=r8) :: selfref(10,ng11)
        REAL(KIND=r8) :: forref(4,ng11)
        PUBLIC kgen_read_externs_rrlw_kg11
    CONTAINS

    ! write subroutines

    ! module extern variables

    SUBROUTINE kgen_read_externs_rrlw_kg11(kgen_unit)
        INTEGER, INTENT(IN) :: kgen_unit
        READ(UNIT=kgen_unit) fracrefa
        READ(UNIT=kgen_unit) fracrefb
        READ(UNIT=kgen_unit) absa
        READ(UNIT=kgen_unit) absb
        READ(UNIT=kgen_unit) ka_mo2
        READ(UNIT=kgen_unit) kb_mo2
        READ(UNIT=kgen_unit) selfref
        READ(UNIT=kgen_unit) forref
    END SUBROUTINE kgen_read_externs_rrlw_kg11

    END MODULE rrlw_kg11
