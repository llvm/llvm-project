
! KGEN-generated Fortran source file
!
! Filename    : rrlw_kg08.f90
! Generated at: 2015-07-06 23:28:43
! KGEN version: 0.4.13



    MODULE rrlw_kg08
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !      use parkind ,only : jpim, jprb
        IMPLICIT NONE
        !-----------------------------------------------------------------
        ! rrtmg_lw ORIGINAL abs. coefficients for interval 8
        ! band 8:  1080-1180 cm-1 (low (i.e.>~300mb) - h2o; high - o3)
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
        ! kao_mco2: real
        ! kbo_mco2: real
        ! kao_mn2o: real
        ! kbo_mn2o: real
        ! kao_mo3 : real
        ! selfrefo: real
        ! forrefo : real
        ! cfc12o  : real
        !cfc22adjo: real
        !-----------------------------------------------------------------
        !-----------------------------------------------------------------
        ! rrtmg_lw COMBINED abs. coefficients for interval 8
        ! band 8:  1080-1180 cm-1 (low (i.e.>~300mb) - h2o; high - o3)
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
        ! ka_mco2 : real
        ! kb_mco2 : real
        ! ka_mn2o : real
        ! kb_mn2o : real
        ! ka_mo3  : real
        ! selfref : real
        ! forref  : real
        ! cfc12   : real
        ! cfc22adj: real
        !
        ! absa    : real
        ! absb    : real
        !-----------------------------------------------------------------
        INTEGER, parameter :: ng8  = 8
        REAL(KIND=r8), dimension(ng8) :: fracrefa
        REAL(KIND=r8), dimension(ng8) :: fracrefb
        REAL(KIND=r8), dimension(ng8) :: cfc12
        REAL(KIND=r8), dimension(ng8) :: cfc22adj
        REAL(KIND=r8) :: absa(65,ng8)
        REAL(KIND=r8) :: absb(235,ng8)
        REAL(KIND=r8) :: ka_mco2(19,ng8)
        REAL(KIND=r8) :: ka_mn2o(19,ng8)
        REAL(KIND=r8) :: ka_mo3(19,ng8)
        REAL(KIND=r8) :: kb_mco2(19,ng8)
        REAL(KIND=r8) :: kb_mn2o(19,ng8)
        REAL(KIND=r8) :: selfref(10,ng8)
        REAL(KIND=r8) :: forref(4,ng8)
        PUBLIC kgen_read_externs_rrlw_kg08
    CONTAINS

    ! write subroutines

    ! module extern variables

    SUBROUTINE kgen_read_externs_rrlw_kg08(kgen_unit)
        INTEGER, INTENT(IN) :: kgen_unit
        READ(UNIT=kgen_unit) fracrefa
        READ(UNIT=kgen_unit) fracrefb
        READ(UNIT=kgen_unit) cfc12
        READ(UNIT=kgen_unit) cfc22adj
        READ(UNIT=kgen_unit) absa
        READ(UNIT=kgen_unit) absb
        READ(UNIT=kgen_unit) ka_mco2
        READ(UNIT=kgen_unit) ka_mn2o
        READ(UNIT=kgen_unit) ka_mo3
        READ(UNIT=kgen_unit) kb_mco2
        READ(UNIT=kgen_unit) kb_mn2o
        READ(UNIT=kgen_unit) selfref
        READ(UNIT=kgen_unit) forref
    END SUBROUTINE kgen_read_externs_rrlw_kg08

    END MODULE rrlw_kg08
