
! KGEN-generated Fortran source file
!
! Filename    : rrlw_kg09.f90
! Generated at: 2015-07-06 23:28:45
! KGEN version: 0.4.13



    MODULE rrlw_kg09
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !      use parkind ,only : jpim, jprb
        IMPLICIT NONE
        !-----------------------------------------------------------------
        ! rrtmg_lw ORIGINAL abs. coefficients for interval 9
        ! band 9:  1180-1390 cm-1 (low - h2o,ch4; high - ch4)
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
        ! rrtmg_lw COMBINED abs. coefficients for interval 9
        ! band 9:  1180-1390 cm-1 (low - h2o,ch4; high - ch4)
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
        INTEGER, parameter :: ng9  = 12
        REAL(KIND=r8), dimension(ng9) :: fracrefb
        REAL(KIND=r8) :: fracrefa(ng9,9)
        REAL(KIND=r8) :: absa(585,ng9)
        REAL(KIND=r8) :: absb(235,ng9)
        REAL(KIND=r8) :: ka_mn2o(9,19,ng9)
        REAL(KIND=r8) :: kb_mn2o(19,ng9)
        REAL(KIND=r8) :: selfref(10,ng9)
        REAL(KIND=r8) :: forref(4,ng9)
        PUBLIC kgen_read_externs_rrlw_kg09
    CONTAINS

    ! write subroutines

    ! module extern variables

    SUBROUTINE kgen_read_externs_rrlw_kg09(kgen_unit)
        INTEGER, INTENT(IN) :: kgen_unit
        READ(UNIT=kgen_unit) fracrefb
        READ(UNIT=kgen_unit) fracrefa
        READ(UNIT=kgen_unit) absa
        READ(UNIT=kgen_unit) absb
        READ(UNIT=kgen_unit) ka_mn2o
        READ(UNIT=kgen_unit) kb_mn2o
        READ(UNIT=kgen_unit) selfref
        READ(UNIT=kgen_unit) forref
    END SUBROUTINE kgen_read_externs_rrlw_kg09

    END MODULE rrlw_kg09
