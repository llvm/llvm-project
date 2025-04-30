
! KGEN-generated Fortran source file
!
! Filename    : rrlw_kg07.f90
! Generated at: 2015-07-06 23:28:45
! KGEN version: 0.4.13



    MODULE rrlw_kg07
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !      use parkind ,only : jpim, jprb
        IMPLICIT NONE
        !-----------------------------------------------------------------
        ! rrtmg_lw ORIGINAL abs. coefficients for interval 7
        ! band 7:  980-1080 cm-1 (low - h2o,o3; high - o3)
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
        ! selfrefo: real
        ! forrefo : real
        !-----------------------------------------------------------------
        !-----------------------------------------------------------------
        ! rrtmg_lw COMBINED abs. coefficients for interval 7
        ! band 7:  980-1080 cm-1 (low - h2o,o3; high - o3)
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
        ! selfref : real
        ! forref  : real
        !
        ! absa    : real
        !-----------------------------------------------------------------
        INTEGER, parameter :: ng7  = 12
        REAL(KIND=r8), dimension(ng7) :: fracrefb
        REAL(KIND=r8) :: fracrefa(ng7,9)
        REAL(KIND=r8) :: absa(585,ng7)
        REAL(KIND=r8) :: absb(235,ng7)
        REAL(KIND=r8) :: ka_mco2(9,19,ng7)
        REAL(KIND=r8) :: kb_mco2(19,ng7)
        REAL(KIND=r8) :: selfref(10,ng7)
        REAL(KIND=r8) :: forref(4,ng7)
        PUBLIC kgen_read_externs_rrlw_kg07
    CONTAINS

    ! write subroutines

    ! module extern variables

    SUBROUTINE kgen_read_externs_rrlw_kg07(kgen_unit)
        INTEGER, INTENT(IN) :: kgen_unit
        READ(UNIT=kgen_unit) fracrefb
        READ(UNIT=kgen_unit) fracrefa
        READ(UNIT=kgen_unit) absa
        READ(UNIT=kgen_unit) absb
        READ(UNIT=kgen_unit) ka_mco2
        READ(UNIT=kgen_unit) kb_mco2
        READ(UNIT=kgen_unit) selfref
        READ(UNIT=kgen_unit) forref
    END SUBROUTINE kgen_read_externs_rrlw_kg07

    END MODULE rrlw_kg07
