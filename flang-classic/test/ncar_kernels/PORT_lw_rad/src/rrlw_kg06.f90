
! KGEN-generated Fortran source file
!
! Filename    : rrlw_kg06.f90
! Generated at: 2015-07-06 23:28:45
! KGEN version: 0.4.13



    MODULE rrlw_kg06
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !      use parkind ,only : jpim, jprb
        IMPLICIT NONE
        !-----------------------------------------------------------------
        ! rrtmg_lw ORIGINAL abs. coefficients for interval 6
        ! band 6:  820-980 cm-1 (low - h2o; high - nothing)
        !
        ! Initial version:  JJMorcrette, ECMWF, jul1998
        ! Revised: MJIacono, AER, jun2006
        !-----------------------------------------------------------------
        !
        !  name     type     purpose
        !  ----   : ----   : ---------------------------------------------
        !fracrefao: real
        ! kao     : real
        ! kao_mco2: real
        ! selfrefo: real
        ! forrefo : real
        !cfc11adjo: real
        ! cfc12o  : real
        !-----------------------------------------------------------------
        !-----------------------------------------------------------------
        ! rrtmg_lw COMBINED abs. coefficients for interval 6
        ! band 6:  820-980 cm-1 (low - h2o; high - nothing)
        !
        ! Initial version:  JJMorcrette, ECMWF, jul1998
        ! Revised: MJIacono, AER, jun2006
        !-----------------------------------------------------------------
        !
        !  name     type     purpose
        !  ----   : ----   : ---------------------------------------------
        !fracrefa : real
        ! ka      : real
        ! ka_mco2 : real
        ! selfref : real
        ! forref  : real
        !cfc11adj : real
        ! cfc12   : real
        !
        ! absa    : real
        !-----------------------------------------------------------------
        INTEGER, parameter :: ng6  = 8
        REAL(KIND=r8), dimension(ng6) :: fracrefa
        REAL(KIND=r8) :: absa(65,ng6)
        REAL(KIND=r8) :: ka_mco2(19,ng6)
        REAL(KIND=r8) :: selfref(10,ng6)
        REAL(KIND=r8) :: forref(4,ng6)
        REAL(KIND=r8), dimension(ng6) :: cfc11adj
        REAL(KIND=r8), dimension(ng6) :: cfc12
        PUBLIC kgen_read_externs_rrlw_kg06
    CONTAINS

    ! write subroutines

    ! module extern variables

    SUBROUTINE kgen_read_externs_rrlw_kg06(kgen_unit)
        INTEGER, INTENT(IN) :: kgen_unit
        READ(UNIT=kgen_unit) fracrefa
        READ(UNIT=kgen_unit) absa
        READ(UNIT=kgen_unit) ka_mco2
        READ(UNIT=kgen_unit) selfref
        READ(UNIT=kgen_unit) forref
        READ(UNIT=kgen_unit) cfc11adj
        READ(UNIT=kgen_unit) cfc12
    END SUBROUTINE kgen_read_externs_rrlw_kg06

    END MODULE rrlw_kg06
