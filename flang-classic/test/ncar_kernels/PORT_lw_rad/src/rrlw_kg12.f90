
! KGEN-generated Fortran source file
!
! Filename    : rrlw_kg12.f90
! Generated at: 2015-07-06 23:28:45
! KGEN version: 0.4.13



    MODULE rrlw_kg12
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !      use parkind ,only : jpim, jprb
        IMPLICIT NONE
        !-----------------------------------------------------------------
        ! rrtmg_lw ORIGINAL abs. coefficients for interval 12
        ! band 12:  1800-2080 cm-1 (low - h2o,co2; high - nothing)
        !
        ! Initial version:  JJMorcrette, ECMWF, jul1998
        ! Revised: MJIacono, AER, jun2006
        !-----------------------------------------------------------------
        !
        !  name     type     purpose
        !  ----   : ----   : ---------------------------------------------
        !fracrefao: real
        ! kao     : real
        ! selfrefo: real
        ! forrefo : real
        !-----------------------------------------------------------------
        !-----------------------------------------------------------------
        ! rrtmg_lw COMBINED abs. coefficients for interval 12
        ! band 12:  1800-2080 cm-1 (low - h2o,co2; high - nothing)
        !
        ! Initial version:  JJMorcrette, ECMWF, jul1998
        ! Revised: MJIacono, AER, jun2006
        !-----------------------------------------------------------------
        !
        !  name     type     purpose
        !  ----   : ----   : ---------------------------------------------
        !fracrefa : real
        ! ka      : real
        ! selfref : real
        ! forref  : real
        !
        ! absa    : real
        !-----------------------------------------------------------------
        INTEGER, parameter :: ng12 = 8
        REAL(KIND=r8) :: fracrefa(ng12,9)
        REAL(KIND=r8) :: absa(585,ng12)
        REAL(KIND=r8) :: selfref(10,ng12)
        REAL(KIND=r8) :: forref(4,ng12)
        PUBLIC kgen_read_externs_rrlw_kg12
    CONTAINS

    ! write subroutines

    ! module extern variables

    SUBROUTINE kgen_read_externs_rrlw_kg12(kgen_unit)
        INTEGER, INTENT(IN) :: kgen_unit
        READ(UNIT=kgen_unit) fracrefa
        READ(UNIT=kgen_unit) absa
        READ(UNIT=kgen_unit) selfref
        READ(UNIT=kgen_unit) forref
    END SUBROUTINE kgen_read_externs_rrlw_kg12

    END MODULE rrlw_kg12
