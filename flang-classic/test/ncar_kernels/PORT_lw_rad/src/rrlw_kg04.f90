
! KGEN-generated Fortran source file
!
! Filename    : rrlw_kg04.f90
! Generated at: 2015-07-06 23:28:45
! KGEN version: 0.4.13



    MODULE rrlw_kg04
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !      use parkind ,only : jpim, jprb
        IMPLICIT NONE
        !-----------------------------------------------------------------
        ! rrtmg_lw ORIGINAL abs. coefficients for interval 4
        ! band 4:  630-700 cm-1 (low - h2o,co2; high - o3,co2)
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
        ! selfrefo: real
        ! forrefo : real
        !-----------------------------------------------------------------
        !-----------------------------------------------------------------
        ! rrtmg_lw COMBINED abs. coefficients for interval 4
        ! band 4:  630-700 cm-1 (low - h2o,co2; high - o3,co2)
        !
        ! Initial version:  JJMorcrette, ECMWF, jul1998
        ! Revised: MJIacono, AER, jun2006
        !-----------------------------------------------------------------
        !
        !  name     type     purpose
        !  ----   : ----   : ---------------------------------------------
        ! absa    : real
        ! absb    : real
        !fracrefa : real
        !fracrefb : real
        ! ka      : real
        ! kb      : real
        ! selfref : real
        ! forref  : real
        !-----------------------------------------------------------------
        INTEGER, parameter :: ng4  = 14
        REAL(KIND=r8) :: fracrefa(ng4,9)
        REAL(KIND=r8) :: fracrefb(ng4,6)
        REAL(KIND=r8) :: absa(585,ng4)
        REAL(KIND=r8) :: absb(1175,ng4)
        REAL(KIND=r8) :: selfref(10,ng4)
        REAL(KIND=r8) :: forref(4,ng4)
        PUBLIC kgen_read_externs_rrlw_kg04
    CONTAINS

    ! write subroutines

    ! module extern variables

    SUBROUTINE kgen_read_externs_rrlw_kg04(kgen_unit)
        INTEGER, INTENT(IN) :: kgen_unit
        READ(UNIT=kgen_unit) fracrefa
        READ(UNIT=kgen_unit) fracrefb
        READ(UNIT=kgen_unit) absa
        READ(UNIT=kgen_unit) absb
        READ(UNIT=kgen_unit) selfref
        READ(UNIT=kgen_unit) forref
    END SUBROUTINE kgen_read_externs_rrlw_kg04

    END MODULE rrlw_kg04
