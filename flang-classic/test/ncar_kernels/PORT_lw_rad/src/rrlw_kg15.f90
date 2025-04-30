
! KGEN-generated Fortran source file
!
! Filename    : rrlw_kg15.f90
! Generated at: 2015-07-06 23:28:43
! KGEN version: 0.4.13



    MODULE rrlw_kg15
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !      use parkind ,only : jpim, r8
        IMPLICIT NONE
        !-----------------------------------------------------------------
        ! rrtmg_lw ORIGINAL abs. coefficients for interval 15
        ! band 15:  2380-2600 cm-1 (low - n2o,co2; high - nothing)
        !
        ! Initial version:  JJMorcrette, ECMWF, jul1998
        ! Revised: MJIacono, AER, jun2006
        !-----------------------------------------------------------------
        !
        !  name     type     purpose
        !  ----   : ----   : ---------------------------------------------
        !fracrefao: real
        ! kao     : real
        ! kao_mn2 : real
        ! selfrefo: real
        ! forrefo : real
        !-----------------------------------------------------------------
        !-----------------------------------------------------------------
        ! rrtmg_lw COMBINED abs. coefficients for interval 15
        ! band 15:  2380-2600 cm-1 (low - n2o,co2; high - nothing)
        !
        ! Initial version:  JJMorcrette, ECMWF, jul1998
        ! Revised: MJIacono, AER, jun2006
        !-----------------------------------------------------------------
        !
        !  name     type     purpose
        !  ----   : ----   : ---------------------------------------------
        !fracrefa : real
        ! ka      : real
        ! ka_mn2  : real
        ! selfref : real
        ! forref  : real
        !
        ! absa    : real
        !-----------------------------------------------------------------
        INTEGER, parameter :: ng15 = 2
        REAL(KIND=r8) :: fracrefa(ng15,9)
        REAL(KIND=r8) :: absa(585,ng15)
        REAL(KIND=r8) :: ka_mn2(9,19,ng15)
        REAL(KIND=r8) :: selfref(10,ng15)
        REAL(KIND=r8) :: forref(4,ng15)
        PUBLIC kgen_read_externs_rrlw_kg15
    CONTAINS

    ! write subroutines

    ! module extern variables

    SUBROUTINE kgen_read_externs_rrlw_kg15(kgen_unit)
        INTEGER, INTENT(IN) :: kgen_unit
        READ(UNIT=kgen_unit) fracrefa
        READ(UNIT=kgen_unit) absa
        READ(UNIT=kgen_unit) ka_mn2
        READ(UNIT=kgen_unit) selfref
        READ(UNIT=kgen_unit) forref
    END SUBROUTINE kgen_read_externs_rrlw_kg15

    END MODULE rrlw_kg15
