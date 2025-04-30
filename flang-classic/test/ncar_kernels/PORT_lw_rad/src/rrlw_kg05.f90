
! KGEN-generated Fortran source file
!
! Filename    : rrlw_kg05.f90
! Generated at: 2015-07-06 23:28:45
! KGEN version: 0.4.13



    MODULE rrlw_kg05
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !      use parkind ,only : jpim, jprb
        IMPLICIT NONE
        !-----------------------------------------------------------------
        ! rrtmg_lw ORIGINAL abs. coefficients for interval 5
        ! band 5:  700-820 cm-1 (low - h2o,co2; high - o3,co2)
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
        ! kao_mo3 : real
        ! selfrefo: real
        ! forrefo : real
        ! ccl4o   : real
        !-----------------------------------------------------------------
        !-----------------------------------------------------------------
        ! rrtmg_lw COMBINED abs. coefficients for interval 5
        ! band 5:  700-820 cm-1 (low - h2o,co2; high - o3,co2)
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
        ! ka_mo3  : real
        ! selfref : real
        ! forref  : real
        ! ccl4    : real
        !
        ! absa    : real
        ! absb    : real
        !-----------------------------------------------------------------
        INTEGER, parameter :: ng5  = 16
        REAL(KIND=r8) :: fracrefa(ng5,9)
        REAL(KIND=r8) :: fracrefb(ng5,5)
        REAL(KIND=r8) :: absa(585,ng5)
        REAL(KIND=r8) :: absb(1175,ng5)
        REAL(KIND=r8) :: ka_mo3(9,19,ng5)
        REAL(KIND=r8) :: selfref(10,ng5)
        REAL(KIND=r8) :: forref(4,ng5)
        REAL(KIND=r8) :: ccl4(ng5)
        PUBLIC kgen_read_externs_rrlw_kg05
    CONTAINS

    ! write subroutines

    ! module extern variables

    SUBROUTINE kgen_read_externs_rrlw_kg05(kgen_unit)
        INTEGER, INTENT(IN) :: kgen_unit
        READ(UNIT=kgen_unit) fracrefa
        READ(UNIT=kgen_unit) fracrefb
        READ(UNIT=kgen_unit) absa
        READ(UNIT=kgen_unit) absb
        READ(UNIT=kgen_unit) ka_mo3
        READ(UNIT=kgen_unit) selfref
        READ(UNIT=kgen_unit) forref
        READ(UNIT=kgen_unit) ccl4
    END SUBROUTINE kgen_read_externs_rrlw_kg05

    END MODULE rrlw_kg05
