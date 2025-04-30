
! KGEN-generated Fortran source file
!
! Filename    : rrlw_kg13.f90
! Generated at: 2015-07-06 23:28:45
! KGEN version: 0.4.13



    MODULE rrlw_kg13
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !      use parkind ,only : jpim, jprb
        IMPLICIT NONE
        !-----------------------------------------------------------------
        ! rrtmg_lw ORIGINAL abs. coefficients for interval 13
        ! band 13:  2080-2250 cm-1 (low - h2o,n2o; high - nothing)
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
        ! kao_mco : real
        ! kbo_mo3 : real
        ! selfrefo: real
        ! forrefo : real
        !-----------------------------------------------------------------
        !-----------------------------------------------------------------
        ! rrtmg_lw COMBINED abs. coefficients for interval 13
        ! band 13:  2080-2250 cm-1 (low - h2o,n2o; high - nothing)
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
        ! ka_mco  : real
        ! kb_mo3  : real
        ! selfref : real
        ! forref  : real
        !
        ! absa    : real
        !-----------------------------------------------------------------
        INTEGER, parameter :: ng13 = 4
        REAL(KIND=r8), dimension(ng13) :: fracrefb
        REAL(KIND=r8) :: fracrefa(ng13,9)
        REAL(KIND=r8) :: absa(585,ng13)
        REAL(KIND=r8) :: ka_mco2(9,19,ng13)
        REAL(KIND=r8) :: ka_mco(9,19,ng13)
        REAL(KIND=r8) :: kb_mo3(19,ng13)
        REAL(KIND=r8) :: selfref(10,ng13)
        REAL(KIND=r8) :: forref(4,ng13)
        PUBLIC kgen_read_externs_rrlw_kg13
    CONTAINS

    ! write subroutines

    ! module extern variables

    SUBROUTINE kgen_read_externs_rrlw_kg13(kgen_unit)
        INTEGER, INTENT(IN) :: kgen_unit
        READ(UNIT=kgen_unit) fracrefb
        READ(UNIT=kgen_unit) fracrefa
        READ(UNIT=kgen_unit) absa
        READ(UNIT=kgen_unit) ka_mco2
        READ(UNIT=kgen_unit) ka_mco
        READ(UNIT=kgen_unit) kb_mo3
        READ(UNIT=kgen_unit) selfref
        READ(UNIT=kgen_unit) forref
    END SUBROUTINE kgen_read_externs_rrlw_kg13

    END MODULE rrlw_kg13
