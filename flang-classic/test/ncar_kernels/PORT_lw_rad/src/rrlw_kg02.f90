
! KGEN-generated Fortran source file
!
! Filename    : rrlw_kg02.f90
! Generated at: 2015-07-06 23:28:44
! KGEN version: 0.4.13



    MODULE rrlw_kg02
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !      use parkind ,only : jpim, jprb
        IMPLICIT NONE
        !-----------------------------------------------------------------
        ! rrtmg_lw ORIGINAL abs. coefficients for interval 2
        ! band 2:  250-500 cm-1 (low - h2o; high - h2o)
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
        ! rrtmg_lw COMBINED abs. coefficients for interval 2
        ! band 2:  250-500 cm-1 (low - h2o; high - h2o)
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
        ! absa    : real
        ! absb    : real
        ! selfref : real
        ! forref  : real
        !
        ! refparam: real
        !-----------------------------------------------------------------
        INTEGER, parameter :: ng2  = 12
        REAL(KIND=r8) :: fracrefa(ng2)
        REAL(KIND=r8) :: fracrefb(ng2)
        REAL(KIND=r8) :: absa(65,ng2)
        REAL(KIND=r8) :: absb(235,ng2)
        REAL(KIND=r8) :: selfref(10,ng2)
        REAL(KIND=r8) :: forref(4,ng2)
        PUBLIC kgen_read_externs_rrlw_kg02
    CONTAINS

    ! write subroutines

    ! module extern variables

    SUBROUTINE kgen_read_externs_rrlw_kg02(kgen_unit)
        INTEGER, INTENT(IN) :: kgen_unit
        READ(UNIT=kgen_unit) fracrefa
        READ(UNIT=kgen_unit) fracrefb
        READ(UNIT=kgen_unit) absa
        READ(UNIT=kgen_unit) absb
        READ(UNIT=kgen_unit) selfref
        READ(UNIT=kgen_unit) forref
    END SUBROUTINE kgen_read_externs_rrlw_kg02

    END MODULE rrlw_kg02
