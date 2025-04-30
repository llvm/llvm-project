
! KGEN-generated Fortran source file
!
! Filename    : rrlw_kg16.f90
! Generated at: 2015-07-06 23:28:43
! KGEN version: 0.4.13



    MODULE rrlw_kg16
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !      use parkind ,only : jpim, jprb
        IMPLICIT NONE
        !-----------------------------------------------------------------
        ! rrtmg_lw ORIGINAL abs. coefficients for interval 16
        ! band 16:  2600-3000 cm-1 (low - h2o,ch4; high - nothing)
        !
        ! Initial version:  JJMorcrette, ECMWF, jul1998
        ! Revised: MJIacono, AER, jun2006
        !-----------------------------------------------------------------
        !
        !  name     type     purpose
        !  ----   : ----   : ---------------------------------------------
        !fracrefao: real
        ! kao     : real
        ! kbo     : real
        ! selfrefo: real
        ! forrefo : real
        !-----------------------------------------------------------------
        !-----------------------------------------------------------------
        ! rrtmg_lw COMBINED abs. coefficients for interval 16
        ! band 16:  2600-3000 cm-1 (low - h2o,ch4; high - nothing)
        !
        ! Initial version:  JJMorcrette, ECMWF, jul1998
        ! Revised: MJIacono, AER, jun2006
        !-----------------------------------------------------------------
        !
        !  name     type     purpose
        !  ----   : ----   : ---------------------------------------------
        !fracrefa : real
        ! ka      : real
        ! kb      : real
        ! selfref : real
        ! forref  : real
        !
        ! absa    : real
        ! absb    : real
        !-----------------------------------------------------------------
        INTEGER, parameter :: ng16 = 2
        REAL(KIND=r8), dimension(ng16) :: fracrefb
        REAL(KIND=r8) :: fracrefa(ng16,9)
        REAL(KIND=r8) :: absa(585,ng16)
        REAL(KIND=r8) :: absb(235,ng16)
        REAL(KIND=r8) :: selfref(10,ng16)
        REAL(KIND=r8) :: forref(4,ng16)
        PUBLIC kgen_read_externs_rrlw_kg16
    CONTAINS

    ! write subroutines

    ! module extern variables

    SUBROUTINE kgen_read_externs_rrlw_kg16(kgen_unit)
        INTEGER, INTENT(IN) :: kgen_unit
        READ(UNIT=kgen_unit) fracrefb
        READ(UNIT=kgen_unit) fracrefa
        READ(UNIT=kgen_unit) absa
        READ(UNIT=kgen_unit) absb
        READ(UNIT=kgen_unit) selfref
        READ(UNIT=kgen_unit) forref
    END SUBROUTINE kgen_read_externs_rrlw_kg16

    END MODULE rrlw_kg16
