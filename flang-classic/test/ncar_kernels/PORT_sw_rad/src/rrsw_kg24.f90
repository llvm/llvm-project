
! KGEN-generated Fortran source file
!
! Filename    : rrsw_kg24.f90
! Generated at: 2015-07-07 00:48:24
! KGEN version: 0.4.13



    MODULE rrsw_kg24
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !      use parkind ,only : jpim, jprb
        USE parrrsw, ONLY: ng24
        IMPLICIT NONE
        !-----------------------------------------------------------------
        ! rrtmg_sw ORIGINAL abs. coefficients for interval 24
        ! band 24: 12850-16000 cm-1 (low - h2o,o2; high - o2)
        !
        ! Initial version:  JJMorcrette, ECMWF, oct1999
        ! Revised: MJIacono, AER, jul2006
        !-----------------------------------------------------------------
        !
        !  name     type     purpose
        !  ----   : ----   : ---------------------------------------------
        ! kao     : real
        ! kbo     : real
        ! selfrefo: real
        ! forrefo : real
        !sfluxrefo: real
        ! abso3ao : real
        ! abso3bo : real
        ! raylao  : real
        ! raylbo  : real
        !-----------------------------------------------------------------
        INTEGER :: layreffr
        REAL(KIND=r8) :: strrat
        !-----------------------------------------------------------------
        ! rrtmg_sw COMBINED abs. coefficients for interval 24
        ! band 24: 12850-16000 cm-1 (low - h2o,o2; high - o2)
        !
        ! Initial version:  JJMorcrette, ECMWF, oct1999
        ! Revised: MJIacono, AER, jul2006
        !-----------------------------------------------------------------
        !
        !  name     type     purpose
        !  ----   : ----   : ---------------------------------------------
        ! ka      : real
        ! kb      : real
        ! absa    : real
        ! absb    : real
        ! selfref : real
        ! forref  : real
        ! sfluxref: real
        ! abso3a  : real
        ! abso3b  : real
        ! rayla   : real
        ! raylb   : real
        !-----------------------------------------------------------------
        REAL(KIND=r8) :: absa(585,ng24)
        REAL(KIND=r8) :: absb(235,ng24)
        REAL(KIND=r8) :: forref(3,ng24)
        REAL(KIND=r8) :: selfref(10,ng24)
        REAL(KIND=r8) :: sfluxref(ng24,9)
        REAL(KIND=r8) :: abso3a(ng24)
        REAL(KIND=r8) :: abso3b(ng24)
        REAL(KIND=r8) :: rayla(ng24,9)
        REAL(KIND=r8) :: raylb(ng24)
        PUBLIC kgen_read_externs_rrsw_kg24
    CONTAINS

    ! write subroutines

    ! module extern variables

    SUBROUTINE kgen_read_externs_rrsw_kg24(kgen_unit)
        INTEGER, INTENT(IN) :: kgen_unit
        READ(UNIT=kgen_unit) layreffr
        READ(UNIT=kgen_unit) strrat
        READ(UNIT=kgen_unit) absa
        READ(UNIT=kgen_unit) absb
        READ(UNIT=kgen_unit) forref
        READ(UNIT=kgen_unit) selfref
        READ(UNIT=kgen_unit) sfluxref
        READ(UNIT=kgen_unit) abso3a
        READ(UNIT=kgen_unit) abso3b
        READ(UNIT=kgen_unit) rayla
        READ(UNIT=kgen_unit) raylb
    END SUBROUTINE kgen_read_externs_rrsw_kg24

    END MODULE rrsw_kg24
