
! KGEN-generated Fortran source file
!
! Filename    : rrsw_kg27.f90
! Generated at: 2015-07-07 00:48:24
! KGEN version: 0.4.13



    MODULE rrsw_kg27
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !      use parkind ,only : jpim, jprb
        USE parrrsw, ONLY: ng27
        IMPLICIT NONE
        !-----------------------------------------------------------------
        ! rrtmg_sw ORIGINAL abs. coefficients for interval 27
        ! band 27: 29000-38000 cm-1 (low - o3; high - o3)
        !
        ! Initial version:  JJMorcrette, ECMWF, oct1999
        ! Revised: MJIacono, AER, jul2006
        !-----------------------------------------------------------------
        !
        !  name     type     purpose
        !  ----   : ----   : ---------------------------------------------
        ! kao     : real
        ! kbo     : real
        !sfluxrefo: real
        ! raylo   : real
        !-----------------------------------------------------------------
        INTEGER :: layreffr
        REAL(KIND=r8) :: scalekur
        !-----------------------------------------------------------------
        ! rrtmg_sw COMBINED abs. coefficients for interval 27
        ! band 27: 29000-38000 cm-1 (low - o3; high - o3)
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
        ! sfluxref: real
        ! rayl    : real
        !-----------------------------------------------------------------
        REAL(KIND=r8) :: absa(65,ng27)
        REAL(KIND=r8) :: absb(235,ng27)
        REAL(KIND=r8) :: sfluxref(ng27)
        REAL(KIND=r8) :: rayl(ng27)
        PUBLIC kgen_read_externs_rrsw_kg27
    CONTAINS

    ! write subroutines

    ! module extern variables

    SUBROUTINE kgen_read_externs_rrsw_kg27(kgen_unit)
        INTEGER, INTENT(IN) :: kgen_unit
        READ(UNIT=kgen_unit) layreffr
        READ(UNIT=kgen_unit) scalekur
        READ(UNIT=kgen_unit) absa
        READ(UNIT=kgen_unit) absb
        READ(UNIT=kgen_unit) sfluxref
        READ(UNIT=kgen_unit) rayl
    END SUBROUTINE kgen_read_externs_rrsw_kg27

    END MODULE rrsw_kg27
