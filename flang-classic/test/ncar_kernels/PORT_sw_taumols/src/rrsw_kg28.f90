
! KGEN-generated Fortran source file
!
! Filename    : rrsw_kg28.f90
! Generated at: 2015-07-31 20:45:43
! KGEN version: 0.4.13



    MODULE rrsw_kg28
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !      use parkind ,only : jpim, jprb
        USE parrrsw, ONLY: ng28
        IMPLICIT NONE
        !-----------------------------------------------------------------
        ! rrtmg_sw ORIGINAL abs. coefficients for interval 28
        ! band 28: 38000-50000 cm-1 (low - o3, o2; high - o3, o2)
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
        !-----------------------------------------------------------------
        INTEGER :: layreffr
        REAL(KIND=r8) :: strrat
        REAL(KIND=r8) :: rayl
        !-----------------------------------------------------------------
        ! rrtmg_sw COMBINED abs. coefficients for interval 28
        ! band 28: 38000-50000 cm-1 (low - o3, o2; high - o3, o2)
        !
        ! Initial version:  JJMorcrette, ECMWF, oct1999
        ! Revised: MJIacono, AER, jul2006
        !-----------------------------------------------------------------
        !
        !  name     type     purpose
        !  ----   : ----   : ---------------------------------------------
        ! ka      : real
        ! kb      : real
        ! sfluxref: real
        !-----------------------------------------------------------------
        REAL(KIND=r8) :: absa(585,ng28)
        REAL(KIND=r8) :: absb(1175,ng28)
        REAL(KIND=r8) :: sfluxref(ng28,5)
        PUBLIC kgen_read_externs_rrsw_kg28
    CONTAINS

    ! write subroutines

    ! module extern variables

    SUBROUTINE kgen_read_externs_rrsw_kg28(kgen_unit)
        INTEGER, INTENT(IN) :: kgen_unit
        READ(UNIT=kgen_unit) layreffr
        READ(UNIT=kgen_unit) strrat
        READ(UNIT=kgen_unit) rayl
        READ(UNIT=kgen_unit) absa
        READ(UNIT=kgen_unit) absb
        READ(UNIT=kgen_unit) sfluxref
    END SUBROUTINE kgen_read_externs_rrsw_kg28

    END MODULE rrsw_kg28
