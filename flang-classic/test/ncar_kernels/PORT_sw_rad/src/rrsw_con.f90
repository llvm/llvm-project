
! KGEN-generated Fortran source file
!
! Filename    : rrsw_con.f90
! Generated at: 2015-07-07 00:48:24
! KGEN version: 0.4.13



    MODULE rrsw_con
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !      use parkind, only : jpim, jprb
        IMPLICIT NONE
        !------------------------------------------------------------------
        ! rrtmg_sw constants
        ! Initial version: MJIacono, AER, jun2006
        !------------------------------------------------------------------
        !  name     type     purpose
        ! -----  :  ----   : ----------------------------------------------
        ! fluxfac:  real   : radiance to flux conversion factor
        ! heatfac:  real   : flux to heating rate conversion factor
        !oneminus:  real   : 1.-1.e-6
        ! pi     :  real   : pi
        ! grav   :  real   : acceleration of gravity (m/s2)
        ! planck :  real   : planck constant
        ! boltz  :  real   : boltzman constant
        ! clight :  real   : speed of light
        ! avogad :  real   : avogadro's constant
        ! alosmt :  real   :
        ! gascon :  real   : gas constant
        ! radcn1 :  real   :
        ! radcn2 :  real   :
        !------------------------------------------------------------------
        REAL(KIND=r8) :: heatfac
        REAL(KIND=r8) :: oneminus
        REAL(KIND=r8) :: pi
        REAL(KIND=r8) :: grav
        REAL(KIND=r8) :: avogad
        PUBLIC kgen_read_externs_rrsw_con
    CONTAINS

    ! write subroutines
    ! No subroutines

    ! module extern variables

    SUBROUTINE kgen_read_externs_rrsw_con(kgen_unit)
        INTEGER, INTENT(IN) :: kgen_unit
        READ(UNIT=kgen_unit) heatfac
        READ(UNIT=kgen_unit) oneminus
        READ(UNIT=kgen_unit) pi
        READ(UNIT=kgen_unit) grav
        READ(UNIT=kgen_unit) avogad
    END SUBROUTINE kgen_read_externs_rrsw_con

    END MODULE rrsw_con
