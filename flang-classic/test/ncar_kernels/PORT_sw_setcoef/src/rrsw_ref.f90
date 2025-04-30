
! KGEN-generated Fortran source file
!
! Filename    : rrsw_ref.f90
! Generated at: 2015-07-27 00:47:03
! KGEN version: 0.4.13



    MODULE rrsw_ref
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !      use parkind, only : jpim, jprb
        IMPLICIT NONE
        !------------------------------------------------------------------
        ! rrtmg_sw reference atmosphere
        ! Based on standard mid-latitude summer profile
        !
        ! Initial version:  JJMorcrette, ECMWF, jul1998
        ! Revised: MJIacono, AER, jun2006
        !------------------------------------------------------------------
        !  name     type     purpose
        ! -----  :  ----   : ----------------------------------------------
        ! pref   :  real   : Reference pressure levels
        ! preflog:  real   : Reference pressure levels, ln(pref)
        ! tref   :  real   : Reference temperature levels for MLS profile
        !------------------------------------------------------------------
        REAL(KIND=r8), dimension(59) :: preflog
        REAL(KIND=r8), dimension(59) :: tref
        PUBLIC kgen_read_externs_rrsw_ref
    CONTAINS

    ! write subroutines

    ! module extern variables

    SUBROUTINE kgen_read_externs_rrsw_ref(kgen_unit)
        INTEGER, INTENT(IN) :: kgen_unit
        READ(UNIT=kgen_unit) preflog
        READ(UNIT=kgen_unit) tref
    END SUBROUTINE kgen_read_externs_rrsw_ref

    END MODULE rrsw_ref
