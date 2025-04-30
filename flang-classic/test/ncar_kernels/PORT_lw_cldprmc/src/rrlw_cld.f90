
! KGEN-generated Fortran source file
!
! Filename    : rrlw_cld.f90
! Generated at: 2015-07-26 20:16:59
! KGEN version: 0.4.13



    MODULE rrlw_cld
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !      use parkind, only : jpim, jprb
        IMPLICIT NONE
        !------------------------------------------------------------------
        ! rrtmg_lw cloud property coefficients
        ! Revised: MJIacono, AER, jun2006
        !------------------------------------------------------------------
        !  name     type     purpose
        ! -----  :  ----   : ----------------------------------------------
        ! abscld1:  real   :
        ! absice0:  real   :
        ! absice1:  real   :
        ! absice2:  real   :
        ! absice3:  real   :
        ! absliq0:  real   :
        ! absliq1:  real   :
        !------------------------------------------------------------------
        REAL(KIND=r8), dimension(2) :: absice0
        REAL(KIND=r8), dimension(2,5) :: absice1
        REAL(KIND=r8), dimension(43,16) :: absice2
        REAL(KIND=r8), dimension(46,16) :: absice3
        REAL(KIND=r8) :: absliq0
        REAL(KIND=r8), dimension(58,16) :: absliq1
        PUBLIC kgen_read_externs_rrlw_cld
    CONTAINS

    ! write subroutines

    ! module extern variables

    SUBROUTINE kgen_read_externs_rrlw_cld(kgen_unit)
        INTEGER, INTENT(IN) :: kgen_unit
        READ(UNIT=kgen_unit) absice0
        READ(UNIT=kgen_unit) absice1
        READ(UNIT=kgen_unit) absice2
        READ(UNIT=kgen_unit) absice3
        READ(UNIT=kgen_unit) absliq0
        READ(UNIT=kgen_unit) absliq1
    END SUBROUTINE kgen_read_externs_rrlw_cld

    END MODULE rrlw_cld
