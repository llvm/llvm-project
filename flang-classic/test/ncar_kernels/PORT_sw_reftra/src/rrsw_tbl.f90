
! KGEN-generated Fortran source file
!
! Filename    : rrsw_tbl.f90
! Generated at: 2015-07-31 20:52:25
! KGEN version: 0.4.13



    MODULE rrsw_tbl
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !      use parkind, only : jpim, jprb
        IMPLICIT NONE
        !------------------------------------------------------------------
        ! rrtmg_sw lookup table arrays
        ! Initial version: MJIacono, AER, may2007
        ! Revised: MJIacono, AER, aug2007
        !------------------------------------------------------------------
        !  name     type     purpose
        ! -----  :  ----   : ----------------------------------------------
        ! ntbl   :  integer: Lookup table dimension
        ! tblint :  real   : Lookup table conversion factor
        ! tau_tbl:  real   : Clear-sky optical depth
        ! exp_tbl:  real   : Exponential lookup table for transmittance
        ! od_lo  :  real   : Value of tau below which expansion is used
        !                  : in place of lookup table
        ! pade   :  real   : Pade approximation constant
        ! bpade  :  real   : Inverse of Pade constant
        !------------------------------------------------------------------
        INTEGER, parameter :: ntbl = 10000
        REAL(KIND=r8), parameter :: tblint = 10000.0
        REAL(KIND=r8), parameter :: od_lo = 0.06
        REAL(KIND=r8), dimension(0:ntbl) :: exp_tbl
        REAL(KIND=r8) :: bpade
        PUBLIC kgen_read_externs_rrsw_tbl
    CONTAINS

    ! write subroutines

    ! module extern variables

    SUBROUTINE kgen_read_externs_rrsw_tbl(kgen_unit)
        INTEGER, INTENT(IN) :: kgen_unit
        READ(UNIT=kgen_unit) exp_tbl
        READ(UNIT=kgen_unit) bpade
    END SUBROUTINE kgen_read_externs_rrsw_tbl

    END MODULE rrsw_tbl
