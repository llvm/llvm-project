
! KGEN-generated Fortran source file
!
! Filename    : rrlw_tbl.f90
! Generated at: 2015-07-26 20:37:04
! KGEN version: 0.4.13



    MODULE rrlw_tbl
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !      use parkind, only : jpim, jprb
        IMPLICIT NONE
        !------------------------------------------------------------------
        ! rrtmg_lw exponential lookup table arrays
        ! Initial version:  JJMorcrette, ECMWF, jul1998
        ! Revised: MJIacono, AER, Jun 2006
        ! Revised: MJIacono, AER, Aug 2007
        !------------------------------------------------------------------
        !  name     type     purpose
        ! -----  :  ----   : ----------------------------------------------
        ! ntbl   :  integer: Lookup table dimension
        ! tblint :  real   : Lookup table conversion factor
        ! tau_tbl:  real   : Clear-sky optical depth (used in cloudy radiative
        !                    transfer)
        ! exp_tbl:  real   : Transmittance lookup table
        ! tfn_tbl:  real   : Tau transition function; i.e. the transition of
        !                    the Planck function from that for the mean layer
        !                    temperature to that for the layer boundary
        !                    temperature as a function of optical depth.
        !                    The "linear in tau" method is used to make
        !                    the table.
        ! pade   :  real   : Pade constant
        ! bpade  :  real   : Inverse of Pade constant
        !------------------------------------------------------------------
        INTEGER, parameter :: ntbl = 10000
        REAL(KIND=r8), parameter :: tblint = 10000.0_r8
        REAL(KIND=r8), dimension(0:ntbl) :: tau_tbl
        REAL(KIND=r8), dimension(0:ntbl) :: exp_tbl
        REAL(KIND=r8), dimension(0:ntbl) :: tfn_tbl
        REAL(KIND=r8) :: bpade
        PUBLIC kgen_read_externs_rrlw_tbl
    CONTAINS

    ! write subroutines

    ! module extern variables

    SUBROUTINE kgen_read_externs_rrlw_tbl(kgen_unit)
        INTEGER, INTENT(IN) :: kgen_unit
        READ(UNIT=kgen_unit) tau_tbl
        READ(UNIT=kgen_unit) exp_tbl
        READ(UNIT=kgen_unit) tfn_tbl
        READ(UNIT=kgen_unit) bpade
    END SUBROUTINE kgen_read_externs_rrlw_tbl

    END MODULE rrlw_tbl
