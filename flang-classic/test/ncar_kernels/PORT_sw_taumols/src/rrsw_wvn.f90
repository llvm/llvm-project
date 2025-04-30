
! KGEN-generated Fortran source file
!
! Filename    : rrsw_wvn.f90
! Generated at: 2015-07-31 20:45:42
! KGEN version: 0.4.13



    MODULE rrsw_wvn
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        !      use parkind, only : jpim, jprb
        USE parrrsw, ONLY: jpb1
        USE parrrsw, ONLY: jpb2
        IMPLICIT NONE
        !------------------------------------------------------------------
        ! rrtmg_sw spectral information
        ! Initial version:  JJMorcrette, ECMWF, jul1998
        ! Revised: MJIacono, AER, jul2006
        !------------------------------------------------------------------
        !  name     type     purpose
        ! -----  :  ----   : ----------------------------------------------
        ! ng     :  integer: Number of original g-intervals in each spectral band
        ! nspa   :  integer:
        ! nspb   :  integer:
        !wavenum1:  real   : Spectral band lower boundary in wavenumbers
        !wavenum2:  real   : Spectral band upper boundary in wavenumbers
        ! delwave:  real   : Spectral band width in wavenumbers
        !
        ! ngc    :  integer: The number of new g-intervals in each band
        ! ngs    :  integer: The cumulative sum of new g-intervals for each band
        ! ngm    :  integer: The index of each new g-interval relative to the
        !                    original 16 g-intervals in each band
        ! ngn    :  integer: The number of original g-intervals that are
        !                    combined to make each new g-intervals in each band
        ! ngb    :  integer: The band index for each new g-interval
        ! wt     :  real   : RRTM weights for the original 16 g-intervals
        ! rwgt   :  real   : Weights for combining original 16 g-intervals
        !                    (224 total) into reduced set of g-intervals
        !                    (112 total)
        !------------------------------------------------------------------
        INTEGER :: nspa(jpb1:jpb2)
        INTEGER :: nspb(jpb1:jpb2)
        PUBLIC kgen_read_externs_rrsw_wvn
    CONTAINS

    ! write subroutines

    ! module extern variables

    SUBROUTINE kgen_read_externs_rrsw_wvn(kgen_unit)
        INTEGER, INTENT(IN) :: kgen_unit
        READ(UNIT=kgen_unit) nspa
        READ(UNIT=kgen_unit) nspb
    END SUBROUTINE kgen_read_externs_rrsw_wvn

    END MODULE rrsw_wvn
