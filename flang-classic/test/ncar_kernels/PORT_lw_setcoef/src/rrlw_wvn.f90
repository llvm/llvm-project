
! KGEN-generated Fortran source file
!
! Filename    : rrlw_wvn.f90
! Generated at: 2015-07-26 18:24:46
! KGEN version: 0.4.13



    MODULE rrlw_wvn
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !      use parkind, only : jpim, jprb
        USE parrrtm, ONLY: nbndlw
        IMPLICIT NONE
        !------------------------------------------------------------------
        ! rrtmg_lw spectral information
        ! Initial version:  JJMorcrette, ECMWF, jul1998
        ! Revised: MJIacono, AER, jun2006
        !------------------------------------------------------------------
        !  name     type     purpose
        ! -----  :  ----   : ----------------------------------------------
        ! ng     :  integer: Number of original g-intervals in each spectral band
        ! nspa   :  integer: For the lower atmosphere, the number of reference
        !                    atmospheres that are stored for each spectral band
        !                    per pressure level and temperature.  Each of these
        !                    atmospheres has different relative amounts of the
        !                    key species for the band (i.e. different binary
        !                    species parameters).
        ! nspb   :  integer: Same as nspa for the upper atmosphere
        !wavenum1:  real   : Spectral band lower boundary in wavenumbers
        !wavenum2:  real   : Spectral band upper boundary in wavenumbers
        ! delwave:  real   : Spectral band width in wavenumbers
        ! totplnk:  real   : Integrated Planck value for each band; (band 16
        !                    includes total from 2600 cm-1 to infinity)
        !                    Used for calculation across total spectrum
        !totplk16:  real   : Integrated Planck value for band 16 (2600-3250 cm-1)
        !                    Used for calculation in band 16 only if
        !                    individual band output requested
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
        !                    (256 total) into reduced set of g-intervals
        !                    (140 total)
        ! nxmol  :  integer: Number of cross-section molecules
        ! ixindx :  integer: Flag for active cross-sections in calculation
        !------------------------------------------------------------------
        REAL(KIND=r8) :: totplnk(181,nbndlw)
        REAL(KIND=r8) :: totplk16(181)
        PUBLIC kgen_read_externs_rrlw_wvn
    CONTAINS

    ! write subroutines

    ! module extern variables

    SUBROUTINE kgen_read_externs_rrlw_wvn(kgen_unit)
        INTEGER, INTENT(IN) :: kgen_unit
        READ(UNIT=kgen_unit) totplnk
        READ(UNIT=kgen_unit) totplk16
    END SUBROUTINE kgen_read_externs_rrlw_wvn

    END MODULE rrlw_wvn
