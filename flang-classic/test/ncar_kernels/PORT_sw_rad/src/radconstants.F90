
! KGEN-generated Fortran source file
!
! Filename    : radconstants.F90
! Generated at: 2015-07-07 00:48:24
! KGEN version: 0.4.13



    MODULE radconstants
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        ! This module contains constants that are specific to the radiative transfer
        ! code used in the RRTMG model.
        IMPLICIT NONE
        PRIVATE
        ! SHORTWAVE DATA
        ! number of shorwave spectral intervals
        INTEGER, parameter, public :: nswbands = 14
        ! Wavenumbers of band boundaries
        !
        ! Note: Currently rad_solar_var extends the lowest band down to
        ! 100 cm^-1 if it is too high to cover the far-IR. Any changes meant
        ! to affect IR solar variability should take note of this.
        ! in cm^-1
        ! in cm^-1
        ! Solar irradiance at 1 A.U. in W/m^2 assumed by radiation code
        ! Rescaled so that sum is precisely 1368.22 and fractional amounts sum to 1.0
        ! None of the following comment appears to be the case any more? This
        ! should be reevalutated and/or removed.
        ! rrtmg (coarse) reference solar flux in rrtmg is initialized as the following
        ! reference data inside rrtmg seems to indicate 1366.44 instead
        !  This data references 1366.442114152342
        !real(r8), parameter :: solar_ref_band_irradiance(nbndsw) = &
        !   (/ &
        !   12.10956827000000_r8, 20.36508467999999_r8, 23.72973826333333_r8, &
        !   22.42769644333333_r8, 55.62661262000000_r8, 102.9314315544444_r8, 24.29361887666667_r8, &
        !   345.7425138000000_r8, 218.1870300666667_r8, 347.1923147000001_r8, &
        !   129.4950181200000_r8, 48.37217043000000_r8, 3.079938997898001_r8, 12.88937733000000_r8 &
        !   /)
        !  Kurucz (fine) reference would seem to imply the following but the above values are from rrtmg_sw_init
        !  (/12.109559, 20.365097, 23.729752, 22.427697, 55.626622, 102.93142, 24.293593, &
        !    345.73655, 218.18416, 347.18406, 129.49407, 50.147238, 3.1197130, 12.793834 /)
        ! These are indices to the band for diagnostic output
        ! index to sw visible band
        ! index to sw near infrared (778-1240 nm) band
        ! index to sw uv (345-441 nm) band
        ! rrtmg band for .67 micron
        ! Number of evenly spaced intervals in rh
        ! The globality of this mesh may not be necessary
        ! Perhaps it could be specific to the aerosol
        ! But it is difficult to see how refined it must be
        ! for lookup.  This value was found to be sufficient
        ! for Sulfate and probably necessary to resolve the
        ! high variation near rh = 1.  Alternative methods
        ! were found to be too slow.
        ! Optimal approach would be for cam to specify size of aerosol
        ! based on each aerosol's characteristics.  Radiation
        ! should know nothing about hygroscopic growth!
        ! LONGWAVE DATA
        ! These are indices to the band for diagnostic output
        ! index to (H20 window) LW band
        ! rrtmg band for 10.5 micron
        ! number of lw bands
        ! Longwave spectral band limits (cm-1)
        ! Longwave spectral band limits (cm-1)
        !These can go away when old camrt disappears
        ! Index of volc. abs., H2O non-window
        ! Index of volc. abs., H2O window
        ! Index of volc. cnt. abs. 0500--0650 cm-1
        ! Index of volc. cnt. abs. 0650--0800 cm-1
        ! Index of volc. cnt. abs. 0800--1000 cm-1
        ! Index of volc. cnt. abs. 1000--1200 cm-1
        ! Index of volc. cnt. abs. 1200--2000 cm-1
        ! GASES TREATED BY RADIATION (line spectrae)
        ! gasses required by radiation
        ! what is the minimum mass mixing ratio that can be supported by radiation implementation?
        ! Length of "optics type" string specified in optics files.
        CONTAINS

        ! write subroutines
        ! No subroutines
        ! No module extern variables
        !------------------------------------------------------------------------------

        !------------------------------------------------------------------------------

        !------------------------------------------------------------------------------

        !------------------------------------------------------------------------------

        !------------------------------------------------------------------------------

        !------------------------------------------------------------------------------

        !------------------------------------------------------------------------------

    END MODULE radconstants
