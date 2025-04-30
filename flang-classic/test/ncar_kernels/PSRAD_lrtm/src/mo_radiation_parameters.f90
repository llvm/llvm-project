
! KGEN-generated Fortran source file
!
! Filename    : mo_radiation_parameters.f90
! Generated at: 2015-02-19 15:30:29
! KGEN version: 0.4.4



    MODULE mo_radiation_parameters
        USE mo_kind, ONLY: wp
        IMPLICIT NONE
        PRIVATE
        PUBLIC i_overlap, l_do_sep_clear_sky
        PUBLIC rad_undef
        ! Standalone radiative transfer parameters
        PUBLIC do_gpoint ! Standalone use only
        ! 1.0 NAMELIST global variables and parameters
        ! --------------------------------
        !< diurnal cycle
        !< &! switch on/off diagnostic
        !of instantaneous aerosol solar (lradforcing(1)) and
        !thermal (lradforcing(2)) radiation forcing
        !< switch to specify perpetual vsop87 year
        !< year if (lyr_perp == .TRUE.)
        !< 0=annual cycle; 1-12 for perpetual month
        ! nmonth currently works for zonal mean ozone and the orbit (year 1987) only
        !< mode of solar constant calculation
        !< default is rrtm solar constant
        !< number of shortwave bands, set in setup
        ! Spectral sampling
        ! 1 is broadband, 2 is MCSI, 3 and up are teams
        ! Number of g-points per time step using MCSI
        ! Integer for perturbing random number seeds
        ! Use unique spectral samples under MCSI? Not yet implemented
        INTEGER :: do_gpoint = 0 ! Standalone use only - specify gpoint to use
        ! Radiation driver
        LOGICAL :: l_do_sep_clear_sky = .true. ! Compute clear-sky fluxes by removing clouds
        INTEGER :: i_overlap = 1 ! 1 = max-ran, 2 = max, 3 = ran
        ! Use separate water vapor amounts in clear, cloudy skies
        !
        ! --- Switches for radiative agents
        !
        !< water vapor, clouds and ice for radiation
        !< carbon dioxide
        !< methane
        !< ozone
        !< molecular oxygen
        !< nitrous oxide
        !< cfc11 and cfc12
        !< greenhouse gase scenario
        !< aerosol model
        !< factor for external co2 scenario (ico2=4)
        !
        ! --- Default gas volume mixing ratios - 1990 values (CMIP5)
        !
        !< CO2
        !< CH4
        !< O2
        !< N20
        !< CFC 11 and CFC 12
        !
        ! 2.0 Non NAMELIST global variables and parameters
        ! --------------------------------
        !
        ! --- radiative transfer parameters
        !
        !< LW Emissivity Factor
        !< LW Diffusivity Factor
        REAL(KIND=wp), parameter :: rad_undef = -999._wp
        !
        !
        !< default solar constant [W/m2] for
        !  AMIP-type CMIP5 simulation
        !++hs
        !< local (orbit relative and possibly
        !                                            time dependent) solar constant
        !< orbit and time dependent solar constant for radiation time step
        !< fraction of TSI in the 14 RRTM SW bands
        !--hs
        !< solar declination at current time step
        !
        ! 3.0 Variables computed by routines in mo_radiation (export to submodels)
        ! --------------------------------
        !
        ! setup_radiation
            PUBLIC read_externs_mo_radiation_parameters
        CONTAINS

        ! module extern variables

        SUBROUTINE read_externs_mo_radiation_parameters(kgen_unit)
        integer, intent(in) :: kgen_unit
        READ(UNIT=kgen_unit) do_gpoint
        READ(UNIT=kgen_unit) l_do_sep_clear_sky
        READ(UNIT=kgen_unit) i_overlap
        END SUBROUTINE read_externs_mo_radiation_parameters


        ! read subroutines
        !---------------------------------------------------------------------------
        !>
        !! @brief Scans a block and fills with solar parameters
        !!
        !! @remarks: This routine calculates the solar zenith angle for each
        !! point in a block of data.  For simulations with no dirunal cycle
        !! the cosine of the zenith angle is set to its average value (assuming
        !! negatives to be zero and for a day divided into nds intervals).
        !! Additionally a field is set indicating the fraction of the day over
        !! which the solar zenith angle is greater than zero.  Otherwise the field
        !! is set to 1 or 0 depending on whether the zenith angle is greater or
        !! less than 1.
        !

    END MODULE mo_radiation_parameters
