
! KGEN-generated Fortran source file
!
! Filename    : constituents.F90
! Generated at: 2015-04-12 19:37:50
! KGEN version: 0.4.9



    MODULE constituents
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        !----------------------------------------------------------------------------------------------
        !
        ! Purpose: Contains data and functions for manipulating advected and non-advected constituents.
        !
        ! Revision history:
        !             B.A. Boville    Original version
        ! June 2003   P. Rasch        Add wet/dry m.r. specifier
        ! 2004-08-28  B. Eaton        Add query function to allow turning off the default 1 output of
        !                             constituents so that chemistry module can make the outfld calls.
        !                             Allow cnst_get_ind to return without aborting when constituent not
        !                             found.
        ! 2006-10-31  B. Eaton        Remove 'non-advected' constituent functionality.
        !----------------------------------------------------------------------------------------------
        IMPLICIT NONE
        PRIVATE
        !
        ! Public interfaces
        !
        ! add a constituent to the list of advected constituents
        ! returns the number of available slots in the constituent array
        ! get the index of a constituent
        ! get the type of a constituent
        ! get the type of a constituent
        ! get the molecular diffusion type of a constituent
        ! query whether constituent initial values are read from initial file
        ! check that number of constituents added equals dimensions (pcnst)
        ! Returns true if default 1 output was specified in the cnst_add calls.
        ! Public data
        INTEGER, parameter, public :: pcnst  = 29 ! number of advected constituents (including water vapor)
        ! constituent names
        ! long name of constituents
        ! Namelist variables
        ! true => obtain initial tracer data from IC file
        !
        ! Constants for each tracer
        ! specific heat at constant pressure (J/kg/K)
        ! specific heat at constant volume (J/kg/K)
        ! molecular weight (kg/kmole)
        ! wet or dry mixing ratio
        ! major or minor species molecular diffusion
        ! gas constant ()
        ! minimum permitted constituent concentration (kg/kg)
        ! for backward compatibility only
        ! upper bndy condition = fixed ?
        ! upper boundary non-zero fixed constituent flux
        ! convective transport : phase 1 or phase 2?
        !++bee - temporary... These names should be declared in the module that makes the addfld and outfld calls.
        ! Lists of tracer names and diagnostics
        ! constituents after physics  (FV core only)
        ! constituents before physics (FV core only)
        ! names of horizontal advection tendencies
        ! names of vertical advection tendencies
        ! names of convection tendencies
        ! names of species slt fixer tendencies
        ! names of total tendencies of species
        ! names of total physics tendencies of species
        ! names of dme adjusted tracers (FV)
        ! names of surface fluxes of species
        ! names for horz + vert + fixer tendencies
        ! Private data
        ! index pointer to last advected tracer
        ! true => read initial values from initial file
        ! true  => default 1 output of constituents in kg/kg
        ! false => chemistry is responsible for making outfld
        !          calls for constituents
        !==============================================================================================
        CONTAINS

        ! write subroutines
        ! No subroutines
        ! No module extern variables
        !==============================================================================================

        !==============================================================================

        !==============================================================================

        !==============================================================================================

        !==============================================================================================


        !==============================================================================

        !==============================================================================

        !==============================================================================

        !==============================================================================
    END MODULE constituents
