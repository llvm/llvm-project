
! KGEN-generated Fortran source file
!
! Filename    : rrtmg_lw_rad.f90
! Generated at: 2015-07-26 20:37:03
! KGEN version: 0.4.13



    MODULE rrtmg_lw_rad
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        !  --------------------------------------------------------------------------
        ! |                                                                          |
        ! |  Copyright 2002-2007, Atmospheric & Environmental Research, Inc. (AER).  |
        ! |  This software may be used, copied, or redistributed as long as it is    |
        ! |  not sold and this copyright notice is reproduced on each copy made.     |
        ! |  This model is provided as is without any express or implied warranties. |
        ! |                       (http://www.rtweb.aer.com/)                        |
        ! |                                                                          |
        !  --------------------------------------------------------------------------
        !
        ! ****************************************************************************
        ! *                                                                          *
        ! *                              RRTMG_LW                                    *
        ! *                                                                          *
        ! *                                                                          *
        ! *                                                                          *
        ! *                   a rapid radiative transfer model                       *
        ! *                       for the longwave region                            *
        ! *             for application to general circulation models                *
        ! *                                                                          *
        ! *                                                                          *
        ! *            Atmospheric and Environmental Research, Inc.                  *
        ! *                        131 Hartwell Avenue                               *
        ! *                        Lexington, MA 02421                               *
        ! *                                                                          *
        ! *                                                                          *
        ! *                           Eli J. Mlawer                                  *
        ! *                        Jennifer S. Delamere                              *
        ! *                         Michael J. Iacono                                *
        ! *                         Shepard A. Clough                                *
        ! *                                                                          *
        ! *                                                                          *
        ! *                                                                          *
        ! *                                                                          *
        ! *                                                                          *
        ! *                                                                          *
        ! *                       email:  miacono@aer.com                            *
        ! *                       email:  emlawer@aer.com                            *
        ! *                       email:  jdelamer@aer.com                           *
        ! *                                                                          *
        ! *        The authors wish to acknowledge the contributions of the          *
        ! *        following people:  Steven J. Taubman, Karen Cady-Pereira,         *
        ! *        Patrick D. Brown, Ronald E. Farren, Luke Chen, Robert Bergstrom.  *
        ! *                                                                          *
        ! ****************************************************************************
        ! -------- Modules --------
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !      use parkind, only : jpim, jprb
        ! Move call to rrtmg_lw_ini and following use association to
        ! GCM initialization area
        !      use rrtmg_lw_init, only: rrtmg_lw_ini
        USE rrtmg_lw_rtrnmc, ONLY: rtrnmc
        IMPLICIT NONE
        ! public interfaces/functions/subroutines
        PUBLIC rrtmg_lw
        !------------------------------------------------------------------
        CONTAINS

        ! write subroutines
        ! No subroutines
        ! No module extern variables
        !------------------------------------------------------------------
        !------------------------------------------------------------------
        ! Public subroutines
        !------------------------------------------------------------------

        SUBROUTINE rrtmg_lw(ncol, nlay, kgen_unit)
                USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
            ! -------- Description --------
            ! This program is the driver subroutine for RRTMG_LW, the AER LW radiation
            ! model for application to GCMs, that has been adapted from RRTM_LW for
            ! improved efficiency.
            !
            ! NOTE: The call to RRTMG_LW_INI should be moved to the GCM initialization
            !  area, since this has to be called only once.
            !
            ! This routine:
            !    a) calls INATM to read in the atmospheric profile from GCM;
            !       all layering in RRTMG is ordered from surface to toa.
            !    b) calls CLDPRMC to set cloud optical depth for McICA based
            !       on input cloud properties
            !    c) calls SETCOEF to calculate various quantities needed for
            !       the radiative transfer algorithm
            !    d) calls TAUMOL to calculate gaseous optical depths for each
            !       of the 16 spectral bands
            !    e) calls RTRNMC (for both clear and cloudy profiles) to perform the
            !       radiative transfer calculation using McICA, the Monte-Carlo
            !       Independent Column Approximation, to represent sub-grid scale
            !       cloud variability
            !    f) passes the necessary fluxes and cooling rates back to GCM
            !
            ! Two modes of operation are possible:
            !     The mode is chosen by using either rrtmg_lw.nomcica.f90 (to not use
            !     McICA) or rrtmg_lw.f90 (to use McICA) to interface with a GCM.
            !
            !    1) Standard, single forward model calculation (imca = 0)
            !    2) Monte Carlo Independent Column Approximation (McICA, Pincus et al.,
            !       JC, 2003) method is applied to the forward model calculation (imca = 1)
            !
            ! This call to RRTMG_LW must be preceeded by a call to the module
            !     mcica_subcol_gen_lw.f90 to run the McICA sub-column cloud generator,
            !     which will provide the cloud physical or cloud optical properties
            !     on the RRTMG quadrature point (ngpt) dimension.
            !
            ! Two methods of cloud property input are possible:
            !     Cloud properties can be input in one of two ways (controlled by input
            !     flags inflglw, iceflglw, and liqflglw; see text file rrtmg_lw_instructions
            !     and subroutine rrtmg_lw_cldprop.f90 for further details):
            !
            !    1) Input cloud fraction and cloud optical depth directly (inflglw = 0)
            !    2) Input cloud fraction and cloud physical properties (inflglw = 1 or 2);
            !       cloud optical properties are calculated by cldprop or cldprmc based
            !       on input settings of iceflglw and liqflglw
            !
            ! One method of aerosol property input is possible:
            !     Aerosol properties can be input in only one way (controlled by input
            !     flag iaer, see text file rrtmg_lw_instructions for further details):
            !
            !    1) Input aerosol optical depth directly by layer and spectral band (iaer=10);
            !       band average optical depth at the mid-point of each spectral band.
            !       RRTMG_LW currently treats only aerosol absorption;
            !       scattering capability is not presently available.
            !
            !
            ! ------- Modifications -------
            !
            ! This version of RRTMG_LW has been modified from RRTM_LW to use a reduced
            ! set of g-points for application to GCMs.
            !
            !-- Original version (derived from RRTM_LW), reduction of g-points, other
            !   revisions for use with GCMs.
            !     1999: M. J. Iacono, AER, Inc.
            !-- Adapted for use with NCAR/CAM.
            !     May 2004: M. J. Iacono, AER, Inc.
            !-- Revised to add McICA capability.
            !     Nov 2005: M. J. Iacono, AER, Inc.
            !-- Conversion to F90 formatting for consistency with rrtmg_sw.
            !     Feb 2007: M. J. Iacono, AER, Inc.
            !-- Modifications to formatting to use assumed-shape arrays.
            !     Aug 2007: M. J. Iacono, AER, Inc.
            !-- Modified to add longwave aerosol absorption.
            !     Apr 2008: M. J. Iacono, AER, Inc.
            ! --------- Modules ----------
            USE parrrtm, ONLY: nbndlw
            USE parrrtm, ONLY: ngptlw
            ! ------- Declarations -------
            ! ----- Input -----
            integer, intent(in) :: kgen_unit
            INTEGER*8 :: kgen_intvar, start_clock, stop_clock, rate_clock
            TYPE(check_t):: check_status
            REAL(KIND=kgen_dp) :: tolerance
            ! chunk identifier
            INTEGER, intent(in) :: ncol ! Number of horizontal columns
            INTEGER, intent(in) :: nlay ! Number of model layers
            ! Cloud overlap method
            !    0: Clear only
            !    1: Random
            !    2: Maximum/random
            !    3: Maximum
            ! Layer pressures (hPa, mb)
            !    Dimensions: (ncol,nlay)
            ! Interface pressures (hPa, mb)
            !    Dimensions: (ncol,nlay+1)
            ! Layer temperatures (K)
            !    Dimensions: (ncol,nlay)
            ! Interface temperatures (K)
            !    Dimensions: (ncol,nlay+1)
            ! Surface temperature (K)
            !    Dimensions: (ncol)
            ! H2O volume mixing ratio
            !    Dimensions: (ncol,nlay)
            ! O3 volume mixing ratio
            !    Dimensions: (ncol,nlay)
            ! CO2 volume mixing ratio
            !    Dimensions: (ncol,nlay)
            ! Methane volume mixing ratio
            !    Dimensions: (ncol,nlay)
            ! O2 volume mixing ratio
            !    Dimensions: (ncol,nlay)
            ! Nitrous oxide volume mixing ratio
            !    Dimensions: (ncol,nlay)
            ! CFC11 volume mixing ratio
            !    Dimensions: (ncol,nlay)
            ! CFC12 volume mixing ratio
            !    Dimensions: (ncol,nlay)
            ! CFC22 volume mixing ratio
            !    Dimensions: (ncol,nlay)
            ! CCL4 volume mixing ratio
            !    Dimensions: (ncol,nlay)
            ! Surface emissivity
            !    Dimensions: (ncol,nbndlw)
            ! Flag for cloud optical properties
            ! Flag for ice particle specification
            ! Flag for liquid droplet specification
            ! Cloud fraction
            !    Dimensions: (ngptlw,ncol,nlay)
            ! Cloud ice water path (g/m2)
            !    Dimensions: (ngptlw,ncol,nlay)
            ! Cloud liquid water path (g/m2)
            !    Dimensions: (ngptlw,ncol,nlay)
            ! Cloud ice effective radius (microns)
            !    Dimensions: (ncol,nlay)
            ! Cloud water drop effective radius (microns)
            !    Dimensions: (ncol,nlay)
            ! Cloud optical depth
            !    Dimensions: (ngptlw,ncol,nlay)
            !      real(kind=r8), intent(in) :: ssacmcl(:,:,:)      ! Cloud single scattering albedo
            !    Dimensions: (ngptlw,ncol,nlay)
            !   for future expansion
            !   lw scattering not yet available
            !      real(kind=r8), intent(in) :: asmcmcl(:,:,:)      ! Cloud asymmetry parameter
            !    Dimensions: (ngptlw,ncol,nlay)
            !   for future expansion
            !   lw scattering not yet available
            ! aerosol optical depth
            !   at mid-point of LW spectral bands
            !    Dimensions: (ncol,nlay,nbndlw)
            !      real(kind=r8), intent(in) :: ssaaer(:,:,:)       ! aerosol single scattering albedo
            !    Dimensions: (ncol,nlay,nbndlw)
            !   for future expansion
            !   (lw aerosols/scattering not yet available)
            !      real(kind=r8), intent(in) :: asmaer(:,:,:)       ! aerosol asymmetry parameter
            !    Dimensions: (ncol,nlay,nbndlw)
            !   for future expansion
            !   (lw aerosols/scattering not yet available)
            ! ----- Output -----
            ! Total sky longwave upward flux (W/m2)
            !    Dimensions: (ncol,nlay+1)
            ! Total sky longwave downward flux (W/m2)
            !    Dimensions: (ncol,nlay+1)
            ! Total sky longwave radiative heating rate (K/d)
            !    Dimensions: (ncol,nlay)
            ! Clear sky longwave upward flux (W/m2)
            !    Dimensions: (ncol,nlay+1)
            ! Clear sky longwave downward flux (W/m2)
            !    Dimensions: (ncol,nlay+1)
            ! Clear sky longwave radiative heating rate (K/d)
            !    Dimensions: (ncol,nlay)
            ! Total sky longwave upward flux spectral (W/m2)
            !    Dimensions: (nbndlw,ncol,nlay+1)
            ! Total sky longwave downward flux spectral (W/m2)
            !    Dimensions: (nbndlw,ncol,nlay+1)
            ! ----- Local -----
            ! Control
            INTEGER :: istart ! beginning band of calculation
            INTEGER :: iend ! ending band of calculation
            INTEGER :: iout ! output option flag (inactive)
            ! aerosol option flag
            ! column loop index
            ! flag for mcica [0=off, 1=on]
            ! value for changing mcica permute seed
            ! layer loop index
            ! g-point loop index
            ! Atmosphere
            ! layer pressures (mb)
            ! layer temperatures (K)
            REAL(KIND=r8) :: pz(ncol,0:nlay) ! level (interface) pressures (hPa, mb)
            ! level (interface) temperatures (K)
            ! surface temperature (K)
            ! dry air column density (mol/cm2)
            ! broadening gas column density (mol/cm2)
            ! molecular amounts (mol/cm-2)
            ! cross-section amounts (mol/cm-2)
            REAL(KIND=r8) :: pwvcm(ncol) ! precipitable water vapor (cm)
            REAL(KIND=r8) :: semiss(ncol,nbndlw) ! lw surface emissivity
            REAL(KIND=r8) :: fracs(ncol,nlay,ngptlw) !
            ! gaseous optical depths
            REAL(KIND=r8) :: taut(ncol,nlay,ngptlw) ! gaseous + aerosol optical depths
            ! aerosol optical depth
            !      real(kind=r8) :: ssaa(nlay,nbndlw)        ! aerosol single scattering albedo
            !   for future expansion
            !   (lw aerosols/scattering not yet available)
            !      real(kind=r8) :: asma(nlay+1,nbndlw)      ! aerosol asymmetry parameter
            !   for future expansion
            !   (lw aerosols/scattering not yet available)
            ! Atmosphere - setcoef
            ! tropopause layer index
            ! lookup table index
            ! lookup table index
            ! lookup table index
            REAL(KIND=r8) :: planklay(ncol,nlay,nbndlw) !
            REAL(KIND=r8) :: planklev(ncol,0:nlay,nbndlw) !
            REAL(KIND=r8) :: plankbnd(ncol,nbndlw) !
            ! column amount (h2o)
            ! column amount (co2)
            ! column amount (o3)
            ! column amount (n2o)
            ! column amount (co)
            ! column amount (ch4)
            ! column amount (o2)
            ! column amount (broadening gases)
            !
            !
            ! Atmosphere/clouds - cldprop
            INTEGER :: ncbands(ncol) ! number of cloud spectral bands
            ! flag for cloud property method
            ! flag for ice cloud properties
            ! flag for liquid cloud properties
            ! Atmosphere/clouds - cldprmc [mcica]
            REAL(KIND=r8) :: cldfmc(ncol,ngptlw,nlay) ! cloud fraction [mcica]
            ! cloud ice water path [mcica]
            ! cloud liquid water path [mcica]
            ! liquid particle size (microns)
            ! ice particle effective radius (microns)
            ! ice particle generalized effective size (microns)
            REAL(KIND=r8) :: taucmc(ncol,ngptlw,nlay) ! cloud optical depth [mcica]
            !      real(kind=r8) :: ssacmc(ngptlw,nlay)     ! cloud single scattering albedo [mcica]
            !   for future expansion
            !   (lw scattering not yet available)
            !      real(kind=r8) :: asmcmc(ngptlw,nlay)     ! cloud asymmetry parameter [mcica]
            !   for future expansion
            !   (lw scattering not yet available)
            ! Output
            REAL(KIND=r8) :: totuflux(ncol,0:nlay)
            REAL(KIND=r8) :: ref_totuflux(ncol,0:nlay) ! upward longwave flux (w/m2)
            REAL(KIND=r8) :: totdflux(ncol,0:nlay)
            REAL(KIND=r8) :: ref_totdflux(ncol,0:nlay) ! downward longwave flux (w/m2)
            REAL(KIND=r8) :: totufluxs(ncol,nbndlw,0:nlay)
            REAL(KIND=r8) :: ref_totufluxs(ncol,nbndlw,0:nlay) ! upward longwave flux spectral (w/m2)
            REAL(KIND=r8) :: totdfluxs(ncol,nbndlw,0:nlay)
            REAL(KIND=r8) :: ref_totdfluxs(ncol,nbndlw,0:nlay) ! downward longwave flux spectral (w/m2)
            REAL(KIND=r8) :: fnet(ncol,0:nlay)
            REAL(KIND=r8) :: ref_fnet(ncol,0:nlay) ! net longwave flux (w/m2)
            REAL(KIND=r8) :: htr(ncol,0:nlay)
            REAL(KIND=r8) :: ref_htr(ncol,0:nlay) ! longwave heating rate (k/day)
            REAL(KIND=r8) :: totuclfl(ncol,0:nlay)
            REAL(KIND=r8) :: ref_totuclfl(ncol,0:nlay) ! clear sky upward longwave flux (w/m2)
            REAL(KIND=r8) :: totdclfl(ncol,0:nlay)
            REAL(KIND=r8) :: ref_totdclfl(ncol,0:nlay) ! clear sky downward longwave flux (w/m2)
            REAL(KIND=r8) :: fnetc(ncol,0:nlay)
            REAL(KIND=r8) :: ref_fnetc(ncol,0:nlay) ! clear sky net longwave flux (w/m2)
            REAL(KIND=r8) :: htrc(ncol,0:nlay)
            REAL(KIND=r8) :: ref_htrc(ncol,0:nlay) ! clear sky longwave heating rate (k/day)
            !DIR$ ATTRIBUTES ALIGN : 64 ::  pz
            ! Initializations
            ! orig:   fluxfac = pi * 2.d4 ! orig:   fluxfac = pi * 2.d4
            ! Set imca to select calculation type:
            !  imca = 0, use standard forward model calculation
            !  imca = 1, use McICA for Monte Carlo treatment of sub-grid cloud variability
            ! *** This version uses McICA (imca = 1) ***
            ! Set icld to select of clear or cloud calculation and cloud overlap method
            ! icld = 0, clear only
            ! icld = 1, with clouds using random cloud overlap
            ! icld = 2, with clouds using maximum/random cloud overlap
            ! icld = 3, with clouds using maximum cloud overlap (McICA only)
            ! Set iaer to select aerosol option
            ! iaer = 0, no aerosols
            ! iaer = 10, input total aerosol optical depth (tauaer) directly
            !Call model and data initialization, compute lookup tables, perform
            ! reduction of g-points from 256 to 140 for input absorption coefficient
            ! data and other arrays.
            !
            ! In a GCM this call should be placed in the model initialization
            ! area, since this has to be called only once.
            !      call rrtmg_lw_ini
            !  This is the main longitude/column loop within RRTMG.
            !  Prepare atmospheric profile from GCM for use in RRTMG, and define
            !  other input parameters.
            !  For cloudy atmosphere, use cldprop to set cloud optical properties based on
            !  input cloud physical properties.  Select method based on choices described
            !  in cldprop.  Cloud fraction, water path, liquid droplet and ice particle
            !  effective radius must be passed into cldprop.  Cloud fraction and cloud
            !  optical depth are transferred to rrtmg_lw arrays in cldprop.
            ! Calculate information needed by the radiative transfer routine
            ! that is specific to this atmosphere, especially some of the
            ! coefficients and indices needed to compute the optical depths
            ! by interpolating data from stored reference atmospheres.
            ! Call the radiative transfer routine.
            ! Either routine can be called to do clear sky calculation.  If clouds
            ! are present, then select routine based on cloud overlap assumption
            ! to be used.  Clear sky calculation is done simultaneously.
            ! For McICA, RTRNMC is called for clear and cloudy calculations.
            !orig tolerance = 1.E-14
            tolerance = 7.E-14  ! PGI/NVIDIA
            CALL kgen_init_check(check_status, tolerance)
            READ(UNIT=kgen_unit) istart
            READ(UNIT=kgen_unit) iend
            READ(UNIT=kgen_unit) iout
            READ(UNIT=kgen_unit) pz
            READ(UNIT=kgen_unit) pwvcm
            READ(UNIT=kgen_unit) semiss
            READ(UNIT=kgen_unit) fracs
            READ(UNIT=kgen_unit) taut
            READ(UNIT=kgen_unit) planklay
            READ(UNIT=kgen_unit) planklev
            READ(UNIT=kgen_unit) plankbnd
            READ(UNIT=kgen_unit) ncbands
            READ(UNIT=kgen_unit) cldfmc
            READ(UNIT=kgen_unit) taucmc
            READ(UNIT=kgen_unit) totuflux
            READ(UNIT=kgen_unit) totdflux
            READ(UNIT=kgen_unit) totufluxs
            READ(UNIT=kgen_unit) totdfluxs
            READ(UNIT=kgen_unit) fnet
            READ(UNIT=kgen_unit) htr
            READ(UNIT=kgen_unit) totuclfl
            READ(UNIT=kgen_unit) totdclfl
            READ(UNIT=kgen_unit) fnetc
            READ(UNIT=kgen_unit) htrc

            READ(UNIT=kgen_unit) ref_totuflux
            READ(UNIT=kgen_unit) ref_totdflux
            READ(UNIT=kgen_unit) ref_totufluxs
            READ(UNIT=kgen_unit) ref_totdfluxs
            READ(UNIT=kgen_unit) ref_fnet
            READ(UNIT=kgen_unit) ref_htr
            READ(UNIT=kgen_unit) ref_totuclfl
            READ(UNIT=kgen_unit) ref_totdclfl
            READ(UNIT=kgen_unit) ref_fnetc
            READ(UNIT=kgen_unit) ref_htrc


            ! call to kernel
      call rtrnmc(ncol, nlay, istart, iend, iout, pz, semiss, ncbands, &
           cldfmc, taucmc, planklay, planklev, plankbnd, &
           pwvcm, fracs, taut, &
           totuflux, totdflux, fnet, htr, &
           totuclfl, totdclfl, fnetc, htrc, totufluxs, totdfluxs )
            ! kernel verification for output variables
            CALL kgen_verify_real_r8_dim2( "totuflux", check_status, totuflux, ref_totuflux)
            CALL kgen_verify_real_r8_dim2( "totdflux", check_status, totdflux, ref_totdflux)
            CALL kgen_verify_real_r8_dim3( "totufluxs", check_status, totufluxs, ref_totufluxs)
            CALL kgen_verify_real_r8_dim3( "totdfluxs", check_status, totdfluxs, ref_totdfluxs)
            CALL kgen_verify_real_r8_dim2( "fnet", check_status, fnet, ref_fnet)
            CALL kgen_verify_real_r8_dim2( "htr", check_status, htr, ref_htr)
            CALL kgen_verify_real_r8_dim2( "totuclfl", check_status, totuclfl, ref_totuclfl)
            CALL kgen_verify_real_r8_dim2( "totdclfl", check_status, totdclfl, ref_totdclfl)
            CALL kgen_verify_real_r8_dim2( "fnetc", check_status, fnetc, ref_fnetc)
            CALL kgen_verify_real_r8_dim2( "htrc", check_status, htrc, ref_htrc)
            CALL kgen_print_check("rtrnmc", check_status)
            CALL system_clock(start_clock, rate_clock)
            DO kgen_intvar=1,10
                CALL rtrnmc(ncol, nlay, istart, iend, iout, pz, semiss, ncbands, cldfmc, taucmc, planklay, planklev, plankbnd, pwvcm, fracs, taut, totuflux, totdflux, fnet, htr, totuclfl, totdclfl, fnetc, htrc, totufluxs, totdfluxs)
            END DO
            CALL system_clock(stop_clock, rate_clock)
            WRITE(*,*)
            PRINT *, "Elapsed time (sec): ", (stop_clock - start_clock)/REAL(rate_clock*10)
            !  Transfer up and down fluxes and heating rate to output arrays.
            !  Vertical indexing goes from bottom to top
        CONTAINS

        ! write subroutines
            SUBROUTINE kgen_read_real_r8_dim2(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                real(KIND=r8), INTENT(OUT), ALLOCATABLE, DIMENSION(:,:) :: var
                LOGICAL :: is_true
                INTEGER :: idx1,idx2
                INTEGER, DIMENSION(2,2) :: kgen_bound

                READ(UNIT = kgen_unit) is_true

                IF ( is_true ) THEN
                    READ(UNIT = kgen_unit) kgen_bound(1, 1)
                    READ(UNIT = kgen_unit) kgen_bound(2, 1)
                    READ(UNIT = kgen_unit) kgen_bound(1, 2)
                    READ(UNIT = kgen_unit) kgen_bound(2, 2)
                    ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
                    READ(UNIT = kgen_unit) var
                    IF ( PRESENT(printvar) ) THEN
                        PRINT *, "** " // printvar // " **", var
                    END IF
                END IF
            END SUBROUTINE kgen_read_real_r8_dim2

            SUBROUTINE kgen_read_real_r8_dim3(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                real(KIND=r8), INTENT(OUT), ALLOCATABLE, DIMENSION(:,:,:) :: var
                LOGICAL :: is_true
                INTEGER :: idx1,idx2,idx3
                INTEGER, DIMENSION(2,3) :: kgen_bound

                READ(UNIT = kgen_unit) is_true

                IF ( is_true ) THEN
                    READ(UNIT = kgen_unit) kgen_bound(1, 1)
                    READ(UNIT = kgen_unit) kgen_bound(2, 1)
                    READ(UNIT = kgen_unit) kgen_bound(1, 2)
                    READ(UNIT = kgen_unit) kgen_bound(2, 2)
                    READ(UNIT = kgen_unit) kgen_bound(1, 3)
                    READ(UNIT = kgen_unit) kgen_bound(2, 3)
                    ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1, kgen_bound(2, 3) - kgen_bound(1, 3) + 1))
                    READ(UNIT = kgen_unit) var
                    IF ( PRESENT(printvar) ) THEN
                        PRINT *, "** " // printvar // " **", var
                    END IF
                END IF
            END SUBROUTINE kgen_read_real_r8_dim3


        ! verify subroutines
            SUBROUTINE kgen_verify_real_r8_dim2( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                real(KIND=r8), intent(in), DIMENSION(:,:) :: var, ref_var
                real(KIND=r8) :: nrmsdiff, rmsdiff
                real(KIND=r8), allocatable, DIMENSION(:,:) :: temp, temp2
                integer :: n
                check_status%numTotal = check_status%numTotal + 1
                IF ( ALL( var == ref_var ) ) THEN
                
                    check_status%numIdentical = check_status%numIdentical + 1            
                    if(check_status%verboseLevel > 1) then
                        WRITE(*,*)
                        WRITE(*,*) "All elements of ", trim(adjustl(varname)), " are IDENTICAL."
                        !WRITE(*,*) "KERNEL: ", var
                        !WRITE(*,*) "REF.  : ", ref_var
                        IF ( ALL( var == 0 ) ) THEN
                            if(check_status%verboseLevel > 2) then
                                WRITE(*,*) "All values are zero."
                            end if
                        END IF
                    end if
                ELSE
                    allocate(temp(SIZE(var,dim=1),SIZE(var,dim=2)))
                    allocate(temp2(SIZE(var,dim=1),SIZE(var,dim=2)))
                
                    n = count(var/=ref_var)
                    where(abs(ref_var) > check_status%minvalue)
                        temp  = ((var-ref_var)/ref_var)**2
                        temp2 = (var-ref_var)**2
                    elsewhere
                        temp  = (var-ref_var)**2
                        temp2 = temp
                    endwhere
                    nrmsdiff = sqrt(sum(temp)/real(n))
                    rmsdiff = sqrt(sum(temp2)/real(n))
                
                    if(check_status%verboseLevel > 0) then
                        WRITE(*,*)
                        WRITE(*,*) trim(adjustl(varname)), " is NOT IDENTICAL."
                        WRITE(*,*) count( var /= ref_var), " of ", size( var ), " elements are different."
                        if(check_status%verboseLevel > 1) then
                            WRITE(*,*) "Average - kernel ", sum(var)/real(size(var))
                            WRITE(*,*) "Average - reference ", sum(ref_var)/real(size(ref_var))
                        endif
                        WRITE(*,*) "RMS of difference is ",rmsdiff
                        WRITE(*,*) "Normalized RMS of difference is ",nrmsdiff
                    end if
                
                    if (nrmsdiff > check_status%tolerance) then
                        check_status%numFatal = check_status%numFatal+1
                    else
                        check_status%numWarning = check_status%numWarning+1
                    endif
                
                    deallocate(temp,temp2)
                END IF
            END SUBROUTINE kgen_verify_real_r8_dim2

            SUBROUTINE kgen_verify_real_r8_dim3( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                real(KIND=r8), intent(in), DIMENSION(:,:,:) :: var, ref_var
                real(KIND=r8) :: nrmsdiff, rmsdiff
                real(KIND=r8), allocatable, DIMENSION(:,:,:) :: temp, temp2
                integer :: n
                check_status%numTotal = check_status%numTotal + 1
                IF ( ALL( var == ref_var ) ) THEN
                
                    check_status%numIdentical = check_status%numIdentical + 1            
                    if(check_status%verboseLevel > 1) then
                        WRITE(*,*)
                        WRITE(*,*) "All elements of ", trim(adjustl(varname)), " are IDENTICAL."
                        !WRITE(*,*) "KERNEL: ", var
                        !WRITE(*,*) "REF.  : ", ref_var
                        IF ( ALL( var == 0 ) ) THEN
                            if(check_status%verboseLevel > 2) then
                                WRITE(*,*) "All values are zero."
                            end if
                        END IF
                    end if
                ELSE
                    allocate(temp(SIZE(var,dim=1),SIZE(var,dim=2),SIZE(var,dim=3)))
                    allocate(temp2(SIZE(var,dim=1),SIZE(var,dim=2),SIZE(var,dim=3)))
                
                    n = count(var/=ref_var)
                    where(abs(ref_var) > check_status%minvalue)
                        temp  = ((var-ref_var)/ref_var)**2
                        temp2 = (var-ref_var)**2
                    elsewhere
                        temp  = (var-ref_var)**2
                        temp2 = temp
                    endwhere
                    nrmsdiff = sqrt(sum(temp)/real(n))
                    rmsdiff = sqrt(sum(temp2)/real(n))
                
                    if(check_status%verboseLevel > 0) then
                        WRITE(*,*)
                        WRITE(*,*) trim(adjustl(varname)), " is NOT IDENTICAL."
                        WRITE(*,*) count( var /= ref_var), " of ", size( var ), " elements are different."
                        if(check_status%verboseLevel > 1) then
                            WRITE(*,*) "Average - kernel ", sum(var)/real(size(var))
                            WRITE(*,*) "Average - reference ", sum(ref_var)/real(size(ref_var))
                        endif
                        WRITE(*,*) "RMS of difference is ",rmsdiff
                        WRITE(*,*) "Normalized RMS of difference is ",nrmsdiff
                    end if
                
                    if (nrmsdiff > check_status%tolerance) then
                        check_status%numFatal = check_status%numFatal+1
                    else
                        check_status%numWarning = check_status%numWarning+1
                    endif
                
                    deallocate(temp,temp2)
                END IF
            END SUBROUTINE kgen_verify_real_r8_dim3

        END SUBROUTINE rrtmg_lw
        !***************************************************************************

    END MODULE rrtmg_lw_rad
