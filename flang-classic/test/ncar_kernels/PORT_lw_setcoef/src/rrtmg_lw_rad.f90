
! KGEN-generated Fortran source file
!
! Filename    : rrtmg_lw_rad.f90
! Generated at: 2015-07-26 18:24:46
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
        USE rrtmg_lw_setcoef, ONLY: setcoef
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
            USE parrrtm, ONLY: mxmol
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
            ! ending band of calculation
            ! output option flag (inactive)
            ! aerosol option flag
            ! column loop index
            ! flag for mcica [0=off, 1=on]
            ! value for changing mcica permute seed
            ! layer loop index
            ! g-point loop index
            ! Atmosphere
            REAL(KIND=r8) :: pavel(ncol,nlay) ! layer pressures (mb)
            REAL(KIND=r8) :: tavel(ncol,nlay) ! layer temperatures (K)
            ! level (interface) pressures (hPa, mb)
            REAL(KIND=r8) :: tz(ncol,0:nlay) ! level (interface) temperatures (K)
            REAL(KIND=r8) :: tbound(ncol) ! surface temperature (K)
            REAL(KIND=r8) :: coldry(ncol,nlay) ! dry air column density (mol/cm2)
            REAL(KIND=r8) :: wbrodl(ncol,nlay) ! broadening gas column density (mol/cm2)
            REAL(KIND=r8) :: wkl(ncol,mxmol,nlay) ! molecular amounts (mol/cm-2)
            ! cross-section amounts (mol/cm-2)
            ! precipitable water vapor (cm)
            REAL(KIND=r8) :: semiss(ncol,nbndlw) ! lw surface emissivity
            !
            ! gaseous optical depths
            ! gaseous + aerosol optical depths
            ! aerosol optical depth
            !      real(kind=r8) :: ssaa(nlay,nbndlw)        ! aerosol single scattering albedo
            !   for future expansion
            !   (lw aerosols/scattering not yet available)
            !      real(kind=r8) :: asma(nlay+1,nbndlw)      ! aerosol asymmetry parameter
            !   for future expansion
            !   (lw aerosols/scattering not yet available)
            ! Atmosphere - setcoef
            INTEGER :: laytrop(ncol)
            INTEGER :: ref_laytrop(ncol) ! tropopause layer index
            INTEGER :: jp(ncol,nlay)
            INTEGER :: ref_jp(ncol,nlay) ! lookup table index
            INTEGER :: jt(ncol,nlay)
            INTEGER :: ref_jt(ncol,nlay) ! lookup table index
            INTEGER :: jt1(ncol,nlay)
            INTEGER :: ref_jt1(ncol,nlay) ! lookup table index
            REAL(KIND=r8) :: planklay(ncol,nlay,nbndlw)
            REAL(KIND=r8) :: ref_planklay(ncol,nlay,nbndlw) !
            REAL(KIND=r8) :: planklev(ncol,0:nlay,nbndlw)
            REAL(KIND=r8) :: ref_planklev(ncol,0:nlay,nbndlw) !
            REAL(KIND=r8) :: plankbnd(ncol,nbndlw)
            REAL(KIND=r8) :: ref_plankbnd(ncol,nbndlw) !
            REAL(KIND=r8) :: colh2o(ncol,nlay)
            REAL(KIND=r8) :: ref_colh2o(ncol,nlay) ! column amount (h2o)
            REAL(KIND=r8) :: colco2(ncol,nlay)
            REAL(KIND=r8) :: ref_colco2(ncol,nlay) ! column amount (co2)
            REAL(KIND=r8) :: colo3(ncol,nlay)
            REAL(KIND=r8) :: ref_colo3(ncol,nlay) ! column amount (o3)
            REAL(KIND=r8) :: coln2o(ncol,nlay)
            REAL(KIND=r8) :: ref_coln2o(ncol,nlay) ! column amount (n2o)
            REAL(KIND=r8) :: colco(ncol,nlay)
            REAL(KIND=r8) :: ref_colco(ncol,nlay) ! column amount (co)
            REAL(KIND=r8) :: colch4(ncol,nlay)
            REAL(KIND=r8) :: ref_colch4(ncol,nlay) ! column amount (ch4)
            REAL(KIND=r8) :: colo2(ncol,nlay)
            REAL(KIND=r8) :: ref_colo2(ncol,nlay) ! column amount (o2)
            REAL(KIND=r8) :: colbrd(ncol,nlay)
            REAL(KIND=r8) :: ref_colbrd(ncol,nlay) ! column amount (broadening gases)
            INTEGER :: indself(ncol,nlay)
            INTEGER :: ref_indself(ncol,nlay)
            INTEGER :: indfor(ncol,nlay)
            INTEGER :: ref_indfor(ncol,nlay)
            REAL(KIND=r8) :: selffac(ncol,nlay)
            REAL(KIND=r8) :: ref_selffac(ncol,nlay)
            REAL(KIND=r8) :: selffrac(ncol,nlay)
            REAL(KIND=r8) :: ref_selffrac(ncol,nlay)
            REAL(KIND=r8) :: forfac(ncol,nlay)
            REAL(KIND=r8) :: ref_forfac(ncol,nlay)
            REAL(KIND=r8) :: forfrac(ncol,nlay)
            REAL(KIND=r8) :: ref_forfrac(ncol,nlay)
            INTEGER :: indminor(ncol,nlay)
            INTEGER :: ref_indminor(ncol,nlay)
            REAL(KIND=r8) :: minorfrac(ncol,nlay)
            REAL(KIND=r8) :: ref_minorfrac(ncol,nlay)
            REAL(KIND=r8) :: scaleminor(ncol,nlay)
            REAL(KIND=r8) :: ref_scaleminor(ncol,nlay)
            REAL(KIND=r8) :: scaleminorn2(ncol,nlay)
            REAL(KIND=r8) :: ref_scaleminorn2(ncol,nlay)
            REAL(KIND=r8) :: fac01(ncol,nlay)
            REAL(KIND=r8) :: ref_fac01(ncol,nlay)
            REAL(KIND=r8) :: fac10(ncol,nlay)
            REAL(KIND=r8) :: ref_fac10(ncol,nlay)
            REAL(KIND=r8) :: fac11(ncol,nlay)
            REAL(KIND=r8) :: ref_fac11(ncol,nlay)
            REAL(KIND=r8) :: fac00(ncol,nlay)
            REAL(KIND=r8) :: ref_fac00(ncol,nlay) !
            REAL(KIND=r8) :: rat_o3co2_1(ncol,nlay)
            REAL(KIND=r8) :: ref_rat_o3co2_1(ncol,nlay)
            REAL(KIND=r8) :: rat_o3co2(ncol,nlay)
            REAL(KIND=r8) :: ref_rat_o3co2(ncol,nlay)
            REAL(KIND=r8) :: rat_h2och4(ncol,nlay)
            REAL(KIND=r8) :: ref_rat_h2och4(ncol,nlay)
            REAL(KIND=r8) :: rat_h2oo3(ncol,nlay)
            REAL(KIND=r8) :: ref_rat_h2oo3(ncol,nlay)
            REAL(KIND=r8) :: rat_h2och4_1(ncol,nlay)
            REAL(KIND=r8) :: ref_rat_h2och4_1(ncol,nlay)
            REAL(KIND=r8) :: rat_h2oo3_1(ncol,nlay)
            REAL(KIND=r8) :: ref_rat_h2oo3_1(ncol,nlay)
            REAL(KIND=r8) :: rat_h2oco2(ncol,nlay)
            REAL(KIND=r8) :: ref_rat_h2oco2(ncol,nlay)
            REAL(KIND=r8) :: rat_n2oco2(ncol,nlay)
            REAL(KIND=r8) :: ref_rat_n2oco2(ncol,nlay)
            REAL(KIND=r8) :: rat_h2on2o(ncol,nlay)
            REAL(KIND=r8) :: ref_rat_h2on2o(ncol,nlay)
            REAL(KIND=r8) :: rat_n2oco2_1(ncol,nlay)
            REAL(KIND=r8) :: ref_rat_n2oco2_1(ncol,nlay)
            REAL(KIND=r8) :: rat_h2oco2_1(ncol,nlay)
            REAL(KIND=r8) :: ref_rat_h2oco2_1(ncol,nlay)
            REAL(KIND=r8) :: rat_h2on2o_1(ncol,nlay)
            REAL(KIND=r8) :: ref_rat_h2on2o_1(ncol,nlay) !
            ! Atmosphere/clouds - cldprop
            ! number of cloud spectral bands
            ! flag for cloud property method
            ! flag for ice cloud properties
            ! flag for liquid cloud properties
            ! Atmosphere/clouds - cldprmc [mcica]
            ! cloud fraction [mcica]
            ! cloud ice water path [mcica]
            ! cloud liquid water path [mcica]
            ! liquid particle size (microns)
            ! ice particle effective radius (microns)
            ! ice particle generalized effective size (microns)
            ! cloud optical depth [mcica]
            !      real(kind=r8) :: ssacmc(ngptlw,nlay)     ! cloud single scattering albedo [mcica]
            !   for future expansion
            !   (lw scattering not yet available)
            !      real(kind=r8) :: asmcmc(ngptlw,nlay)     ! cloud asymmetry parameter [mcica]
            !   for future expansion
            !   (lw scattering not yet available)
            ! Output
            ! upward longwave flux (w/m2)
            ! downward longwave flux (w/m2)
            ! upward longwave flux spectral (w/m2)
            ! downward longwave flux spectral (w/m2)
            ! net longwave flux (w/m2)
            ! longwave heating rate (k/day)
            ! clear sky upward longwave flux (w/m2)
            ! clear sky downward longwave flux (w/m2)
            ! clear sky net longwave flux (w/m2)
            ! clear sky longwave heating rate (k/day)
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
            tolerance = 1.E-14
            CALL kgen_init_check(check_status, tolerance)
            READ(UNIT=kgen_unit) istart
            READ(UNIT=kgen_unit) pavel
            READ(UNIT=kgen_unit) tavel
            READ(UNIT=kgen_unit) tz
            READ(UNIT=kgen_unit) tbound
            READ(UNIT=kgen_unit) coldry
            READ(UNIT=kgen_unit) wbrodl
            READ(UNIT=kgen_unit) wkl
            READ(UNIT=kgen_unit) semiss
            READ(UNIT=kgen_unit) laytrop
            READ(UNIT=kgen_unit) jp
            READ(UNIT=kgen_unit) jt
            READ(UNIT=kgen_unit) jt1
            READ(UNIT=kgen_unit) planklay
            READ(UNIT=kgen_unit) planklev
            READ(UNIT=kgen_unit) plankbnd
            READ(UNIT=kgen_unit) colh2o
            READ(UNIT=kgen_unit) colco2
            READ(UNIT=kgen_unit) colo3
            READ(UNIT=kgen_unit) coln2o
            READ(UNIT=kgen_unit) colco
            READ(UNIT=kgen_unit) colch4
            READ(UNIT=kgen_unit) colo2
            READ(UNIT=kgen_unit) colbrd
            READ(UNIT=kgen_unit) indself
            READ(UNIT=kgen_unit) indfor
            READ(UNIT=kgen_unit) selffac
            READ(UNIT=kgen_unit) selffrac
            READ(UNIT=kgen_unit) forfac
            READ(UNIT=kgen_unit) forfrac
            READ(UNIT=kgen_unit) indminor
            READ(UNIT=kgen_unit) minorfrac
            READ(UNIT=kgen_unit) scaleminor
            READ(UNIT=kgen_unit) scaleminorn2
            READ(UNIT=kgen_unit) fac01
            READ(UNIT=kgen_unit) fac10
            READ(UNIT=kgen_unit) fac11
            READ(UNIT=kgen_unit) fac00
            READ(UNIT=kgen_unit) rat_o3co2_1
            READ(UNIT=kgen_unit) rat_o3co2
            READ(UNIT=kgen_unit) rat_h2och4
            READ(UNIT=kgen_unit) rat_h2oo3
            READ(UNIT=kgen_unit) rat_h2och4_1
            READ(UNIT=kgen_unit) rat_h2oo3_1
            READ(UNIT=kgen_unit) rat_h2oco2
            READ(UNIT=kgen_unit) rat_n2oco2
            READ(UNIT=kgen_unit) rat_h2on2o
            READ(UNIT=kgen_unit) rat_n2oco2_1
            READ(UNIT=kgen_unit) rat_h2oco2_1
            READ(UNIT=kgen_unit) rat_h2on2o_1

            READ(UNIT=kgen_unit) ref_laytrop
            READ(UNIT=kgen_unit) ref_jp
            READ(UNIT=kgen_unit) ref_jt
            READ(UNIT=kgen_unit) ref_jt1
            READ(UNIT=kgen_unit) ref_planklay
            READ(UNIT=kgen_unit) ref_planklev
            READ(UNIT=kgen_unit) ref_plankbnd
            READ(UNIT=kgen_unit) ref_colh2o
            READ(UNIT=kgen_unit) ref_colco2
            READ(UNIT=kgen_unit) ref_colo3
            READ(UNIT=kgen_unit) ref_coln2o
            READ(UNIT=kgen_unit) ref_colco
            READ(UNIT=kgen_unit) ref_colch4
            READ(UNIT=kgen_unit) ref_colo2
            READ(UNIT=kgen_unit) ref_colbrd
            READ(UNIT=kgen_unit) ref_indself
            READ(UNIT=kgen_unit) ref_indfor
            READ(UNIT=kgen_unit) ref_selffac
            READ(UNIT=kgen_unit) ref_selffrac
            READ(UNIT=kgen_unit) ref_forfac
            READ(UNIT=kgen_unit) ref_forfrac
            READ(UNIT=kgen_unit) ref_indminor
            READ(UNIT=kgen_unit) ref_minorfrac
            READ(UNIT=kgen_unit) ref_scaleminor
            READ(UNIT=kgen_unit) ref_scaleminorn2
            READ(UNIT=kgen_unit) ref_fac01
            READ(UNIT=kgen_unit) ref_fac10
            READ(UNIT=kgen_unit) ref_fac11
            READ(UNIT=kgen_unit) ref_fac00
            READ(UNIT=kgen_unit) ref_rat_o3co2_1
            READ(UNIT=kgen_unit) ref_rat_o3co2
            READ(UNIT=kgen_unit) ref_rat_h2och4
            READ(UNIT=kgen_unit) ref_rat_h2oo3
            READ(UNIT=kgen_unit) ref_rat_h2och4_1
            READ(UNIT=kgen_unit) ref_rat_h2oo3_1
            READ(UNIT=kgen_unit) ref_rat_h2oco2
            READ(UNIT=kgen_unit) ref_rat_n2oco2
            READ(UNIT=kgen_unit) ref_rat_h2on2o
            READ(UNIT=kgen_unit) ref_rat_n2oco2_1
            READ(UNIT=kgen_unit) ref_rat_h2oco2_1
            READ(UNIT=kgen_unit) ref_rat_h2on2o_1


            ! call to kernel
      call setcoef(ncol,nlay, istart, pavel, tavel, tz, tbound, semiss, &
           coldry, wkl, wbrodl, &
           laytrop, jp, jt, jt1, planklay, planklev, plankbnd, &
           colh2o, colco2, colo3, coln2o, colco, colch4, colo2, &
           colbrd, fac00, fac01, fac10, fac11, &
           rat_h2oco2, rat_h2oco2_1, rat_h2oo3, rat_h2oo3_1, &
           rat_h2on2o, rat_h2on2o_1, rat_h2och4, rat_h2och4_1, &
           rat_n2oco2, rat_n2oco2_1, rat_o3co2, rat_o3co2_1, &
           selffac, selffrac, indself, forfac, forfrac, indfor, &
           minorfrac, scaleminor, scaleminorn2, indminor)
            ! kernel verification for output variables
            CALL kgen_verify_integer_4_dim1( "laytrop", check_status, laytrop, ref_laytrop)
            CALL kgen_verify_integer_4_dim2( "jp", check_status, jp, ref_jp)
            CALL kgen_verify_integer_4_dim2( "jt", check_status, jt, ref_jt)
            CALL kgen_verify_integer_4_dim2( "jt1", check_status, jt1, ref_jt1)
            CALL kgen_verify_real_r8_dim3( "planklay", check_status, planklay, ref_planklay)
            CALL kgen_verify_real_r8_dim3( "planklev", check_status, planklev, ref_planklev)
            CALL kgen_verify_real_r8_dim2( "plankbnd", check_status, plankbnd, ref_plankbnd)
            CALL kgen_verify_real_r8_dim2( "colh2o", check_status, colh2o, ref_colh2o)
            CALL kgen_verify_real_r8_dim2( "colco2", check_status, colco2, ref_colco2)
            CALL kgen_verify_real_r8_dim2( "colo3", check_status, colo3, ref_colo3)
            CALL kgen_verify_real_r8_dim2( "coln2o", check_status, coln2o, ref_coln2o)
            CALL kgen_verify_real_r8_dim2( "colco", check_status, colco, ref_colco)
            CALL kgen_verify_real_r8_dim2( "colch4", check_status, colch4, ref_colch4)
            CALL kgen_verify_real_r8_dim2( "colo2", check_status, colo2, ref_colo2)
            CALL kgen_verify_real_r8_dim2( "colbrd", check_status, colbrd, ref_colbrd)
            CALL kgen_verify_integer_4_dim2( "indself", check_status, indself, ref_indself)
            CALL kgen_verify_integer_4_dim2( "indfor", check_status, indfor, ref_indfor)
            CALL kgen_verify_real_r8_dim2( "selffac", check_status, selffac, ref_selffac)
            CALL kgen_verify_real_r8_dim2( "selffrac", check_status, selffrac, ref_selffrac)
            CALL kgen_verify_real_r8_dim2( "forfac", check_status, forfac, ref_forfac)
            CALL kgen_verify_real_r8_dim2( "forfrac", check_status, forfrac, ref_forfrac)
            CALL kgen_verify_integer_4_dim2( "indminor", check_status, indminor, ref_indminor)
            CALL kgen_verify_real_r8_dim2( "minorfrac", check_status, minorfrac, ref_minorfrac)
            CALL kgen_verify_real_r8_dim2( "scaleminor", check_status, scaleminor, ref_scaleminor)
            CALL kgen_verify_real_r8_dim2( "scaleminorn2", check_status, scaleminorn2, ref_scaleminorn2)
            CALL kgen_verify_real_r8_dim2( "fac01", check_status, fac01, ref_fac01)
            CALL kgen_verify_real_r8_dim2( "fac10", check_status, fac10, ref_fac10)
            CALL kgen_verify_real_r8_dim2( "fac11", check_status, fac11, ref_fac11)
            CALL kgen_verify_real_r8_dim2( "fac00", check_status, fac00, ref_fac00)
            CALL kgen_verify_real_r8_dim2( "rat_o3co2_1", check_status, rat_o3co2_1, ref_rat_o3co2_1)
            CALL kgen_verify_real_r8_dim2( "rat_o3co2", check_status, rat_o3co2, ref_rat_o3co2)
            CALL kgen_verify_real_r8_dim2( "rat_h2och4", check_status, rat_h2och4, ref_rat_h2och4)
            CALL kgen_verify_real_r8_dim2( "rat_h2oo3", check_status, rat_h2oo3, ref_rat_h2oo3)
            CALL kgen_verify_real_r8_dim2( "rat_h2och4_1", check_status, rat_h2och4_1, ref_rat_h2och4_1)
            CALL kgen_verify_real_r8_dim2( "rat_h2oo3_1", check_status, rat_h2oo3_1, ref_rat_h2oo3_1)
            CALL kgen_verify_real_r8_dim2( "rat_h2oco2", check_status, rat_h2oco2, ref_rat_h2oco2)
            CALL kgen_verify_real_r8_dim2( "rat_n2oco2", check_status, rat_n2oco2, ref_rat_n2oco2)
            CALL kgen_verify_real_r8_dim2( "rat_h2on2o", check_status, rat_h2on2o, ref_rat_h2on2o)
            CALL kgen_verify_real_r8_dim2( "rat_n2oco2_1", check_status, rat_n2oco2_1, ref_rat_n2oco2_1)
            CALL kgen_verify_real_r8_dim2( "rat_h2oco2_1", check_status, rat_h2oco2_1, ref_rat_h2oco2_1)
            CALL kgen_verify_real_r8_dim2( "rat_h2on2o_1", check_status, rat_h2on2o_1, ref_rat_h2on2o_1)
            CALL kgen_print_check("setcoef", check_status)
            CALL system_clock(start_clock, rate_clock)
            DO kgen_intvar=1,10
                CALL setcoef(ncol, nlay, istart, pavel, tavel, tz, tbound, semiss, coldry, wkl, wbrodl, laytrop, &
jp, jt, jt1, planklay, planklev, plankbnd, colh2o, colco2, colo3, coln2o, colco, colch4, colo2, colbrd, fac00, &
fac01, fac10, fac11, rat_h2oco2, rat_h2oco2_1, rat_h2oo3, rat_h2oo3_1, rat_h2on2o, rat_h2on2o_1, rat_h2och4, &
rat_h2och4_1, rat_n2oco2, rat_n2oco2_1, rat_o3co2, rat_o3co2_1, selffac, selffrac, indself, forfac, forfrac, &
indfor, minorfrac, scaleminor, scaleminorn2, indminor)
            END DO
            CALL system_clock(stop_clock, rate_clock)
            WRITE(*,*)
            PRINT *, "Elapsed time (sec): ", (stop_clock - start_clock)/REAL(rate_clock*10)
            ! Call the radiative transfer routine.
            ! Either routine can be called to do clear sky calculation.  If clouds
            ! are present, then select routine based on cloud overlap assumption
            ! to be used.  Clear sky calculation is done simultaneously.
            ! For McICA, RTRNMC is called for clear and cloudy calculations.
            !  Transfer up and down fluxes and heating rate to output arrays.
            !  Vertical indexing goes from bottom to top
        CONTAINS

        ! write subroutines
            SUBROUTINE kgen_read_integer_4_dim1(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                integer(KIND=4), INTENT(OUT), ALLOCATABLE, DIMENSION(:) :: var
                LOGICAL :: is_true
                INTEGER :: idx1
                INTEGER, DIMENSION(2,1) :: kgen_bound

                READ(UNIT = kgen_unit) is_true

                IF ( is_true ) THEN
                    READ(UNIT = kgen_unit) kgen_bound(1, 1)
                    READ(UNIT = kgen_unit) kgen_bound(2, 1)
                    ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1))
                    READ(UNIT = kgen_unit) var
                    IF ( PRESENT(printvar) ) THEN
                        PRINT *, "** " // printvar // " **", var
                    END IF
                END IF
            END SUBROUTINE kgen_read_integer_4_dim1

            SUBROUTINE kgen_read_integer_4_dim2(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                integer(KIND=4), INTENT(OUT), ALLOCATABLE, DIMENSION(:,:) :: var
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
            END SUBROUTINE kgen_read_integer_4_dim2

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


        ! verify subroutines
            SUBROUTINE kgen_verify_integer_4_dim1( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                integer, intent(in), DIMENSION(:) :: var, ref_var
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
                    if(check_status%verboseLevel > 0) then
                        WRITE(*,*)
                        WRITE(*,*) trim(adjustl(varname)), " is NOT IDENTICAL."
                        WRITE(*,*) count( var /= ref_var), " of ", size( var ), " elements are different."
                    end if
                
                    check_status%numFatal = check_status%numFatal+1
                END IF
            END SUBROUTINE kgen_verify_integer_4_dim1

            SUBROUTINE kgen_verify_integer_4_dim2( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                integer, intent(in), DIMENSION(:,:) :: var, ref_var
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
                    if(check_status%verboseLevel > 0) then
                        WRITE(*,*)
                        WRITE(*,*) trim(adjustl(varname)), " is NOT IDENTICAL."
                        WRITE(*,*) count( var /= ref_var), " of ", size( var ), " elements are different."
                    end if
                
                    check_status%numFatal = check_status%numFatal+1
                END IF
            END SUBROUTINE kgen_verify_integer_4_dim2

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

        END SUBROUTINE rrtmg_lw
        !***************************************************************************

    END MODULE rrtmg_lw_rad
