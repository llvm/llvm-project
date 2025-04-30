
! KGEN-generated Fortran source file
!
! Filename    : rrtmg_sw_rad.f90
! Generated at: 2015-07-27 00:47:03
! KGEN version: 0.4.13



    MODULE rrtmg_sw_rad
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
        ! *                             RRTMG_SW                                     *
        ! *                                                                          *
        ! *                                                                          *
        ! *                                                                          *
        ! *                 a rapid radiative transfer model                         *
        ! *                  for the solar spectral region                           *
        ! *           for application to general circulation models                  *
        ! *                                                                          *
        ! *                                                                          *
        ! *           Atmospheric and Environmental Research, Inc.                   *
        ! *                       131 Hartwell Avenue                                *
        ! *                       Lexington, MA 02421                                *
        ! *                                                                          *
        ! *                                                                          *
        ! *                          Eli J. Mlawer                                   *
        ! *                       Jennifer S. Delamere                               *
        ! *                        Michael J. Iacono                                 *
        ! *                        Shepard A. Clough                                 *
        ! *                                                                          *
        ! *                                                                          *
        ! *                                                                          *
        ! *                                                                          *
        ! *                                                                          *
        ! *                                                                          *
        ! *                      email:  miacono@aer.com                             *
        ! *                      email:  emlawer@aer.com                             *
        ! *                      email:  jdelamer@aer.com                            *
        ! *                                                                          *
        ! *       The authors wish to acknowledge the contributions of the           *
        ! *       following people:  Steven J. Taubman, Patrick D. Brown,            *
        ! *       Ronald E. Farren, Luke Chen, Robert Bergstrom.                     *
        ! *                                                                          *
        ! ****************************************************************************
        ! --------- Modules ---------
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !      use parkind, only : jpim, jprb
        ! Move call to rrtmg_sw_ini and following use association to
        ! GCM initialization area
        !      use rrtmg_sw_init, only: rrtmg_sw_ini
        USE rrtmg_sw_setcoef, ONLY: setcoef_sw
        IMPLICIT NONE
        ! public interfaces/functions/subroutines
        !      public :: rrtmg_sw, inatm_sw, earth_sun
        PUBLIC rrtmg_sw
        !------------------------------------------------------------------
        CONTAINS

        ! write subroutines
        ! No subroutines
        ! No module extern variables
        !------------------------------------------------------------------
        !------------------------------------------------------------------
        ! Public subroutines
        !------------------------------------------------------------------

        SUBROUTINE rrtmg_sw(ncol, nlay, kgen_unit)
                USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
            ! ------- Description -------
            ! This program is the driver for RRTMG_SW, the AER SW radiation model for
            !  application to GCMs, that has been adapted from RRTM_SW for improved
            !  efficiency and to provide fractional cloudiness and cloud overlap
            !  capability using McICA.
            !
            ! Note: The call to RRTMG_SW_INI should be moved to the GCM initialization
            !  area, since this has to be called only once.
            !
            ! This routine
            !    b) calls INATM_SW to read in the atmospheric profile;
            !       all layering in RRTMG is ordered from surface to toa.
            !    c) calls CLDPRMC_SW to set cloud optical depth for McICA based
            !       on input cloud properties
            !    d) calls SETCOEF_SW to calculate various quantities needed for
            !       the radiative transfer algorithm
            !    e) calls SPCVMC to call the two-stream model that in turn
            !       calls TAUMOL to calculate gaseous optical depths for each
            !       of the 16 spectral bands and to perform the radiative transfer
            !       using McICA, the Monte-Carlo Independent Column Approximation,
            !       to represent sub-grid scale cloud variability
            !    f) passes the calculated fluxes and cooling rates back to GCM
            !
            ! Two modes of operation are possible:
            !     The mode is chosen by using either rrtmg_sw.nomcica.f90 (to not use
            !     McICA) or rrtmg_sw.f90 (to use McICA) to interface with a GCM.
            !
            !    1) Standard, single forward model calculation (imca = 0); this is
            !       valid only for clear sky or fully overcast clouds
            !    2) Monte Carlo Independent Column Approximation (McICA, Pincus et al.,
            !       JC, 2003) method is applied to the forward model calculation (imca = 1)
            !       This method is valid for clear sky or partial cloud conditions.
            !
            ! This call to RRTMG_SW must be preceeded by a call to the module
            !     mcica_subcol_gen_sw.f90 to run the McICA sub-column cloud generator,
            !     which will provide the cloud physical or cloud optical properties
            !     on the RRTMG quadrature point (ngptsw) dimension.
            !
            ! Two methods of cloud property input are possible:
            !     Cloud properties can be input in one of two ways (controlled by input
            !     flags inflag, iceflag and liqflag; see text file rrtmg_sw_instructions
            !     and subroutine rrtmg_sw_cldprop.f90 for further details):
            !
            !    1) Input cloud fraction, cloud optical depth, single scattering albedo
            !       and asymmetry parameter directly (inflgsw = 0)
            !    2) Input cloud fraction and cloud physical properties: ice fracion,
            !       ice and liquid particle sizes (inflgsw = 1 or 2);
            !       cloud optical properties are calculated by cldprop or cldprmc based
            !       on input settings of iceflgsw and liqflgsw
            !
            ! Two methods of aerosol property input are possible:
            !     Aerosol properties can be input in one of two ways (controlled by input
            !     flag iaer, see text file rrtmg_sw_instructions for further details):
            !
            !    1) Input aerosol optical depth, single scattering albedo and asymmetry
            !       parameter directly by layer and spectral band (iaer=10)
            !    2) Input aerosol optical depth and 0.55 micron directly by layer and use
            !       one or more of six ECMWF aerosol types (iaer=6)
            !
            !
            ! ------- Modifications -------
            !
            ! This version of RRTMG_SW has been modified from RRTM_SW to use a reduced
            ! set of g-point intervals and a two-stream model for application to GCMs.
            !
            !-- Original version (derived from RRTM_SW)
            !     2002: AER. Inc.
            !-- Conversion to F90 formatting; addition of 2-stream radiative transfer
            !     Feb 2003: J.-J. Morcrette, ECMWF
            !-- Additional modifications for GCM application
            !     Aug 2003: M. J. Iacono, AER Inc.
            !-- Total number of g-points reduced from 224 to 112.  Original
            !   set of 224 can be restored by exchanging code in module parrrsw.f90
            !   and in file rrtmg_sw_init.f90.
            !     Apr 2004: M. J. Iacono, AER, Inc.
            !-- Modifications to include output for direct and diffuse
            !   downward fluxes.  There are output as "true" fluxes without
            !   any delta scaling applied.  Code can be commented to exclude
            !   this calculation in source file rrtmg_sw_spcvrt.f90.
            !     Jan 2005: E. J. Mlawer, M. J. Iacono, AER, Inc.
            !-- Revised to add McICA capability.
            !     Nov 2005: M. J. Iacono, AER, Inc.
            !-- Reformatted for consistency with rrtmg_lw.
            !     Feb 2007: M. J. Iacono, AER, Inc.
            !-- Modifications to formatting to use assumed-shape arrays.
            !     Aug 2007: M. J. Iacono, AER, Inc.
            !-- Modified to output direct and diffuse fluxes either with or without
            !   delta scaling based on setting of idelm flag
            !     Dec 2008: M. J. Iacono, AER, Inc.
            ! --------- Modules ---------
            USE parrrsw, ONLY: mxmol
            ! ------- Declarations
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
            ! UV/vis surface albedo direct rad
            !    Dimensions: (ncol)
            ! Near-IR surface albedo direct rad
            !    Dimensions: (ncol)
            ! UV/vis surface albedo: diffuse rad
            !    Dimensions: (ncol)
            ! Near-IR surface albedo: diffuse rad
            !    Dimensions: (ncol)
            ! Day of the year (used to get Earth/Sun
            !  distance if adjflx not provided)
            ! Flux adjustment for Earth/Sun distance
            ! Cosine of solar zenith angle
            !    Dimensions: (ncol)
            ! Solar constant (Wm-2) scaling per band
            ! Flag for cloud optical properties
            ! Flag for ice particle specification
            ! Flag for liquid droplet specification
            ! Cloud fraction
            !    Dimensions: (ngptsw,ncol,nlay)
            ! Cloud optical depth
            !    Dimensions: (ngptsw,ncol,nlay)
            ! Cloud single scattering albedo
            !    Dimensions: (ngptsw,ncol,nlay)
            ! Cloud asymmetry parameter
            !    Dimensions: (ngptsw,ncol,nlay)
            ! Cloud forward scattering parameter
            !    Dimensions: (ngptsw,ncol,nlay)
            ! Cloud ice water path (g/m2)
            !    Dimensions: (ngptsw,ncol,nlay)
            ! Cloud liquid water path (g/m2)
            !    Dimensions: (ngptsw,ncol,nlay)
            ! Cloud ice effective radius (microns)
            !    Dimensions: (ncol,nlay)
            ! Cloud water drop effective radius (microns)
            !    Dimensions: (ncol,nlay)
            ! Aerosol optical depth (iaer=10 only)
            !    Dimensions: (ncol,nlay,nbndsw)
            ! (non-delta scaled)
            ! Aerosol single scattering albedo (iaer=10 only)
            !    Dimensions: (ncol,nlay,nbndsw)
            ! (non-delta scaled)
            ! Aerosol asymmetry parameter (iaer=10 only)
            !    Dimensions: (ncol,nlay,nbndsw)
            ! (non-delta scaled)
            !      real(kind=r8), intent(in) :: ecaer(:,:,:)         ! Aerosol optical depth at 0.55 micron (iaer=6 only)
            !    Dimensions: (ncol,nlay,naerec)
            ! (non-delta scaled)
            ! ----- Output -----
            ! Total sky shortwave upward flux (W/m2)
            !    Dimensions: (ncol,nlay+1)
            ! Total sky shortwave downward flux (W/m2)
            !    Dimensions: (ncol,nlay+1)
            ! Total sky shortwave radiative heating rate (K/d)
            !    Dimensions: (ncol,nlay)
            ! Clear sky shortwave upward flux (W/m2)
            !    Dimensions: (ncol,nlay+1)
            ! Clear sky shortwave downward flux (W/m2)
            !    Dimensions: (ncol,nlay+1)
            ! Clear sky shortwave radiative heating rate (K/d)
            !    Dimensions: (ncol,nlay)
            ! Direct downward shortwave flux, UV/vis
            ! Diffuse downward shortwave flux, UV/vis
            ! Direct downward shortwave flux, near-IR
            ! Diffuse downward shortwave flux, near-IR
            ! Net shortwave flux, near-IR
            ! Net clear sky shortwave flux, near-IR
            ! shortwave spectral flux up
            ! shortwave spectral flux down
            ! ----- Local -----
            ! Control
            ! beginning band of calculation
            ! ending band of calculation
            ! cldprop/cldprmc use flag
            ! output option flag (inactive)
            ! aerosol option flag
            ! delta-m scaling flag
            ! [0 = direct and diffuse fluxes are unscaled]
            ! [1 = direct and diffuse fluxes are scaled]
            ! (total downward fluxes are always delta scaled)
            ! instrumental cosine response flag (inactive)
            ! column loop index
            ! layer loop index                       ! jk
            ! band loop index                        ! jsw
            ! indices
            ! layer loop index
            ! value for changing mcica permute seed
            ! flag for mcica [0=off, 1=on]
            ! epsilon
            ! flux to heating conversion ratio
            ! Atmosphere
            REAL(KIND=r8) :: pavel(ncol,nlay) ! layer pressures (mb)
            REAL(KIND=r8) :: tavel(ncol,nlay) ! layer temperatures (K)
            REAL(KIND=r8) :: pz(ncol,0:nlay) ! level (interface) pressures (hPa, mb)
            REAL(KIND=r8) :: tz(ncol,0:nlay) ! level (interface) temperatures (K)
            REAL(KIND=r8) :: tbound(ncol) ! surface temperature (K)
            ! layer pressure thickness (hPa, mb)
            REAL(KIND=r8) :: coldry(ncol,nlay) ! dry air column amount
            REAL(KIND=r8) :: wkl(ncol,mxmol,nlay) ! molecular amounts (mol/cm-2)
            !      real(kind=r8) :: earth_sun               ! function for Earth/Sun distance factor
            ! Cosine of solar zenith angle
            ! adjustment for current Earth/Sun distance
            !      real(kind=r8) :: solvar(jpband)           ! solar constant scaling factor from rrtmg_sw
            !  default value of 1368.22 Wm-2 at 1 AU
            ! surface albedo, direct          ! zalbp
            ! surface albedo, diffuse         ! zalbd
            ! Aerosol optical depth
            ! Aerosol single scattering albedo
            ! Aerosol asymmetry parameter
            ! Atmosphere - setcoef
            INTEGER :: laytrop(ncol)
            INTEGER :: ref_laytrop(ncol) ! tropopause layer index
            INTEGER :: layswtch(ncol)
            INTEGER :: ref_layswtch(ncol) !
            INTEGER :: laylow(ncol)
            INTEGER :: ref_laylow(ncol) !
            INTEGER :: jp(ncol,nlay)
            INTEGER :: ref_jp(ncol,nlay) !
            INTEGER :: jt(ncol,nlay)
            INTEGER :: ref_jt(ncol,nlay) !
            INTEGER :: jt1(ncol,nlay)
            INTEGER :: ref_jt1(ncol,nlay) !
            REAL(KIND=r8) :: colh2o(ncol,nlay)
            REAL(KIND=r8) :: ref_colh2o(ncol,nlay) ! column amount (h2o)
            REAL(KIND=r8) :: colco2(ncol,nlay)
            REAL(KIND=r8) :: ref_colco2(ncol,nlay) ! column amount (co2)
            REAL(KIND=r8) :: colo3(ncol,nlay)
            REAL(KIND=r8) :: ref_colo3(ncol,nlay) ! column amount (o3)
            REAL(KIND=r8) :: coln2o(ncol,nlay)
            REAL(KIND=r8) :: ref_coln2o(ncol,nlay) ! column amount (n2o)
            REAL(KIND=r8) :: colch4(ncol,nlay)
            REAL(KIND=r8) :: ref_colch4(ncol,nlay) ! column amount (ch4)
            REAL(KIND=r8) :: colo2(ncol,nlay)
            REAL(KIND=r8) :: ref_colo2(ncol,nlay) ! column amount (o2)
            REAL(KIND=r8) :: colmol(ncol,nlay)
            REAL(KIND=r8) :: ref_colmol(ncol,nlay) ! column amount
            REAL(KIND=r8) :: co2mult(ncol,nlay)
            REAL(KIND=r8) :: ref_co2mult(ncol,nlay) ! column amount
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
            REAL(KIND=r8) :: fac00(ncol,nlay)
            REAL(KIND=r8) :: ref_fac00(ncol,nlay)
            REAL(KIND=r8) :: fac01(ncol,nlay)
            REAL(KIND=r8) :: ref_fac01(ncol,nlay)
            REAL(KIND=r8) :: fac11(ncol,nlay)
            REAL(KIND=r8) :: ref_fac11(ncol,nlay)
            REAL(KIND=r8) :: fac10(ncol,nlay)
            REAL(KIND=r8) :: ref_fac10(ncol,nlay) !
            ! Atmosphere/clouds - cldprop
            ! number of cloud spectral bands
            ! flag for cloud property method
            ! flag for ice cloud properties
            ! flag for liquid cloud properties
            !      real(kind=r8) :: cldfrac(nlay)            ! layer cloud fraction
            !      real(kind=r8) :: tauc(nlay)               ! cloud optical depth (non-delta scaled)
            !      real(kind=r8) :: ssac(nlay)               ! cloud single scattering albedo (non-delta scaled)
            !      real(kind=r8) :: asmc(nlay)               ! cloud asymmetry parameter (non-delta scaled)
            !      real(kind=r8) :: ciwp(nlay)               ! cloud ice water path
            !      real(kind=r8) :: clwp(nlay)               ! cloud liquid water path
            !      real(kind=r8) :: rei(nlay)                ! cloud ice particle size
            !      real(kind=r8) :: rel(nlay)                ! cloud liquid particle size
            !      real(kind=r8) :: taucloud(nlay,jpband)    ! cloud optical depth
            !      real(kind=r8) :: taucldorig(nlay,jpband)  ! cloud optical depth (non-delta scaled)
            !      real(kind=r8) :: ssacloud(nlay,jpband)    ! cloud single scattering albedo
            !      real(kind=r8) :: asmcloud(nlay,jpband)    ! cloud asymmetry parameter
            ! Atmosphere/clouds - cldprmc [mcica]
            ! cloud fraction [mcica]
            ! cloud ice water path [mcica]
            ! cloud liquid water path [mcica]
            ! liquid particle size (microns)
            ! ice particle effective radius (microns)
            ! ice particle generalized effective size (microns)
            ! cloud optical depth [mcica]
            ! unscaled cloud optical depth [mcica]
            ! cloud single scattering albedo [mcica]
            ! cloud asymmetry parameter [mcica]
            ! cloud forward scattering fraction [mcica]
            ! Atmosphere/clouds/aerosol - spcvrt,spcvmc
            ! cloud optical depth
            ! unscaled cloud optical depth
            ! cloud asymmetry parameter
            !  (first moment of phase function)
            ! cloud single scattering albedo
            ! total aerosol optical depth
            ! total aerosol asymmetry parameter
            ! total aerosol single scattering albedo
            ! cloud fraction [mcica]
            ! cloud optical depth [mcica]
            ! unscaled cloud optical depth [mcica]
            ! cloud asymmetry parameter [mcica]
            ! cloud single scattering albedo [mcica]
            ! temporary upward shortwave flux (w/m2)
            ! temporary downward shortwave flux (w/m2)
            ! temporary clear sky upward shortwave flux (w/m2)
            ! temporary clear sky downward shortwave flux (w/m2)
            ! temporary downward direct shortwave flux (w/m2)
            ! temporary clear sky downward direct shortwave flux (w/m2)
            ! temporary UV downward shortwave flux (w/m2)
            ! temporary clear sky UV downward shortwave flux (w/m2)
            ! temporary UV downward direct shortwave flux (w/m2)
            ! temporary clear sky UV downward direct shortwave flux (w/m2)
            ! temporary near-IR downward shortwave flux (w/m2)
            ! temporary clear sky near-IR downward shortwave flux (w/m2)
            ! temporary near-IR downward direct shortwave flux (w/m2)
            ! temporary clear sky near-IR downward direct shortwave flux (w/m2)
            ! Added for near-IR flux diagnostic
            ! temporary near-IR downward shortwave flux (w/m2)
            ! temporary clear sky near-IR downward shortwave flux (w/m2)
            ! Optional output fields
            ! Total sky shortwave net flux (W/m2)
            ! Clear sky shortwave net flux (W/m2)
            ! Direct downward shortwave surface flux
            ! Diffuse downward shortwave surface flux
            ! Total sky downward shortwave flux, UV/vis
            ! Total sky downward shortwave flux, near-IR
            ! temporary upward shortwave flux spectral (w/m2)
            ! temporary downward shortwave flux spectral (w/m2)
            ! Output - inactive
            !      real(kind=r8) :: zuvfu(nlay+2)         ! temporary upward UV shortwave flux (w/m2)
            !      real(kind=r8) :: zuvfd(nlay+2)         ! temporary downward UV shortwave flux (w/m2)
            !      real(kind=r8) :: zuvcu(nlay+2)         ! temporary clear sky upward UV shortwave flux (w/m2)
            !      real(kind=r8) :: zuvcd(nlay+2)         ! temporary clear sky downward UV shortwave flux (w/m2)
            !      real(kind=r8) :: zvsfu(nlay+2)         ! temporary upward visible shortwave flux (w/m2)
            !      real(kind=r8) :: zvsfd(nlay+2)         ! temporary downward visible shortwave flux (w/m2)
            !      real(kind=r8) :: zvscu(nlay+2)         ! temporary clear sky upward visible shortwave flux (w/m2)
            !      real(kind=r8) :: zvscd(nlay+2)         ! temporary clear sky downward visible shortwave flux (w/m2)
            !      real(kind=r8) :: znifu(nlay+2)         ! temporary upward near-IR shortwave flux (w/m2)
            !      real(kind=r8) :: znifd(nlay+2)         ! temporary downward near-IR shortwave flux (w/m2)
            !      real(kind=r8) :: znicu(nlay+2)         ! temporary clear sky upward near-IR shortwave flux (w/m2)
            !      real(kind=r8) :: znicd(nlay+2)         ! temporary clear sky downward near-IR shortwave flux (w/m2)
            ! Initializations
            ! In a GCM with or without McICA, set nlon to the longitude dimension
            !
            ! Set imca to select calculation type:
            !  imca = 0, use standard forward model calculation (clear and overcast only)
            !  imca = 1, use McICA for Monte Carlo treatment of sub-grid cloud variability
            !            (clear, overcast or partial cloud conditions)
            ! *** This version uses McICA (imca = 1) ***
            ! Set icld to select of clear or cloud calculation and cloud
            ! overlap method (read by subroutine readprof from input file INPUT_RRTM):
            ! icld = 0, clear only
            ! icld = 1, with clouds using random cloud overlap (McICA only)
            ! icld = 2, with clouds using maximum/random cloud overlap (McICA only)
            ! icld = 3, with clouds using maximum cloud overlap (McICA only)
            ! Set iaer to select aerosol option
            ! iaer = 0, no aerosols
            ! iaer = 6, use six ECMWF aerosol types
            !           input aerosol optical depth at 0.55 microns for each aerosol type (ecaer)
            ! iaer = 10, input total aerosol optical depth, single scattering albedo
            !            and asymmetry parameter (tauaer, ssaaer, asmaer) directly
            ! Set idelm to select between delta-M scaled or unscaled output direct and diffuse fluxes
            ! NOTE: total downward fluxes are always delta scaled
            ! idelm = 0, output direct and diffuse flux components are not delta scaled
            !            (direct flux does not include forward scattering peak)
            ! idelm = 1, output direct and diffuse flux components are delta scaled (default)
            !            (direct flux includes part or most of forward scattering peak)
            ! Call model and data initialization, compute lookup tables, perform
            ! reduction of g-points from 224 to 112 for input absorption
            ! coefficient data and other arrays.
            !
            ! In a GCM this call should be placed in the model initialization
            ! area, since this has to be called only once.
            !      call rrtmg_sw_ini
            ! This is the main longitude/column loop in RRTMG.
            ! Modify to loop over all columns (nlon) or over daylight columns
            !JMD #define OLD_INATM_SW 1
            !  For cloudy atmosphere, use cldprop to set cloud optical properties based on
            !  input cloud physical properties.  Select method based on choices described
            !  in cldprop.  Cloud fraction, water path, liquid droplet and ice particle
            !  effective radius must be passed in cldprop.  Cloud fraction and cloud
            !  optical properties are transferred to rrtmg_sw arrays in cldprop.
            ! Calculate coefficients for the temperature and pressure dependence of the
            ! molecular absorption coefficients by interpolating data from stored
            tolerance = 1.E-14
            CALL kgen_init_check(check_status, tolerance)
            READ(UNIT=kgen_unit) pavel
            READ(UNIT=kgen_unit) tavel
            READ(UNIT=kgen_unit) pz
            READ(UNIT=kgen_unit) tz
            READ(UNIT=kgen_unit) tbound
            READ(UNIT=kgen_unit) coldry
            READ(UNIT=kgen_unit) wkl
            READ(UNIT=kgen_unit) laytrop
            READ(UNIT=kgen_unit) layswtch
            READ(UNIT=kgen_unit) laylow
            READ(UNIT=kgen_unit) jp
            READ(UNIT=kgen_unit) jt
            READ(UNIT=kgen_unit) jt1
            READ(UNIT=kgen_unit) colh2o
            READ(UNIT=kgen_unit) colco2
            READ(UNIT=kgen_unit) colo3
            READ(UNIT=kgen_unit) coln2o
            READ(UNIT=kgen_unit) colch4
            READ(UNIT=kgen_unit) colo2
            READ(UNIT=kgen_unit) colmol
            READ(UNIT=kgen_unit) co2mult
            READ(UNIT=kgen_unit) indself
            READ(UNIT=kgen_unit) indfor
            READ(UNIT=kgen_unit) selffac
            READ(UNIT=kgen_unit) selffrac
            READ(UNIT=kgen_unit) forfac
            READ(UNIT=kgen_unit) forfrac
            READ(UNIT=kgen_unit) fac00
            READ(UNIT=kgen_unit) fac01
            READ(UNIT=kgen_unit) fac11
            READ(UNIT=kgen_unit) fac10

            READ(UNIT=kgen_unit) ref_laytrop
            READ(UNIT=kgen_unit) ref_layswtch
            READ(UNIT=kgen_unit) ref_laylow
            READ(UNIT=kgen_unit) ref_jp
            READ(UNIT=kgen_unit) ref_jt
            READ(UNIT=kgen_unit) ref_jt1
            READ(UNIT=kgen_unit) ref_colh2o
            READ(UNIT=kgen_unit) ref_colco2
            READ(UNIT=kgen_unit) ref_colo3
            READ(UNIT=kgen_unit) ref_coln2o
            READ(UNIT=kgen_unit) ref_colch4
            READ(UNIT=kgen_unit) ref_colo2
            READ(UNIT=kgen_unit) ref_colmol
            READ(UNIT=kgen_unit) ref_co2mult
            READ(UNIT=kgen_unit) ref_indself
            READ(UNIT=kgen_unit) ref_indfor
            READ(UNIT=kgen_unit) ref_selffac
            READ(UNIT=kgen_unit) ref_selffrac
            READ(UNIT=kgen_unit) ref_forfac
            READ(UNIT=kgen_unit) ref_forfrac
            READ(UNIT=kgen_unit) ref_fac00
            READ(UNIT=kgen_unit) ref_fac01
            READ(UNIT=kgen_unit) ref_fac11
            READ(UNIT=kgen_unit) ref_fac10


            ! call to kernel
         call setcoef_sw(ncol,nlay, pavel, tavel, pz, tz, tbound, coldry, wkl, &
                         laytrop, layswtch, laylow, jp, jt, jt1, &
                         co2mult, colch4, colco2, colh2o, colmol, coln2o, &
                         colo2, colo3, fac00, fac01, fac10, fac11, &
                         selffac, selffrac, indself, forfac, forfrac, indfor)
            ! kernel verification for output variables
            CALL kgen_verify_integer_4_dim1( "laytrop", check_status, laytrop, ref_laytrop)
            CALL kgen_verify_integer_4_dim1( "layswtch", check_status, layswtch, ref_layswtch)
            CALL kgen_verify_integer_4_dim1( "laylow", check_status, laylow, ref_laylow)
            CALL kgen_verify_integer_4_dim2( "jp", check_status, jp, ref_jp)
            CALL kgen_verify_integer_4_dim2( "jt", check_status, jt, ref_jt)
            CALL kgen_verify_integer_4_dim2( "jt1", check_status, jt1, ref_jt1)
            CALL kgen_verify_real_r8_dim2( "colh2o", check_status, colh2o, ref_colh2o)
            CALL kgen_verify_real_r8_dim2( "colco2", check_status, colco2, ref_colco2)
            CALL kgen_verify_real_r8_dim2( "colo3", check_status, colo3, ref_colo3)
            CALL kgen_verify_real_r8_dim2( "coln2o", check_status, coln2o, ref_coln2o)
            CALL kgen_verify_real_r8_dim2( "colch4", check_status, colch4, ref_colch4)
            CALL kgen_verify_real_r8_dim2( "colo2", check_status, colo2, ref_colo2)
            CALL kgen_verify_real_r8_dim2( "colmol", check_status, colmol, ref_colmol)
            CALL kgen_verify_real_r8_dim2( "co2mult", check_status, co2mult, ref_co2mult)
            CALL kgen_verify_integer_4_dim2( "indself", check_status, indself, ref_indself)
            CALL kgen_verify_integer_4_dim2( "indfor", check_status, indfor, ref_indfor)
            CALL kgen_verify_real_r8_dim2( "selffac", check_status, selffac, ref_selffac)
            CALL kgen_verify_real_r8_dim2( "selffrac", check_status, selffrac, ref_selffrac)
            CALL kgen_verify_real_r8_dim2( "forfac", check_status, forfac, ref_forfac)
            CALL kgen_verify_real_r8_dim2( "forfrac", check_status, forfrac, ref_forfrac)
            CALL kgen_verify_real_r8_dim2( "fac00", check_status, fac00, ref_fac00)
            CALL kgen_verify_real_r8_dim2( "fac01", check_status, fac01, ref_fac01)
            CALL kgen_verify_real_r8_dim2( "fac11", check_status, fac11, ref_fac11)
            CALL kgen_verify_real_r8_dim2( "fac10", check_status, fac10, ref_fac10)
            CALL kgen_print_check("setcoef_sw", check_status)
            CALL system_clock(start_clock, rate_clock)
            DO kgen_intvar=1,10
                CALL setcoef_sw(ncol, nlay, pavel, tavel, pz, tz, tbound, coldry, wkl, laytrop, layswtch, laylow, &
jp, jt, jt1, co2mult, colch4, colco2, colh2o, colmol, coln2o, colo2, colo3, fac00, fac01, fac10, fac11, selffac, &
selffrac, indself, forfac, forfrac, indfor)
            END DO
            CALL system_clock(stop_clock, rate_clock)
            WRITE(*,*)
            PRINT *, "Elapsed time (sec): ", (stop_clock - start_clock)/REAL(rate_clock*10)
            !do iplon = 1, ncol         ! reference atmospheres.
            ! call setcoef_sw(nlay, pavel(iplon,:), tavel(iplon,:), pz(iplon,:), tz(iplon,:), tbound(iplon), coldry(iplon,:), wkl(
            ! iplon,:,:), &
            !                laytrop(iplon), layswtch(iplon), laylow(iplon), jp(iplon,:), jt(iplon,:), jt1(iplon,:), &
            !                co2mult(iplon,:), colch4(iplon,:), colco2(iplon,:), colh2o(iplon,:), colmol(iplon,:), coln2o(iplon,:)
            ! , &
            !                colo2(iplon,:), colo3(iplon,:), fac00(iplon,:), fac01(iplon,:), fac10(iplon,:), fac11(iplon,:), &
            !                selffac(iplon,:), selffrac(iplon,:), indself(iplon,:), forfac(iplon,:), forfrac(iplon,:), indfor(
            ! iplon,:))
            !end do
            ! Cosine of the solar zenith angle
            !  Prevent using value of zero; ideally, SW model is not called from host model when sun
            !  is below horizon
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

        END SUBROUTINE rrtmg_sw
        !*************************************************************************

        !***************************************************************************

    END MODULE rrtmg_sw_rad
