
! KGEN-generated Fortran source file
!
! Filename    : rrtmg_lw_rad.f90
! Generated at: 2015-07-26 18:45:57
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
        IMPLICIT NONE
        ! public interfaces/functions/subroutines
        PUBLIC inatm
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

        SUBROUTINE rrtmg_lw(ncol, nlay, icld, play, plev, tlay, tlev, tsfc, h2ovmr, o3vmr, co2vmr, ch4vmr, o2vmr, n2ovmr, &
        cfc11vmr, cfc12vmr, cfc22vmr, ccl4vmr, emis, inflglw, iceflglw, liqflglw, cldfmcl, ciwpmcl, clwpmcl, reicmcl, relqmcl, &
        taucmcl, tauaer, kgen_unit)
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
            USE parrrtm, ONLY: ngptlw
            USE parrrtm, ONLY: maxxsec
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
            INTEGER, intent(inout) :: icld ! Cloud overlap method
            !    0: Clear only
            !    1: Random
            !    2: Maximum/random
            !    3: Maximum
            REAL(KIND=r8), intent(in) :: play(:,:) ! Layer pressures (hPa, mb)
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(in) :: plev(:,:) ! Interface pressures (hPa, mb)
            !    Dimensions: (ncol,nlay+1)
            REAL(KIND=r8), intent(in) :: tlay(:,:) ! Layer temperatures (K)
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(in) :: tlev(:,:) ! Interface temperatures (K)
            !    Dimensions: (ncol,nlay+1)
            REAL(KIND=r8), intent(in) :: tsfc(:) ! Surface temperature (K)
            !    Dimensions: (ncol)
            REAL(KIND=r8), intent(in) :: h2ovmr(:,:) ! H2O volume mixing ratio
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(in) :: o3vmr(:,:) ! O3 volume mixing ratio
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(in) :: co2vmr(:,:) ! CO2 volume mixing ratio
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(in) :: ch4vmr(:,:) ! Methane volume mixing ratio
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(in) :: o2vmr(:,:) ! O2 volume mixing ratio
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(in) :: n2ovmr(:,:) ! Nitrous oxide volume mixing ratio
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(in) :: cfc11vmr(:,:) ! CFC11 volume mixing ratio
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(in) :: cfc12vmr(:,:) ! CFC12 volume mixing ratio
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(in) :: cfc22vmr(:,:) ! CFC22 volume mixing ratio
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(in) :: ccl4vmr(:,:) ! CCL4 volume mixing ratio
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(in) :: emis(:,:) ! Surface emissivity
            !    Dimensions: (ncol,nbndlw)
            INTEGER, intent(in) :: inflglw ! Flag for cloud optical properties
            INTEGER, intent(in) :: iceflglw ! Flag for ice particle specification
            INTEGER, intent(in) :: liqflglw ! Flag for liquid droplet specification
            REAL(KIND=r8), intent(in) :: cldfmcl(:,:,:) ! Cloud fraction
            !    Dimensions: (ngptlw,ncol,nlay)
            REAL(KIND=r8), intent(in) :: ciwpmcl(:,:,:) ! Cloud ice water path (g/m2)
            !    Dimensions: (ngptlw,ncol,nlay)
            REAL(KIND=r8), intent(in) :: clwpmcl(:,:,:) ! Cloud liquid water path (g/m2)
            !    Dimensions: (ngptlw,ncol,nlay)
            REAL(KIND=r8), intent(in) :: reicmcl(:,:) ! Cloud ice effective radius (microns)
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(in) :: relqmcl(:,:) ! Cloud water drop effective radius (microns)
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(in) :: taucmcl(:,:,:) ! Cloud optical depth
            !    Dimensions: (ngptlw,ncol,nlay)
            !      real(kind=r8), intent(in) :: ssacmcl(:,:,:)      ! Cloud single scattering albedo
            !    Dimensions: (ngptlw,ncol,nlay)
            !   for future expansion
            !   lw scattering not yet available
            !      real(kind=r8), intent(in) :: asmcmcl(:,:,:)      ! Cloud asymmetry parameter
            !    Dimensions: (ngptlw,ncol,nlay)
            !   for future expansion
            !   lw scattering not yet available
            REAL(KIND=r8), intent(in) :: tauaer(:,:,:) ! aerosol optical depth
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
            ! beginning band of calculation
            ! ending band of calculation
            ! output option flag (inactive)
            INTEGER :: iaer ! aerosol option flag
            ! column loop index
            ! flag for mcica [0=off, 1=on]
            ! value for changing mcica permute seed
            ! layer loop index
            ! g-point loop index
            ! Atmosphere
            REAL(KIND=r8) :: pavel(ncol,nlay)
            REAL(KIND=r8) :: ref_pavel(ncol,nlay) ! layer pressures (mb)
            REAL(KIND=r8) :: tavel(ncol,nlay)
            REAL(KIND=r8) :: ref_tavel(ncol,nlay) ! layer temperatures (K)
            REAL(KIND=r8) :: pz(ncol,0:nlay)
            REAL(KIND=r8) :: ref_pz(ncol,0:nlay) ! level (interface) pressures (hPa, mb)
            REAL(KIND=r8) :: tz(ncol,0:nlay)
            REAL(KIND=r8) :: ref_tz(ncol,0:nlay) ! level (interface) temperatures (K)
            REAL(KIND=r8) :: tbound(ncol)
            REAL(KIND=r8) :: ref_tbound(ncol) ! surface temperature (K)
            REAL(KIND=r8) :: coldry(ncol,nlay)
            REAL(KIND=r8) :: ref_coldry(ncol,nlay) ! dry air column density (mol/cm2)
            REAL(KIND=r8) :: wbrodl(ncol,nlay)
            REAL(KIND=r8) :: ref_wbrodl(ncol,nlay) ! broadening gas column density (mol/cm2)
            REAL(KIND=r8) :: wkl(ncol,mxmol,nlay)
            REAL(KIND=r8) :: ref_wkl(ncol,mxmol,nlay) ! molecular amounts (mol/cm-2)
            REAL(KIND=r8) :: wx(ncol,maxxsec,nlay)
            REAL(KIND=r8) :: ref_wx(ncol,maxxsec,nlay) ! cross-section amounts (mol/cm-2)
            REAL(KIND=r8) :: pwvcm(ncol)
            REAL(KIND=r8) :: ref_pwvcm(ncol) ! precipitable water vapor (cm)
            REAL(KIND=r8) :: semiss(ncol,nbndlw)
            REAL(KIND=r8) :: ref_semiss(ncol,nbndlw) ! lw surface emissivity
            !
            ! gaseous optical depths
            ! gaseous + aerosol optical depths
            REAL(KIND=r8) :: taua(ncol,nlay,nbndlw)
            REAL(KIND=r8) :: ref_taua(ncol,nlay,nbndlw) ! aerosol optical depth
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
            !
            !
            !
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
            ! number of cloud spectral bands
            INTEGER :: inflag
            INTEGER :: ref_inflag ! flag for cloud property method
            INTEGER :: iceflag
            INTEGER :: ref_iceflag ! flag for ice cloud properties
            INTEGER :: liqflag
            INTEGER :: ref_liqflag ! flag for liquid cloud properties
            ! Atmosphere/clouds - cldprmc [mcica]
            REAL(KIND=r8) :: cldfmc(ncol,ngptlw,nlay)
            REAL(KIND=r8) :: ref_cldfmc(ncol,ngptlw,nlay) ! cloud fraction [mcica]
            REAL(KIND=r8) :: ciwpmc(ncol,ngptlw,nlay)
            REAL(KIND=r8) :: ref_ciwpmc(ncol,ngptlw,nlay) ! cloud ice water path [mcica]
            REAL(KIND=r8) :: clwpmc(ncol,ngptlw,nlay)
            REAL(KIND=r8) :: ref_clwpmc(ncol,ngptlw,nlay) ! cloud liquid water path [mcica]
            REAL(KIND=r8) :: relqmc(ncol,nlay)
            REAL(KIND=r8) :: ref_relqmc(ncol,nlay) ! liquid particle size (microns)
            REAL(KIND=r8) :: reicmc(ncol,nlay)
            REAL(KIND=r8) :: ref_reicmc(ncol,nlay) ! ice particle effective radius (microns)
            REAL(KIND=r8) :: dgesmc(ncol,nlay)
            REAL(KIND=r8) :: ref_dgesmc(ncol,nlay) ! ice particle generalized effective size (microns)
            REAL(KIND=r8) :: taucmc(ncol,ngptlw,nlay)
            REAL(KIND=r8) :: ref_taucmc(ncol,ngptlw,nlay) ! cloud optical depth [mcica]
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
            tolerance = 1.E-14
            CALL kgen_init_check(check_status, tolerance)
            READ(UNIT=kgen_unit) iaer
            READ(UNIT=kgen_unit) pavel
            READ(UNIT=kgen_unit) tavel
            READ(UNIT=kgen_unit) pz
            READ(UNIT=kgen_unit) tz
            READ(UNIT=kgen_unit) tbound
            READ(UNIT=kgen_unit) coldry
            READ(UNIT=kgen_unit) wbrodl
            READ(UNIT=kgen_unit) wkl
            READ(UNIT=kgen_unit) wx
            READ(UNIT=kgen_unit) pwvcm
            READ(UNIT=kgen_unit) semiss
            READ(UNIT=kgen_unit) taua
            READ(UNIT=kgen_unit) inflag
            READ(UNIT=kgen_unit) iceflag
            READ(UNIT=kgen_unit) liqflag
            READ(UNIT=kgen_unit) cldfmc
            READ(UNIT=kgen_unit) ciwpmc
            READ(UNIT=kgen_unit) clwpmc
            READ(UNIT=kgen_unit) relqmc
            READ(UNIT=kgen_unit) reicmc
            READ(UNIT=kgen_unit) dgesmc
            READ(UNIT=kgen_unit) taucmc

            READ(UNIT=kgen_unit) ref_pavel
            READ(UNIT=kgen_unit) ref_tavel
            READ(UNIT=kgen_unit) ref_pz
            READ(UNIT=kgen_unit) ref_tz
            READ(UNIT=kgen_unit) ref_tbound
            READ(UNIT=kgen_unit) ref_coldry
            READ(UNIT=kgen_unit) ref_wbrodl
            READ(UNIT=kgen_unit) ref_wkl
            READ(UNIT=kgen_unit) ref_wx
            READ(UNIT=kgen_unit) ref_pwvcm
            READ(UNIT=kgen_unit) ref_semiss
            READ(UNIT=kgen_unit) ref_taua
            READ(UNIT=kgen_unit) ref_inflag
            READ(UNIT=kgen_unit) ref_iceflag
            READ(UNIT=kgen_unit) ref_liqflag
            READ(UNIT=kgen_unit) ref_cldfmc
            READ(UNIT=kgen_unit) ref_ciwpmc
            READ(UNIT=kgen_unit) ref_clwpmc
            READ(UNIT=kgen_unit) ref_relqmc
            READ(UNIT=kgen_unit) ref_reicmc
            READ(UNIT=kgen_unit) ref_dgesmc
            READ(UNIT=kgen_unit) ref_taucmc


            ! call to kernel
      call inatm (ncol, nlay, icld, iaer, &
           play, plev, tlay, tlev, tsfc, h2ovmr, &
           o3vmr, co2vmr, ch4vmr, o2vmr, n2ovmr, cfc11vmr, cfc12vmr, &
           cfc22vmr, ccl4vmr, emis, inflglw, iceflglw, liqflglw, &
           cldfmcl, taucmcl, ciwpmcl, clwpmcl, reicmcl, relqmcl, tauaer, &
           pavel, pz, tavel, tz, tbound, semiss, coldry, &
           wkl, wbrodl, wx, pwvcm, inflag, iceflag, liqflag, &
           cldfmc, taucmc, ciwpmc, clwpmc, reicmc, dgesmc, relqmc, taua)
            ! kernel verification for output variables
            CALL kgen_verify_real_r8_dim2( "pavel", check_status, pavel, ref_pavel)
            CALL kgen_verify_real_r8_dim2( "tavel", check_status, tavel, ref_tavel)
            CALL kgen_verify_real_r8_dim2( "pz", check_status, pz, ref_pz)
            CALL kgen_verify_real_r8_dim2( "tz", check_status, tz, ref_tz)
            CALL kgen_verify_real_r8_dim1( "tbound", check_status, tbound, ref_tbound)
            CALL kgen_verify_real_r8_dim2( "coldry", check_status, coldry, ref_coldry)
            CALL kgen_verify_real_r8_dim2( "wbrodl", check_status, wbrodl, ref_wbrodl)
            CALL kgen_verify_real_r8_dim3( "wkl", check_status, wkl, ref_wkl)
            CALL kgen_verify_real_r8_dim3( "wx", check_status, wx, ref_wx)
            CALL kgen_verify_real_r8_dim1( "pwvcm", check_status, pwvcm, ref_pwvcm)
            CALL kgen_verify_real_r8_dim2( "semiss", check_status, semiss, ref_semiss)
            CALL kgen_verify_real_r8_dim3( "taua", check_status, taua, ref_taua)
            CALL kgen_verify_integer( "inflag", check_status, inflag, ref_inflag)
            CALL kgen_verify_integer( "iceflag", check_status, iceflag, ref_iceflag)
            CALL kgen_verify_integer( "liqflag", check_status, liqflag, ref_liqflag)
            CALL kgen_verify_real_r8_dim3( "cldfmc", check_status, cldfmc, ref_cldfmc)
            CALL kgen_verify_real_r8_dim3( "ciwpmc", check_status, ciwpmc, ref_ciwpmc)
            CALL kgen_verify_real_r8_dim3( "clwpmc", check_status, clwpmc, ref_clwpmc)
            CALL kgen_verify_real_r8_dim2( "relqmc", check_status, relqmc, ref_relqmc)
            CALL kgen_verify_real_r8_dim2( "reicmc", check_status, reicmc, ref_reicmc)
            CALL kgen_verify_real_r8_dim2( "dgesmc", check_status, dgesmc, ref_dgesmc)
            CALL kgen_verify_real_r8_dim3( "taucmc", check_status, taucmc, ref_taucmc)
            CALL kgen_print_check("inatm", check_status)
            CALL system_clock(start_clock, rate_clock)
            DO kgen_intvar=1,10
                CALL inatm(ncol, nlay, icld, iaer, play, plev, tlay, tlev, tsfc, h2ovmr, o3vmr, co2vmr, ch4vmr, &
o2vmr, n2ovmr, cfc11vmr, cfc12vmr, cfc22vmr, ccl4vmr, emis, inflglw, iceflglw, liqflglw, cldfmcl, taucmcl, &
ciwpmcl, clwpmcl, reicmcl, relqmcl, tauaer, pavel, pz, tavel, tz, tbound, semiss, coldry, wkl, wbrodl, wx, &
pwvcm, inflag, iceflag, liqflag, cldfmc, taucmc, ciwpmc, clwpmc, reicmc, dgesmc, relqmc, taua)
            END DO
            CALL system_clock(stop_clock, rate_clock)
            WRITE(*,*)
            PRINT *, "Elapsed time (sec): ", (stop_clock - start_clock)/REAL(rate_clock*10)
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

            SUBROUTINE kgen_read_real_r8_dim1(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                real(KIND=r8), INTENT(OUT), ALLOCATABLE, DIMENSION(:) :: var
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
            END SUBROUTINE kgen_read_real_r8_dim1

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

            SUBROUTINE kgen_verify_real_r8_dim1( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                real(KIND=r8), intent(in), DIMENSION(:) :: var, ref_var
                real(KIND=r8) :: nrmsdiff, rmsdiff
                real(KIND=r8), allocatable, DIMENSION(:) :: temp, temp2
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
                    allocate(temp(SIZE(var,dim=1)))
                    allocate(temp2(SIZE(var,dim=1)))
                
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
            END SUBROUTINE kgen_verify_real_r8_dim1

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

            SUBROUTINE kgen_verify_integer( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                integer, intent(in) :: var, ref_var
                check_status%numTotal = check_status%numTotal + 1
                IF ( var == ref_var ) THEN
                    check_status%numIdentical = check_status%numIdentical + 1
                    if(check_status%verboseLevel > 1) then
                        WRITE(*,*)
                        WRITE(*,*) trim(adjustl(varname)), " is IDENTICAL( ", var, " )."
                    endif
                ELSE
                    if(check_status%verboseLevel > 0) then
                        WRITE(*,*)
                        WRITE(*,*) trim(adjustl(varname)), " is NOT IDENTICAL."
                        if(check_status%verboseLevel > 2) then
                            WRITE(*,*) "KERNEL: ", var
                            WRITE(*,*) "REF.  : ", ref_var
                        end if
                    end if
                    check_status%numFatal = check_status%numFatal + 1
                END IF
            END SUBROUTINE kgen_verify_integer

        END SUBROUTINE rrtmg_lw
        !***************************************************************************

        SUBROUTINE inatm(ncol, nlay, icld, iaer, play, plev, tlay, tlev, tsfc, h2ovmr, o3vmr, co2vmr, ch4vmr, o2vmr, n2ovmr, &
        cfc11vmr, cfc12vmr, cfc22vmr, ccl4vmr, emis, inflglw, iceflglw, liqflglw, cldfmcl, taucmcl, ciwpmcl, clwpmcl, reicmcl, &
        relqmcl, tauaer, pavel, pz, tavel, tz, tbound, semiss, coldry, wkl, wbrodl, wx, pwvcm, inflag, iceflag, liqflag, cldfmc, &
        taucmc, ciwpmc, clwpmc, reicmc, dgesmc, relqmc, taua)
            !***************************************************************************
            !
            !  Input atmospheric profile from GCM, and prepare it for use in RRTMG_LW.
            !  Set other RRTMG_LW input parameters.
            !
            !***************************************************************************
            ! --------- Modules ----------
            USE parrrtm, ONLY: nmol
            USE parrrtm, ONLY: maxxsec
            USE parrrtm, ONLY: nbndlw
            USE parrrtm, ONLY: ngptlw
            USE rrlw_con, ONLY: grav
            USE rrlw_con, ONLY: avogad
            USE rrlw_wvn, ONLY: ixindx
            ! ------- Declarations -------
            ! ----- Input -----
            INTEGER, intent(in) :: ncol ! total number of columns
            INTEGER, intent(in) :: nlay ! Number of model layers
            INTEGER, intent(in) :: icld ! clear/cloud and cloud overlap flag
            INTEGER, intent(in) :: iaer ! aerosol option flag
            REAL(KIND=r8), intent(in) :: play(:,:) ! Layer pressures (hPa, mb)
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(in) :: plev(:,:) ! Interface pressures (hPa, mb)
            !    Dimensions: (ncol,nlay+1)
            REAL(KIND=r8), intent(in) :: tlay(:,:) ! Layer temperatures (K)
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(in) :: tlev(:,:) ! Interface temperatures (K)
            !    Dimensions: (ncol,nlay+1)
            REAL(KIND=r8), intent(in) :: tsfc(:) ! Surface temperature (K)
            !    Dimensions: (ncol)
            REAL(KIND=r8), intent(in) :: h2ovmr(:,:) ! H2O volume mixing ratio
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(in) :: o3vmr(:,:) ! O3 volume mixing ratio
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(in) :: co2vmr(:,:) ! CO2 volume mixing ratio
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(in) :: ch4vmr(:,:) ! Methane volume mixing ratio
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(in) :: o2vmr(:,:) ! O2 volume mixing ratio
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(in) :: n2ovmr(:,:) ! Nitrous oxide volume mixing ratio
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(in) :: cfc11vmr(:,:) ! CFC11 volume mixing ratio
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(in) :: cfc12vmr(:,:) ! CFC12 volume mixing ratio
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(in) :: cfc22vmr(:,:) ! CFC22 volume mixing ratio
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(in) :: ccl4vmr(:,:) ! CCL4 volume mixing ratio
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(in) :: emis(:,:) ! Surface emissivity
            !    Dimensions: (ncol,nbndlw)
            INTEGER, intent(in) :: inflglw ! Flag for cloud optical properties
            INTEGER, intent(in) :: iceflglw ! Flag for ice particle specification
            INTEGER, intent(in) :: liqflglw ! Flag for liquid droplet specification
            REAL(KIND=r8), intent(in) :: cldfmcl(:,:,:) ! Cloud fraction
            !    Dimensions: (ngptlw,ncol,nlay)
            REAL(KIND=r8), intent(in) :: ciwpmcl(:,:,:) ! Cloud ice water path (g/m2)
            !    Dimensions: (ngptlw,ncol,nlay)
            REAL(KIND=r8), intent(in) :: clwpmcl(:,:,:) ! Cloud liquid water path (g/m2)
            !    Dimensions: (ngptlw,ncol,nlay)
            REAL(KIND=r8), intent(in) :: reicmcl(:,:) ! Cloud ice effective radius (microns)
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(in) :: relqmcl(:,:) ! Cloud water drop effective radius (microns)
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(in) :: taucmcl(:,:,:) ! Cloud optical depth
            !    Dimensions: (ngptlw,ncol,nlay)
            REAL(KIND=r8), intent(in) :: tauaer(:,:,:) ! Aerosol optical depth
            !    Dimensions: (ncol,nlay,nbndlw)
            ! ----- Output -----
            ! Atmosphere
            REAL(KIND=r8), intent(out) :: pavel(:,:) ! layer pressures (mb)
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(out) :: tavel(:,:) ! layer temperatures (K)
            !    Dimensions: (ncol, nlay)
            REAL(KIND=r8), intent(out) :: pz(ncol,0:nlay) ! level (interface) pressures (hPa, mb)
            !    Dimensions: (ncol,0:nlay)
            REAL(KIND=r8), intent(out) :: tz(ncol,0:nlay) ! level (interface) temperatures (K)
            !    Dimensions: (ncol,0:nlay)
            REAL(KIND=r8), intent(out) :: tbound(:) ! surface temperature (K)
            !    Dimensions: (ncol)
            REAL(KIND=r8), intent(out) :: coldry(ncol,nlay) ! dry air column density (mol/cm2)
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(out) :: wbrodl(:,:) ! broadening gas column density (mol/cm2)
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(out) :: wkl(:,:,:) ! molecular amounts (mol/cm-2)
            !    Dimensions: (ncol,mxmol,nlay)
            REAL(KIND=r8), intent(out) :: wx(:,:,:) ! cross-section amounts (mol/cm-2)
            !    Dimensions: (ncol,maxxsec,nlay)
            REAL(KIND=r8), intent(out) :: pwvcm(:) ! precipitable water vapor (cm)
            !    Dimensions: (ncol)
            REAL(KIND=r8), intent(out) :: semiss(:,:) ! lw surface emissivity
            !    Dimensions: (ncol,nbndlw)
            ! Atmosphere/clouds - cldprop
            INTEGER, intent(out) :: inflag ! flag for cloud property method
            INTEGER, intent(out) :: iceflag ! flag for ice cloud properties
            INTEGER, intent(out) :: liqflag ! flag for liquid cloud properties
            REAL(KIND=r8), intent(out) :: cldfmc(:,:,:) ! cloud fraction [mcica]
            !    Dimensions: (ncol,ngptlw,nlay)
            REAL(KIND=r8), intent(out) :: ciwpmc(:,:,:) ! cloud ice water path [mcica]
            !    Dimensions: (ncol,ngptlw,nlay)
            REAL(KIND=r8), intent(out) :: clwpmc(:,:,:) ! cloud liquid water path [mcica]
            !    Dimensions: (ncol,ngptlw,nlay)
            REAL(KIND=r8), intent(out) :: relqmc(:,:) ! liquid particle effective radius (microns)
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(out) :: reicmc(:,:) ! ice particle effective radius (microns)
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(out) :: dgesmc(:,:) ! ice particle generalized effective size (microns)
            !    Dimensions: (ncol,nlay)
            REAL(KIND=r8), intent(out) :: taucmc(:,:,:) ! cloud optical depth [mcica]
            !    Dimensions: (ncol,ngptlw,nlay)
            REAL(KIND=r8), intent(out) :: taua(:,:,:) ! Aerosol optical depth
            ! Dimensions: (ncol,nlay,nbndlw)
            ! ----- Local -----
            REAL(KIND=r8), parameter :: amd = 28.9660_r8 ! Effective molecular weight of dry air (g/mol)
            REAL(KIND=r8), parameter :: amw = 18.0160_r8 ! Molecular weight of water vapor (g/mol)
            !      real(kind=r8), parameter :: amc = 44.0098_r8      ! Molecular weight of carbon dioxide (g/mol)
            !      real(kind=r8), parameter :: amo = 47.9998_r8      ! Molecular weight of ozone (g/mol)
            !      real(kind=r8), parameter :: amo2 = 31.9999_r8     ! Molecular weight of oxygen (g/mol)
            !      real(kind=r8), parameter :: amch4 = 16.0430_r8    ! Molecular weight of methane (g/mol)
            !      real(kind=r8), parameter :: amn2o = 44.0128_r8    ! Molecular weight of nitrous oxide (g/mol)
            !      real(kind=r8), parameter :: amc11 = 137.3684_r8   ! Molecular weight of CFC11 (g/mol) - CCL3F
            !      real(kind=r8), parameter :: amc12 = 120.9138_r8   ! Molecular weight of CFC12 (g/mol) - CCL2F2
            !      real(kind=r8), parameter :: amc22 = 86.4688_r8    ! Molecular weight of CFC22 (g/mol) - CHCLF2
            !      real(kind=r8), parameter :: amcl4 = 153.823_r8    ! Molecular weight of CCL4 (g/mol) - CCL4
            ! Set molecular weight ratios (for converting mmr to vmr)
            !  e.g. h2ovmr = h2ommr * amdw)
            ! Molecular weight of dry air / water vapor
            ! Molecular weight of dry air / carbon dioxide
            ! Molecular weight of dry air / ozone
            ! Molecular weight of dry air / methane
            ! Molecular weight of dry air / nitrous oxide
            ! Molecular weight of dry air / CFC11
            ! Molecular weight of dry air / CFC12
            ! Stefan-Boltzmann constant (W/m2K4)
            INTEGER :: l, iplon
            INTEGER :: imol
            INTEGER :: ix
            INTEGER :: n
            INTEGER :: ib
            INTEGER :: ig ! Loop indices
            REAL(KIND=r8) :: amttl
            REAL(KIND=r8) :: wvttl
            REAL(KIND=r8) :: summol
            REAL(KIND=r8) :: wvsh
            ! promote temporary scalars to vectors
            REAL(KIND=r8) :: amm(ncol,nlay) ! pr
            !  Initialize all molecular amounts and cloud properties to zero here, then pass input amounts
            !  into RRTM arrays below.
            !JMD !DIR$ ASSUME_ALIGNED pz:64
            !  Set surface temperature.
      tbound = tsfc
            !  Install input GCM arrays into RRTMG_LW arrays for pressure, temperature,
            !  and molecular amounts.
            !  Pressures are input in mb, or are converted to mb here.
            !  Molecular amounts are input in volume mixing ratio, or are converted from
            !  mass mixing ratio (or specific humidity for h2o) to volume mixing ratio
            !  here. These are then converted to molecular amount (molec/cm2) below.
            !  The dry air column COLDRY (in molec/cm2) is calculated from the level
            !  pressures, pz (in mb), based on the hydrostatic equation and includes a
            !  correction to account for h2o in the layer.  The molecular weight of moist
            !  air (amm) is calculated for each layer.
            !  Note: In RRTMG, layer indexing goes from bottom to top, and coding below
            !  assumes GCM input fields are also bottom to top. Input layer indexing
            !  from GCM fields should be reversed here if necessary.
      pz(:,0) = plev(:,nlay+1)
      tz(:,0) = tlev(:,nlay+1)
      do l = 1, nlay
    do iplon=1,ncol
         pavel(iplon,l) = play(iplon,nlay-l+1)
         tavel(iplon,l) = tlay(iplon,nlay-l+1)
         pz(iplon,l) = plev(iplon,nlay-l+1)
         tz(iplon,l) = tlev(iplon,nlay-l+1)
                    ! For h2o input in vmr:
         wkl(iplon,1,l) = h2ovmr(iplon,nlay-l+1)
                    ! For h2o input in mmr:
                    !         wkl(1,l) = h2o(iplon,nlay-l)*amdw
                    ! For h2o input in specific humidity;
                    !         wkl(1,l) = (h2o(iplon,nlay-l)/(1._r8 - h2o(iplon,nlay-l)))*amdw
         wkl(iplon,2,l) = co2vmr(iplon,nlay-l+1)
         wkl(iplon,3,l) = o3vmr(iplon,nlay-l+1)
         wkl(iplon,4,l) = n2ovmr(iplon,nlay-l+1)
         wkl(iplon,5,l) = 0._r8
         wkl(iplon,6,l) = ch4vmr(iplon,nlay-l+1)
         wkl(iplon,7,l) = o2vmr(iplon,nlay-l+1)
         amm(iplon,l) = (1._r8 - wkl(iplon,1,l)) * amd + wkl(iplon,1,l) * amw            
         coldry(iplon,l) = (pz(iplon,l-1)-pz(iplon,l)) * 1.e3_r8 * avogad / &
                     (1.e2_r8 * grav * amm(iplon,l) * (1._r8 + wkl(iplon,1,l)))
                    ! Set cross section molecule amounts from input; convert to vmr if necessary
         wx(iplon,1,l) = ccl4vmr(iplon,nlay-l+1)
         wx(iplon,2,l) = cfc11vmr(iplon,nlay-l+1)
         wx(iplon,3,l) = cfc12vmr(iplon,nlay-l+1)
         wx(iplon,4,l) = cfc22vmr(iplon,nlay-l+1)
   enddo
        enddo
      coldry(:,nlay) = (pz(:,nlay-1)) * 1.e3_r8 * avogad / &
                        (1.e2_r8 * grav * amm(:,nlay) * (1._r8 + wkl(:,1,nlay-1)))
            ! At this point all molecular amounts in wkl and wx are in volume mixing ratio;
            ! convert to molec/cm2 based on coldry for use in rrtm.  also, compute precipitable
            ! water vapor for diffusivity angle adjustments in rtrn and rtrnmr.
    do iplon = 1,ncol
      amttl = 0.0_r8
      wvttl = 0.0_r8
      do l = 1, nlay
         summol = 0.0_r8
         do imol = 2, nmol
            summol = summol + wkl(iplon,imol,l)
         enddo
         wbrodl(iplon,l) = coldry(iplon,l) * (1._r8 - summol)
         do imol = 1, nmol
            wkl(iplon,imol,l) = coldry(iplon,l) * wkl(iplon,imol,l)
         enddo
         amttl = amttl + coldry(iplon,l)+wkl(iplon,1,l)
         wvttl = wvttl + wkl(iplon,1,l)
         do ix = 1,maxxsec
            if (ixindx(ix) .ne. 0) then
               wx(iplon,ixindx(ix),l) = coldry(iplon,l) * wx(iplon,ix,l) * 1.e-20_r8
            endif
         enddo
      enddo
      wvsh = (amw * wvttl) / (amd * amttl)
      pwvcm(iplon) = wvsh * (1.e3_r8 * pz(iplon,0)) / (1.e2_r8 * grav)
                ! Set spectral surface emissivity for each longwave band.
      do n=1,nbndlw
         semiss(iplon,n) = emis(iplon,n)
                    !          semiss(n) = 1.0_r8
      enddo
    enddo 
            ! Transfer aerosol optical properties to RRTM variable;
            ! modify to reverse layer indexing here if necessary.
      if (iaer .ge. 1) then 
         do ib = 1, nbndlw
            do l = 1, nlay-1
               do iplon=1,ncol
                  taua(iplon,l,ib) = tauaer(iplon,nlay-l,ib)
               enddo
            enddo
         enddo
      endif
            ! Transfer cloud fraction and cloud optical properties to RRTM variables,
            ! modify to reverse layer indexing here if necessary.
      if (icld .ge. 1) then 
         inflag = inflglw
         iceflag = iceflglw
         liqflag = liqflglw
                ! Move incoming GCM cloud arrays to RRTMG cloud arrays.
                ! For GCM input, incoming reice is in effective radius; for Fu parameterization (iceflag = 3)
                ! convert effective radius to generalized effective size using method of Mitchell, JAS, 2002:
         do l = 1, nlay-1
            do ig = 1, ngptlw
            do iplon=1,ncol
               cldfmc(iplon,ig,l) = cldfmcl(ig,iplon,nlay-l)
               taucmc(iplon,ig,l) = taucmcl(ig,iplon,nlay-l)
               ciwpmc(iplon,ig,l) = ciwpmcl(ig,iplon,nlay-l)
               clwpmc(iplon,ig,l) = clwpmcl(ig,iplon,nlay-l)
            enddo
            enddo
            do iplon=1,ncol
               reicmc(iplon,l) = reicmcl(iplon,nlay-l)
               relqmc(iplon,l) = relqmcl(iplon,nlay-l)
            enddo
            if (iceflag .eq. 3) then
               do iplon=1,ncol
                  dgesmc(iplon,l) = 1.5396_r8 * reicmcl(iplon,nlay-l)
               enddo
            endif
         enddo
                ! If an extra layer is being used in RRTMG, set all cloud properties to zero in the extra layer.
       do iplon=1,ncol
         cldfmc(iplon,:,nlay) = 0.0_r8
         taucmc(iplon,:,nlay) = 0.0_r8
         ciwpmc(iplon,:,nlay) = 0.0_r8
         clwpmc(iplon,:,nlay) = 0.0_r8
         reicmc(iplon,nlay) = 0.0_r8
         dgesmc(iplon,nlay) = 0.0_r8
         relqmc(iplon,nlay) = 0.0_r8
         taua(iplon,nlay,:) = 0.0_r8
       enddo 
      endif
        END SUBROUTINE inatm
    END MODULE rrtmg_lw_rad
