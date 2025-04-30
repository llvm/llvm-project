
! KGEN-generated Fortran source file
!
! Filename    : micro_mg2_0.F90
! Generated at: 2015-03-31 09:44:40
! KGEN version: 0.4.5



    MODULE micro_mg2_0
        !---------------------------------------------------------------------------------
        ! Purpose:
        !   MG microphysics version 2.0 - Update of MG microphysics with
        !                                 prognostic precipitation.
        !
        ! Author: Andrew Gettelman, Hugh Morrison.
        ! Contributions from: Peter Caldwell, Xiaohong Liu and Steve Ghan
        ! Version 2 history: Sep 2011: Development begun.
        !                    Feb 2013: Added of prognostic precipitation.
        ! invoked in 1 by specifying -microphys=mg2.0
        !
        ! for questions contact Hugh Morrison, Andrew Gettelman
        ! e-mail: morrison@ucar.edu, andrew@ucar.edu
        !---------------------------------------------------------------------------------
        !
        ! NOTE: Modified to allow other microphysics packages (e.g. CARMA) to do ice
        ! microphysics in cooperation with the MG liquid microphysics. This is
        ! controlled by the do_cldice variable.
        !
        ! If do_cldice is false, then MG microphysics should not update CLDICE or
        ! NUMICE; it is assumed that the other microphysics scheme will have updated
        ! CLDICE and NUMICE. The other microphysics should handle the following
        ! processes that would have been done by MG:
        !   - Detrainment (liquid and ice)
        !   - Homogeneous ice nucleation
        !   - Heterogeneous ice nucleation
        !   - Bergeron process
        !   - Melting of ice
        !   - Freezing of cloud drops
        !   - Autoconversion (ice -> snow)
        !   - Growth/Sublimation of ice
        !   - Sedimentation of ice
        !
        ! This option has not been updated since the introduction of prognostic
        ! precipitation, and probably should be adjusted to cover snow as well.
        !
        !---------------------------------------------------------------------------------
        ! Based on micro_mg (restructuring of former cldwat2m_micro)
        ! Author: Andrew Gettelman, Hugh Morrison.
        ! Contributions from: Xiaohong Liu and Steve Ghan
        ! December 2005-May 2010
        ! Description in: Morrison and Gettelman, 2008. J. Climate (MG2008)
        !                 Gettelman et al., 2010 J. Geophys. Res. - Atmospheres (G2010)
        ! for questions contact Hugh Morrison, Andrew Gettelman
        ! e-mail: morrison@ucar.edu, andrew@ucar.edu
        !---------------------------------------------------------------------------------
        ! Code comments added by HM, 093011
        ! General code structure:
        !
        ! Code is divided into two main subroutines:
        !   subroutine micro_mg_init --> initializes microphysics routine, should be called
        !                                  once at start of simulation
        !   subroutine micro_mg_tend --> main microphysics routine to be called each time step
        !                                this also calls several smaller subroutines to calculate
        !                                microphysical processes and other utilities
        !
        ! List of external functions:
        !   qsat_water --> for calculating saturation vapor pressure with respect to liquid water
        !   qsat_ice --> for calculating saturation vapor pressure with respect to ice
        !   gamma   --> standard mathematical gamma function
        ! .........................................................................
        ! List of inputs through use statement in fortran90:
        ! Variable Name                      Description                Units
        ! .........................................................................
        ! gravit          acceleration due to gravity                    m s-2
        ! rair            dry air gas constant for air                  J kg-1 K-1
        ! tmelt           temperature of melting point for water          K
        ! cpair           specific heat at constant pressure for dry air J kg-1 K-1
        ! rh2o            gas constant for water vapor                  J kg-1 K-1
        ! latvap          latent heat of vaporization                   J kg-1
        ! latice          latent heat of fusion                         J kg-1
        ! qsat_water      external function for calculating liquid water
        !                 saturation vapor pressure/humidity              -
        ! qsat_ice        external function for calculating ice
        !                 saturation vapor pressure/humidity              pa
        ! rhmini          relative humidity threshold parameter for
        !                 nucleating ice                                  -
        ! .........................................................................
        ! NOTE: List of all inputs/outputs passed through the call/subroutine statement
        !       for micro_mg_tend is given below at the start of subroutine micro_mg_tend.
        !---------------------------------------------------------------------------------
        ! Procedures required:
        ! 1) An implementation of the gamma function (if not intrinsic).
        ! 2) saturation vapor pressure and specific humidity over water
        ! 3) svp over ice
        USE shr_spfn_mod, ONLY: gamma => shr_spfn_gamma
        USE wv_sat_methods, ONLY: qsat_water => wv_sat_qsat_water
        USE wv_sat_methods, ONLY: qsat_ice => wv_sat_qsat_ice
        ! Parameters from the utilities module.
        USE micro_mg_utils, ONLY: r8
        USE micro_mg_utils, ONLY: qsmall
        USE micro_mg_utils, ONLY: mincld
        USE micro_mg_utils, ONLY: ar
        USE micro_mg_utils, ONLY: as
        USE micro_mg_utils, ONLY: rhow
        USE micro_mg_utils, ONLY: ai
        USE micro_mg_utils, ONLY: mi0
        USE micro_mg_utils, ONLY: br
        USE micro_mg_utils, ONLY: bs
        USE micro_mg_utils, ONLY: pi
        USE micro_mg_utils, ONLY: rhosn
        USE micro_mg_utils, ONLY: omsm
        USE micro_mg_utils, ONLY: rising_factorial
        USE micro_mg_utils, ONLY: bc
        USE micro_mg_utils, ONLY: bi
        USE micro_mg_utils, ONLY: rhows
        USE micro_mg_utils, ONLY: rhoi
        IMPLICIT NONE
        PRIVATE
        PUBLIC micro_mg_tend
        ! switch for specification rather than prediction of droplet and crystal number
        ! note: number will be adjusted as needed to keep mean size within bounds,
        ! even when specified droplet or ice number is used
        ! If constant cloud ice number is set (nicons = .true.),
        ! then all microphysical processes except mass transfer due to ice nucleation
        ! (mnuccd) are based on the fixed cloud ice number. Calculation of
        ! mnuccd follows from the prognosed ice crystal number ni.
        ! nccons = .true. to specify constant cloud droplet number
        ! nicons = .true. to specify constant cloud ice number
        LOGICAL, parameter, public :: nccons = .false.
        LOGICAL, parameter, public :: nicons = .false.
        !=========================================================
        ! Private module parameters
        !=========================================================
        ! parameters for specified ice and droplet number concentration
        ! note: these are local in-cloud values, not grid-mean
        REAL(KIND=r8), parameter :: ncnst = 100.e6_r8 ! droplet num concentration when nccons=.true. (m-3)
        REAL(KIND=r8), parameter :: ninst = 0.1e6_r8 ! ice num concentration when nicons=.true. (m-3)
        !Range of cloudsat reflectivities (dBz) for analytic simulator
        REAL(KIND=r8), parameter :: csmin = -30._r8
        REAL(KIND=r8), parameter :: csmax = 26._r8
        REAL(KIND=r8), parameter :: mindbz = -99._r8
        REAL(KIND=r8), parameter :: minrefl = 1.26e-10_r8 ! minrefl = 10._r8**(mindbz/10._r8)
        ! autoconversion size threshold for cloud ice to snow (m)
        REAL(KIND=r8) :: dcs
        ! minimum mass of new crystal due to freezing of cloud droplets done
        ! externally (kg)
        REAL(KIND=r8), parameter :: mi0l_min = 4._r8/3._r8*pi*rhow*(4.e-6_r8)**3
        !=========================================================
        ! Constants set in initialization
        !=========================================================
        ! Set using arguments to micro_mg_init
        REAL(KIND=r8) :: g ! gravity
        REAL(KIND=r8) :: r ! dry air gas constant
        REAL(KIND=r8) :: rv ! water vapor gas constant
        REAL(KIND=r8) :: cpp ! specific heat of dry air
        REAL(KIND=r8) :: tmelt ! freezing point of water (K)
        ! latent heats of:
        REAL(KIND=r8) :: xxlv ! vaporization
        REAL(KIND=r8) :: xlf ! freezing
        REAL(KIND=r8) :: xxls ! sublimation
        REAL(KIND=r8) :: rhmini ! Minimum rh for ice cloud fraction > 0.
        ! flags
        LOGICAL :: microp_uniform
        LOGICAL :: do_cldice
        LOGICAL :: use_hetfrz_classnuc
        REAL(KIND=r8) :: rhosu ! typical 850mn air density
        REAL(KIND=r8) :: icenuct ! ice nucleation temperature: currently -5 degrees C
        REAL(KIND=r8) :: snowmelt ! what temp to melt all snow: currently 2 degrees C
        REAL(KIND=r8) :: rainfrze ! what temp to freeze all rain: currently -5 degrees C
        ! additional constants to help speed up code
        REAL(KIND=r8) :: gamma_br_plus1
        REAL(KIND=r8) :: gamma_br_plus4
        REAL(KIND=r8) :: gamma_bs_plus1
        REAL(KIND=r8) :: gamma_bs_plus4
        REAL(KIND=r8) :: gamma_bi_plus1
        REAL(KIND=r8) :: gamma_bi_plus4
        REAL(KIND=r8) :: xxlv_squared
        REAL(KIND=r8) :: xxls_squared
        CHARACTER(LEN=16) :: micro_mg_precip_frac_method ! type of precipitation fraction method
        REAL(KIND=r8) :: micro_mg_berg_eff_factor ! berg efficiency factor
        !===============================================================================
            PUBLIC kgen_read_externs_micro_mg2_0
        CONTAINS

        ! write subroutines
        ! No subroutines

        ! module extern variables

        SUBROUTINE kgen_read_externs_micro_mg2_0(kgen_unit)
            INTEGER, INTENT(IN) :: kgen_unit
            READ(UNIT=kgen_unit) dcs
            READ(UNIT=kgen_unit) g
            READ(UNIT=kgen_unit) r
            READ(UNIT=kgen_unit) rv
            READ(UNIT=kgen_unit) cpp
            READ(UNIT=kgen_unit) tmelt
            READ(UNIT=kgen_unit) xxlv
            READ(UNIT=kgen_unit) xlf
            READ(UNIT=kgen_unit) xxls
            READ(UNIT=kgen_unit) rhmini
            READ(UNIT=kgen_unit) microp_uniform
            READ(UNIT=kgen_unit) do_cldice
            READ(UNIT=kgen_unit) use_hetfrz_classnuc
            READ(UNIT=kgen_unit) rhosu
            READ(UNIT=kgen_unit) icenuct
            READ(UNIT=kgen_unit) snowmelt
            READ(UNIT=kgen_unit) rainfrze
            READ(UNIT=kgen_unit) gamma_br_plus1
            READ(UNIT=kgen_unit) gamma_br_plus4
            READ(UNIT=kgen_unit) gamma_bs_plus1
            READ(UNIT=kgen_unit) gamma_bs_plus4
            READ(UNIT=kgen_unit) gamma_bi_plus1
            READ(UNIT=kgen_unit) gamma_bi_plus4
            READ(UNIT=kgen_unit) xxlv_squared
            READ(UNIT=kgen_unit) xxls_squared
            READ(UNIT=kgen_unit) micro_mg_precip_frac_method
            READ(UNIT=kgen_unit) micro_mg_berg_eff_factor
        END SUBROUTINE kgen_read_externs_micro_mg2_0

        !===============================================================================

        !===============================================================================
        !microphysics routine for each timestep goes here...

        SUBROUTINE micro_mg_tend(mgncol, nlev, deltatin, t, q, qcn, qin, ncn, nin, qrn, qsn, nrn, nsn, relvar, accre_enhan, p, &
        pdel, cldn, liqcldf, icecldf, qcsinksum_rate1ord, naai, npccn, rndst, nacon, tlat, qvlat, qctend, qitend, nctend, nitend, &
        qrtend, qstend, nrtend, nstend, effc, effc_fn, effi, prect, preci, nevapr, evapsnow, prain, prodsnow, cmeout, deffi, &
        pgamrad, lamcrad, qsout, dsout, rflx, sflx, qrout, reff_rain, reff_snow, qcsevap, qisevap, qvres, cmeitot, vtrmc, vtrmi, &
        umr, ums, qcsedten, qisedten, qrsedten, qssedten, pratot, prctot, mnuccctot, mnuccttot, msacwitot, psacwstot, bergstot, &
        bergtot, melttot, homotot, qcrestot, prcitot, praitot, qirestot, mnuccrtot, pracstot, meltsdttot, frzrdttot, mnuccdtot, &
        nrout, nsout, refl, arefl, areflz, frefl, csrfl, acsrfl, fcsrfl, rercld, ncai, ncal, qrout2, qsout2, nrout2, nsout2, &
        drout2, dsout2, freqs, freqr, nfice, qcrat, errstring, tnd_qsnow, tnd_nsnow, re_ice, prer_evap, frzimm, frzcnt, frzdep)
            ! Below arguments are "optional" (pass null pointers to omit).
            ! Constituent properties.
            USE micro_mg_utils, ONLY: mg_liq_props
            USE micro_mg_utils, ONLY: mg_ice_props
            USE micro_mg_utils, ONLY: mg_rain_props
            USE micro_mg_utils, ONLY: mg_snow_props
            ! Size calculation functions.
            USE micro_mg_utils, ONLY: size_dist_param_liq
            USE micro_mg_utils, ONLY: size_dist_param_basic
            USE micro_mg_utils, ONLY: avg_diameter
            ! Microphysical processes.
            USE micro_mg_utils, ONLY: kk2000_liq_autoconversion
            USE micro_mg_utils, ONLY: ice_autoconversion
            USE micro_mg_utils, ONLY: immersion_freezing
            USE micro_mg_utils, ONLY: contact_freezing
            USE micro_mg_utils, ONLY: snow_self_aggregation
            USE micro_mg_utils, ONLY: accrete_cloud_water_snow
            USE micro_mg_utils, ONLY: secondary_ice_production
            USE micro_mg_utils, ONLY: accrete_rain_snow
            USE micro_mg_utils, ONLY: heterogeneous_rain_freezing
            USE micro_mg_utils, ONLY: accrete_cloud_water_rain
            USE micro_mg_utils, ONLY: self_collection_rain
            USE micro_mg_utils, ONLY: accrete_cloud_ice_snow
            USE micro_mg_utils, ONLY: evaporate_sublimate_precip
            USE micro_mg_utils, ONLY: bergeron_process_snow
            USE micro_mg_utils, ONLY: ice_deposition_sublimation
            !Authors: Hugh Morrison, Andrew Gettelman, NCAR, Peter Caldwell, LLNL
            ! e-mail: morrison@ucar.edu, andrew@ucar.edu
            ! input arguments
            INTEGER, intent(in) :: mgncol ! number of microphysics columns
            INTEGER, intent(in) :: nlev ! number of layers
            REAL(KIND=r8), intent(in) :: deltatin ! time step (s)
            REAL(KIND=r8), intent(in) :: t(:,:) ! input temperature (K)
            REAL(KIND=r8), intent(in) :: q(:,:) ! input h20 vapor mixing ratio (kg/kg)
            ! note: all input cloud variables are grid-averaged
            REAL(KIND=r8), intent(in) :: qcn(:,:) ! cloud water mixing ratio (kg/kg)
            REAL(KIND=r8), intent(in) :: qin(:,:) ! cloud ice mixing ratio (kg/kg)
            REAL(KIND=r8), intent(in) :: ncn(:,:) ! cloud water number conc (1/kg)
            REAL(KIND=r8), intent(in) :: nin(:,:) ! cloud ice number conc (1/kg)
            REAL(KIND=r8), intent(in) :: qrn(:,:) ! rain mixing ratio (kg/kg)
            REAL(KIND=r8), intent(in) :: qsn(:,:) ! snow mixing ratio (kg/kg)
            REAL(KIND=r8), intent(in) :: nrn(:,:) ! rain number conc (1/kg)
            REAL(KIND=r8), intent(in) :: nsn(:,:) ! snow number conc (1/kg)
            REAL(KIND=r8), intent(in) :: relvar(:,:) ! cloud water relative variance (-)
            REAL(KIND=r8), intent(in) :: accre_enhan(:,:) ! optional accretion
            ! enhancement factor (-)
            REAL(KIND=r8), intent(in) :: p(:,:) ! air pressure (pa)
            REAL(KIND=r8), intent(in) :: pdel(:,:) ! pressure difference across level (pa)
            REAL(KIND=r8), intent(in) :: cldn(:,:) ! cloud fraction (no units)
            REAL(KIND=r8), intent(in) :: liqcldf(:,:) ! liquid cloud fraction (no units)
            REAL(KIND=r8), intent(in) :: icecldf(:,:) ! ice cloud fraction (no units)
            ! used for scavenging
            ! Inputs for aerosol activation
            REAL(KIND=r8), intent(in) :: naai(:,:) ! ice nucleation number (from microp_aero_ts) (1/kg)
            REAL(KIND=r8), intent(in) :: npccn(:,:) ! ccn activated number tendency (from microp_aero_ts) (1/kg*s)
            ! Note that for these variables, the dust bin is assumed to be the last index.
            ! (For example, in 1, the last dimension is always size 4.)
            REAL(KIND=r8), intent(in) :: rndst(:,:,:) ! radius of each dust bin, for contact freezing (from microp_aero_ts) (m)
            REAL(KIND=r8), intent(in) :: nacon(:,:,:) ! number in each dust bin, for contact freezing  (from microp_aero_ts) (1/m^3)
            ! output arguments
            REAL(KIND=r8), intent(out) :: qcsinksum_rate1ord(:,:) ! 1st order rate for
            ! direct cw to precip conversion
            REAL(KIND=r8), intent(out) :: tlat(:,:) ! latent heating rate       (W/kg)
            REAL(KIND=r8), intent(out) :: qvlat(:,:) ! microphysical tendency qv (1/s)
            REAL(KIND=r8), intent(out) :: qctend(:,:) ! microphysical tendency qc (1/s)
            REAL(KIND=r8), intent(out) :: qitend(:,:) ! microphysical tendency qi (1/s)
            REAL(KIND=r8), intent(out) :: nctend(:,:) ! microphysical tendency nc (1/(kg*s))
            REAL(KIND=r8), intent(out) :: nitend(:,:) ! microphysical tendency ni (1/(kg*s))
            REAL(KIND=r8), intent(out) :: qrtend(:,:) ! microphysical tendency qr (1/s)
            REAL(KIND=r8), intent(out) :: qstend(:,:) ! microphysical tendency qs (1/s)
            REAL(KIND=r8), intent(out) :: nrtend(:,:) ! microphysical tendency nr (1/(kg*s))
            REAL(KIND=r8), intent(out) :: nstend(:,:) ! microphysical tendency ns (1/(kg*s))
            REAL(KIND=r8), intent(out) :: effc(:,:) ! droplet effective radius (micron)
            REAL(KIND=r8), intent(out) :: effc_fn(:,:) ! droplet effective radius, assuming nc = 1.e8 kg-1
            REAL(KIND=r8), intent(out) :: effi(:,:) ! cloud ice effective radius (micron)
            REAL(KIND=r8), intent(out) :: prect(:) ! surface precip rate (m/s)
            REAL(KIND=r8), intent(out) :: preci(:) ! cloud ice/snow precip rate (m/s)
            REAL(KIND=r8), intent(out) :: nevapr(:,:) ! evaporation rate of rain + snow (1/s)
            REAL(KIND=r8), intent(out) :: evapsnow(:,:) ! sublimation rate of snow (1/s)
            REAL(KIND=r8), intent(out) :: prain(:,:) ! production of rain + snow (1/s)
            REAL(KIND=r8), intent(out) :: prodsnow(:,:) ! production of snow (1/s)
            REAL(KIND=r8), intent(out) :: cmeout(:,:) ! evap/sub of cloud (1/s)
            REAL(KIND=r8), intent(out) :: deffi(:,:) ! ice effective diameter for optics (radiation) (micron)
            REAL(KIND=r8), intent(out) :: pgamrad(:,:) ! ice gamma parameter for optics (radiation) (no units)
            REAL(KIND=r8), intent(out) :: lamcrad(:,:) ! slope of droplet distribution for optics (radiation) (1/m)
            REAL(KIND=r8), intent(out) :: qsout(:,:) ! snow mixing ratio (kg/kg)
            REAL(KIND=r8), intent(out) :: dsout(:,:) ! snow diameter (m)
            REAL(KIND=r8), intent(out) :: rflx(:,:) ! grid-box average rain flux (kg m^-2 s^-1)
            REAL(KIND=r8), intent(out) :: sflx(:,:) ! grid-box average snow flux (kg m^-2 s^-1)
            REAL(KIND=r8), intent(out) :: qrout(:,:) ! grid-box average rain mixing ratio (kg/kg)
            REAL(KIND=r8), intent(out) :: reff_rain(:,:) ! rain effective radius (micron)
            REAL(KIND=r8), intent(out) :: reff_snow(:,:) ! snow effective radius (micron)
            REAL(KIND=r8), intent(out) :: qcsevap(:,:) ! cloud water evaporation due to sedimentation (1/s)
            REAL(KIND=r8), intent(out) :: qisevap(:,:) ! cloud ice sublimation due to sublimation (1/s)
            REAL(KIND=r8), intent(out) :: qvres(:,:) ! residual condensation term to ensure RH < 100% (1/s)
            REAL(KIND=r8), intent(out) :: cmeitot(:,:) ! grid-mean cloud ice sub/dep (1/s)
            REAL(KIND=r8), intent(out) :: vtrmc(:,:) ! mass-weighted cloud water fallspeed (m/s)
            REAL(KIND=r8), intent(out) :: vtrmi(:,:) ! mass-weighted cloud ice fallspeed (m/s)
            REAL(KIND=r8), intent(out) :: umr(:,:) ! mass weighted rain fallspeed (m/s)
            REAL(KIND=r8), intent(out) :: ums(:,:) ! mass weighted snow fallspeed (m/s)
            REAL(KIND=r8), intent(out) :: qcsedten(:,:) ! qc sedimentation tendency (1/s)
            REAL(KIND=r8), intent(out) :: qisedten(:,:) ! qi sedimentation tendency (1/s)
            REAL(KIND=r8), intent(out) :: qrsedten(:,:) ! qr sedimentation tendency (1/s)
            REAL(KIND=r8), intent(out) :: qssedten(:,:) ! qs sedimentation tendency (1/s)
            ! microphysical process rates for output (mixing ratio tendencies) (all have units of 1/s)
            REAL(KIND=r8), intent(out) :: pratot(:,:) ! accretion of cloud by rain
            REAL(KIND=r8), intent(out) :: prctot(:,:) ! autoconversion of cloud to rain
            REAL(KIND=r8), intent(out) :: mnuccctot(:,:) ! mixing ratio tend due to immersion freezing
            REAL(KIND=r8), intent(out) :: mnuccttot(:,:) ! mixing ratio tend due to contact freezing
            REAL(KIND=r8), intent(out) :: msacwitot(:,:) ! mixing ratio tend due to H-M splintering
            REAL(KIND=r8), intent(out) :: psacwstot(:,:) ! collection of cloud water by snow
            REAL(KIND=r8), intent(out) :: bergstot(:,:) ! bergeron process on snow
            REAL(KIND=r8), intent(out) :: bergtot(:,:) ! bergeron process on cloud ice
            REAL(KIND=r8), intent(out) :: melttot(:,:) ! melting of cloud ice
            REAL(KIND=r8), intent(out) :: homotot(:,:) ! homogeneous freezing cloud water
            REAL(KIND=r8), intent(out) :: qcrestot(:,:) ! residual cloud condensation due to removal of excess supersat
            REAL(KIND=r8), intent(out) :: prcitot(:,:) ! autoconversion of cloud ice to snow
            REAL(KIND=r8), intent(out) :: praitot(:,:) ! accretion of cloud ice by snow
            REAL(KIND=r8), intent(out) :: qirestot(:,:) ! residual ice deposition due to removal of excess supersat
            REAL(KIND=r8), intent(out) :: mnuccrtot(:,:) ! mixing ratio tendency due to heterogeneous freezing of rain to snow (1/s)
            REAL(KIND=r8), intent(out) :: pracstot(:,:) ! mixing ratio tendency due to accretion of rain by snow (1/s)
            REAL(KIND=r8), intent(out) :: meltsdttot(:,:) ! latent heating rate due to melting of snow  (W/kg)
            REAL(KIND=r8), intent(out) :: frzrdttot(:,:) ! latent heating rate due to homogeneous freezing of rain (W/kg)
            REAL(KIND=r8), intent(out) :: mnuccdtot(:,:) ! mass tendency from ice nucleation
            REAL(KIND=r8), intent(out) :: nrout(:,:) ! rain number concentration (1/m3)
            REAL(KIND=r8), intent(out) :: nsout(:,:) ! snow number concentration (1/m3)
            REAL(KIND=r8), intent(out) :: refl(:,:) ! analytic radar reflectivity
            REAL(KIND=r8), intent(out) :: arefl(:,:) ! average reflectivity will zero points outside valid range
            REAL(KIND=r8), intent(out) :: areflz(:,:) ! average reflectivity in z.
            REAL(KIND=r8), intent(out) :: frefl(:,:) ! fractional occurrence of radar reflectivity
            REAL(KIND=r8), intent(out) :: csrfl(:,:) ! cloudsat reflectivity
            REAL(KIND=r8), intent(out) :: acsrfl(:,:) ! cloudsat average
            REAL(KIND=r8), intent(out) :: fcsrfl(:,:) ! cloudsat fractional occurrence of radar reflectivity
            REAL(KIND=r8), intent(out) :: rercld(:,:) ! effective radius calculation for rain + cloud
            REAL(KIND=r8), intent(out) :: ncai(:,:) ! output number conc of ice nuclei available (1/m3)
            REAL(KIND=r8), intent(out) :: ncal(:,:) ! output number conc of CCN (1/m3)
            REAL(KIND=r8), intent(out) :: qrout2(:,:) ! copy of qrout as used to compute drout2
            REAL(KIND=r8), intent(out) :: qsout2(:,:) ! copy of qsout as used to compute dsout2
            REAL(KIND=r8), intent(out) :: nrout2(:,:) ! copy of nrout as used to compute drout2
            REAL(KIND=r8), intent(out) :: nsout2(:,:) ! copy of nsout as used to compute dsout2
            REAL(KIND=r8), intent(out) :: drout2(:,:) ! mean rain particle diameter (m)
            REAL(KIND=r8), intent(out) :: dsout2(:,:) ! mean snow particle diameter (m)
            REAL(KIND=r8), intent(out) :: freqs(:,:) ! fractional occurrence of snow
            REAL(KIND=r8), intent(out) :: freqr(:,:) ! fractional occurrence of rain
            REAL(KIND=r8), intent(out) :: nfice(:,:) ! fractional occurrence of ice
            REAL(KIND=r8), intent(out) :: qcrat(:,:) ! limiter for qc process rates (1=no limit --> 0. no qc)
            REAL(KIND=r8), intent(out) :: prer_evap(:,:)
            CHARACTER(LEN=128), intent(out) :: errstring ! output status (non-blank for error return)
            ! Tendencies calculated by external schemes that can replace MG's native
            ! process tendencies.
            ! Used with CARMA cirrus microphysics
            ! (or similar external microphysics model)
            REAL(KIND=r8), intent(in), pointer :: tnd_qsnow(:,:) ! snow mass tendency (kg/kg/s)
            REAL(KIND=r8), intent(in), pointer :: tnd_nsnow(:,:) ! snow number tendency (#/kg/s)
            REAL(KIND=r8), intent(in), pointer :: re_ice(:,:) ! ice effective radius (m)
            ! From external ice nucleation.
            REAL(KIND=r8), intent(in), pointer :: frzimm(:,:) ! Number tendency due to immersion freezing (1/cm3)
            REAL(KIND=r8), intent(in), pointer :: frzcnt(:,:) ! Number tendency due to contact freezing (1/cm3)
            REAL(KIND=r8), intent(in), pointer :: frzdep(:,:) ! Number tendency due to deposition nucleation (1/cm3)
            ! local workspace
            ! all units mks unless otherwise stated
            ! local copies of input variables
            REAL(KIND=r8) :: qc(mgncol,nlev) ! cloud liquid mixing ratio (kg/kg)
            REAL(KIND=r8) :: qi(mgncol,nlev) ! cloud ice mixing ratio (kg/kg)
            REAL(KIND=r8) :: nc(mgncol,nlev) ! cloud liquid number concentration (1/kg)
            REAL(KIND=r8) :: ni(mgncol,nlev) ! cloud liquid number concentration (1/kg)
            REAL(KIND=r8) :: qr(mgncol,nlev) ! rain mixing ratio (kg/kg)
            REAL(KIND=r8) :: qs(mgncol,nlev) ! snow mixing ratio (kg/kg)
            REAL(KIND=r8) :: nr(mgncol,nlev) ! rain number concentration (1/kg)
            REAL(KIND=r8) :: ns(mgncol,nlev) ! snow number concentration (1/kg)
            ! general purpose variables
            REAL(KIND=r8) :: deltat ! sub-time step (s)
            REAL(KIND=r8) :: mtime ! the assumed ice nucleation timescale
            ! physical properties of the air at a given point
            REAL(KIND=r8) :: rho(mgncol,nlev) ! density (kg m-3)
            REAL(KIND=r8) :: dv(mgncol,nlev) ! diffusivity of water vapor
            REAL(KIND=r8) :: mu(mgncol,nlev) ! viscosity
            REAL(KIND=r8) :: sc(mgncol,nlev) ! schmidt number
            REAL(KIND=r8) :: rhof(mgncol,nlev) ! density correction factor for fallspeed
            ! cloud fractions
            REAL(KIND=r8) :: precip_frac(mgncol,nlev) ! precip fraction assuming maximum overlap
            REAL(KIND=r8) :: cldm(mgncol,nlev) ! cloud fraction
            REAL(KIND=r8) :: icldm(mgncol,nlev) ! ice cloud fraction
            REAL(KIND=r8) :: lcldm(mgncol,nlev) ! liq cloud fraction
            ! mass mixing ratios
            REAL(KIND=r8) :: qcic(mgncol,nlev) ! in-cloud cloud liquid
            REAL(KIND=r8) :: qiic(mgncol,nlev) ! in-cloud cloud ice
            REAL(KIND=r8) :: qsic(mgncol,nlev) ! in-precip snow
            REAL(KIND=r8) :: qric(mgncol,nlev) ! in-precip rain
            ! number concentrations
            REAL(KIND=r8) :: ncic(mgncol,nlev) ! in-cloud droplet
            REAL(KIND=r8) :: niic(mgncol,nlev) ! in-cloud cloud ice
            REAL(KIND=r8) :: nsic(mgncol,nlev) ! in-precip snow
            REAL(KIND=r8) :: nric(mgncol,nlev) ! in-precip rain
            ! maximum allowed ni value
            REAL(KIND=r8) :: nimax(mgncol,nlev)
            ! Size distribution parameters for:
            ! cloud ice
            REAL(KIND=r8) :: lami(mgncol,nlev) ! slope
            REAL(KIND=r8) :: n0i(mgncol,nlev) ! intercept
            ! cloud liquid
            REAL(KIND=r8) :: lamc(mgncol,nlev) ! slope
            REAL(KIND=r8) :: pgam(mgncol,nlev) ! spectral width parameter
            ! snow
            REAL(KIND=r8) :: lams(mgncol,nlev) ! slope
            REAL(KIND=r8) :: n0s(mgncol,nlev) ! intercept
            ! rain
            REAL(KIND=r8) :: lamr(mgncol,nlev) ! slope
            REAL(KIND=r8) :: n0r(mgncol,nlev) ! intercept
            ! Rates/tendencies due to:
            ! Instantaneous snow melting
            REAL(KIND=r8) :: minstsm(mgncol,nlev) ! mass mixing ratio
            REAL(KIND=r8) :: ninstsm(mgncol,nlev) ! number concentration
            ! Instantaneous rain freezing
            REAL(KIND=r8) :: minstrf(mgncol,nlev) ! mass mixing ratio
            REAL(KIND=r8) :: ninstrf(mgncol,nlev) ! number concentration
            ! deposition of cloud ice
            REAL(KIND=r8) :: vap_dep(mgncol,nlev) ! deposition from vapor to ice PMC 12/3/12
            ! sublimation of cloud ice
            REAL(KIND=r8) :: ice_sublim(mgncol,nlev) ! sublimation from ice to vapor PMC 12/3/12
            ! ice nucleation
            REAL(KIND=r8) :: nnuccd(mgncol,nlev) ! number rate from deposition/cond.-freezing
            REAL(KIND=r8) :: mnuccd(mgncol,nlev) ! mass mixing ratio
            ! freezing of cloud water
            REAL(KIND=r8) :: mnuccc(mgncol,nlev) ! mass mixing ratio
            REAL(KIND=r8) :: nnuccc(mgncol,nlev) ! number concentration
            ! contact freezing of cloud water
            REAL(KIND=r8) :: mnucct(mgncol,nlev) ! mass mixing ratio
            REAL(KIND=r8) :: nnucct(mgncol,nlev) ! number concentration
            ! deposition nucleation in mixed-phase clouds (from external scheme)
            REAL(KIND=r8) :: mnudep(mgncol,nlev) ! mass mixing ratio
            REAL(KIND=r8) :: nnudep(mgncol,nlev) ! number concentration
            ! ice multiplication
            REAL(KIND=r8) :: msacwi(mgncol,nlev) ! mass mixing ratio
            REAL(KIND=r8) :: nsacwi(mgncol,nlev) ! number concentration
            ! autoconversion of cloud droplets
            REAL(KIND=r8) :: prc(mgncol,nlev) ! mass mixing ratio
            REAL(KIND=r8) :: nprc(mgncol,nlev) ! number concentration (rain)
            REAL(KIND=r8) :: nprc1(mgncol,nlev) ! number concentration (cloud droplets)
            ! self-aggregation of snow
            REAL(KIND=r8) :: nsagg(mgncol,nlev) ! number concentration
            ! self-collection of rain
            REAL(KIND=r8) :: nragg(mgncol,nlev) ! number concentration
            ! collection of droplets by snow
            REAL(KIND=r8) :: psacws(mgncol,nlev) ! mass mixing ratio
            REAL(KIND=r8) :: npsacws(mgncol,nlev) ! number concentration
            ! collection of rain by snow
            REAL(KIND=r8) :: pracs(mgncol,nlev) ! mass mixing ratio
            REAL(KIND=r8) :: npracs(mgncol,nlev) ! number concentration
            ! freezing of rain
            REAL(KIND=r8) :: mnuccr(mgncol,nlev) ! mass mixing ratio
            REAL(KIND=r8) :: nnuccr(mgncol,nlev) ! number concentration
            ! freezing of rain to form ice (mg add 4/26/13)
            REAL(KIND=r8) :: mnuccri(mgncol,nlev) ! mass mixing ratio
            REAL(KIND=r8) :: nnuccri(mgncol,nlev) ! number concentration
            ! accretion of droplets by rain
            REAL(KIND=r8) :: pra(mgncol,nlev) ! mass mixing ratio
            REAL(KIND=r8) :: npra(mgncol,nlev) ! number concentration
            ! autoconversion of cloud ice to snow
            REAL(KIND=r8) :: prci(mgncol,nlev) ! mass mixing ratio
            REAL(KIND=r8) :: nprci(mgncol,nlev) ! number concentration
            ! accretion of cloud ice by snow
            REAL(KIND=r8) :: prai(mgncol,nlev) ! mass mixing ratio
            REAL(KIND=r8) :: nprai(mgncol,nlev) ! number concentration
            ! evaporation of rain
            REAL(KIND=r8) :: pre(mgncol,nlev) ! mass mixing ratio
            ! sublimation of snow
            REAL(KIND=r8) :: prds(mgncol,nlev) ! mass mixing ratio
            ! number evaporation
            REAL(KIND=r8) :: nsubi(mgncol,nlev) ! cloud ice
            REAL(KIND=r8) :: nsubc(mgncol,nlev) ! droplet
            REAL(KIND=r8) :: nsubs(mgncol,nlev) ! snow
            REAL(KIND=r8) :: nsubr(mgncol,nlev) ! rain
            ! bergeron process
            REAL(KIND=r8) :: berg(mgncol,nlev) ! mass mixing ratio (cloud ice)
            REAL(KIND=r8) :: bergs(mgncol,nlev) ! mass mixing ratio (snow)
            ! fallspeeds
            ! number-weighted
            REAL(KIND=r8) :: uns(mgncol,nlev) ! snow
            REAL(KIND=r8) :: unr(mgncol,nlev) ! rain
            ! air density corrected fallspeed parameters
            REAL(KIND=r8) :: arn(mgncol,nlev) ! rain
            REAL(KIND=r8) :: asn(mgncol,nlev) ! snow
            REAL(KIND=r8) :: acn(mgncol,nlev) ! cloud droplet
            REAL(KIND=r8) :: ain(mgncol,nlev) ! cloud ice
            ! Mass of liquid droplets used with external heterogeneous freezing.
            REAL(KIND=r8) :: mi0l(mgncol)
            ! saturation vapor pressures
            REAL(KIND=r8) :: esl(mgncol,nlev) ! liquid
            REAL(KIND=r8) :: esi(mgncol,nlev) ! ice
            REAL(KIND=r8) :: esn ! checking for RH after rain evap
            ! saturation vapor mixing ratios
            REAL(KIND=r8) :: qvl(mgncol,nlev) ! liquid
            REAL(KIND=r8) :: qvi(mgncol,nlev) ! ice
            REAL(KIND=r8) :: qvn ! checking for RH after rain evap
            ! relative humidity
            REAL(KIND=r8) :: relhum(mgncol,nlev)
            ! parameters for cloud water and cloud ice sedimentation calculations
            REAL(KIND=r8) :: fc(nlev)
            REAL(KIND=r8) :: fnc(nlev)
            REAL(KIND=r8) :: fi(nlev)
            REAL(KIND=r8) :: fni(nlev)
            REAL(KIND=r8) :: fr(nlev)
            REAL(KIND=r8) :: fnr(nlev)
            REAL(KIND=r8) :: fs(nlev)
            REAL(KIND=r8) :: fns(nlev)
            REAL(KIND=r8) :: faloutc(nlev)
            REAL(KIND=r8) :: faloutnc(nlev)
            REAL(KIND=r8) :: falouti(nlev)
            REAL(KIND=r8) :: faloutni(nlev)
            REAL(KIND=r8) :: faloutr(nlev)
            REAL(KIND=r8) :: faloutnr(nlev)
            REAL(KIND=r8) :: falouts(nlev)
            REAL(KIND=r8) :: faloutns(nlev)
            REAL(KIND=r8) :: faltndc
            REAL(KIND=r8) :: faltndnc
            REAL(KIND=r8) :: faltndi
            REAL(KIND=r8) :: faltndni
            REAL(KIND=r8) :: faltndqie
            REAL(KIND=r8) :: faltndqce
            REAL(KIND=r8) :: faltndr
            REAL(KIND=r8) :: faltndnr
            REAL(KIND=r8) :: faltnds
            REAL(KIND=r8) :: faltndns
            REAL(KIND=r8) :: rainrt(mgncol,nlev) ! rain rate for reflectivity calculation
            ! dummy variables
            REAL(KIND=r8) :: dum
            REAL(KIND=r8) :: dum1
            REAL(KIND=r8) :: dum2
            ! dummies for checking RH
            REAL(KIND=r8) :: qtmp
            REAL(KIND=r8) :: ttmp
            ! dummies for conservation check
            REAL(KIND=r8) :: ratio
            REAL(KIND=r8) :: tmpfrz
            ! dummies for in-cloud variables
            REAL(KIND=r8) :: dumc(mgncol,nlev) ! qc
            REAL(KIND=r8) :: dumnc(mgncol,nlev) ! nc
            REAL(KIND=r8) :: dumi(mgncol,nlev) ! qi
            REAL(KIND=r8) :: dumni(mgncol,nlev) ! ni
            REAL(KIND=r8) :: dumr(mgncol,nlev) ! rain mixing ratio
            REAL(KIND=r8) :: dumnr(mgncol,nlev) ! rain number concentration
            REAL(KIND=r8) :: dums(mgncol,nlev) ! snow mixing ratio
            REAL(KIND=r8) :: dumns(mgncol,nlev) ! snow number concentration
            ! Array dummy variable
            REAL(KIND=r8) :: dum_2d(mgncol,nlev)
            ! loop array variables
            ! "i" and "k" are column/level iterators for internal (MG) variables
            ! "n" is used for other looping (currently just sedimentation)
            INTEGER :: k
            INTEGER :: i
            INTEGER :: n
            ! number of sub-steps for loops over "n" (for sedimentation)
            INTEGER :: nstep
            !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            ! default return error message
            errstring = ' '
            IF (.not. (do_cldice .or.        (associated(tnd_qsnow) .and. associated(tnd_nsnow) .and. associated(re_ice)))) THEN
                errstring = "MG's native cloud ice processes are disabled, but no replacement values were passed in."
            END IF 
            IF (use_hetfrz_classnuc .and. (.not.        (associated(frzimm) .and. associated(frzcnt) .and. associated(frzdep)))) THEN
                errstring = "External heterogeneous freezing is enabled, but the required tendencies were not all passed in."
            END IF 
            ! Process inputs
            ! assign variable deltat to deltatin
            deltat = deltatin
            ! Copies of input concentrations that may be changed internally.
            qc = qcn
            nc = ncn
            qi = qin
            ni = nin
            qr = qrn
            nr = nrn
            qs = qsn
            ns = nsn
            ! cldn: used to set cldm, unused for subcolumns
            ! liqcldf: used to set lcldm, unused for subcolumns
            ! icecldf: used to set icldm, unused for subcolumns
            IF (microp_uniform) THEN
                ! subcolumns, set cloud fraction variables to one
                ! if cloud water or ice is present, if not present
                ! set to mincld (mincld used instead of zero, to prevent
                ! possible division by zero errors).
                WHERE ( qc >= qsmall )
                    lcldm = 1._r8
                    ELSEWHERE
                    lcldm = mincld
                END WHERE 
                WHERE ( qi >= qsmall )
                    icldm = 1._r8
                    ELSEWHERE
                    icldm = mincld
                END WHERE 
                cldm = max(icldm, lcldm)
                ELSE
                ! get cloud fraction, check for minimum
                cldm = max(cldn,mincld)
                lcldm = max(liqcldf,mincld)
                icldm = max(icecldf,mincld)
            END IF 
            ! Initialize local variables
            ! local physical properties
            rho = p/(r*t)
            dv = 8.794e-5_r8 * t**1.81_r8 / p
            mu = 1.496e-6_r8 * t**1.5_r8 / (t + 120._r8)
            sc = mu/(rho*dv)
            ! air density adjustment for fallspeed parameters
            ! includes air density correction factor to the
            ! power of 0.54 following Heymsfield and Bansemer 2007
            rhof = (rhosu/rho)**0.54_r8
            arn = ar*rhof
            asn = as*rhof
            acn = g*rhow/(18._r8*mu)
            ain = ai*(rhosu/rho)**0.35_r8
            !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            ! Get humidity and saturation vapor pressures
            DO k=1,nlev
                DO i=1,mgncol
                    CALL qsat_water(t(i,k), p(i,k), esl(i,k), qvl(i,k))
                    ! make sure when above freezing that esi=esl, not active yet
                    IF (t(i,k) >= tmelt) THEN
                        esi(i,k) = esl(i,k)
                        qvi(i,k) = qvl(i,k)
                        ELSE
                        CALL qsat_ice(t(i,k), p(i,k), esi(i,k), qvi(i,k))
                    END IF 
                END DO 
            END DO 
            relhum = q / max(qvl, qsmall)
            !===============================================
            ! set mtime here to avoid answer-changing
            mtime = deltat
            ! initialize microphysics output
            qcsevap = 0._r8
            qisevap = 0._r8
            qvres = 0._r8
            cmeitot = 0._r8
            vtrmc = 0._r8
            vtrmi = 0._r8
            qcsedten = 0._r8
            qisedten = 0._r8
            qrsedten = 0._r8
            qssedten = 0._r8
            pratot = 0._r8
            prctot = 0._r8
            mnuccctot = 0._r8
            mnuccttot = 0._r8
            msacwitot = 0._r8
            psacwstot = 0._r8
            bergstot = 0._r8
            bergtot = 0._r8
            melttot = 0._r8
            homotot = 0._r8
            qcrestot = 0._r8
            prcitot = 0._r8
            praitot = 0._r8
            qirestot = 0._r8
            mnuccrtot = 0._r8
            pracstot = 0._r8
            meltsdttot = 0._r8
            frzrdttot = 0._r8
            mnuccdtot = 0._r8
            rflx = 0._r8
            sflx = 0._r8
            ! initialize precip output
            qrout = 0._r8
            qsout = 0._r8
            nrout = 0._r8
            nsout = 0._r8
            ! for refl calc
            rainrt = 0._r8
            ! initialize rain size
            rercld = 0._r8
            qcsinksum_rate1ord = 0._r8
            ! initialize variables for trop_mozart
            nevapr = 0._r8
            prer_evap = 0._r8
            evapsnow = 0._r8
            prain = 0._r8
            prodsnow = 0._r8
            cmeout = 0._r8
            precip_frac = mincld
            lamc = 0._r8
            ! initialize microphysical tendencies
            tlat = 0._r8
            qvlat = 0._r8
            qctend = 0._r8
            qitend = 0._r8
            qstend = 0._r8
            qrtend = 0._r8
            nctend = 0._r8
            nitend = 0._r8
            nrtend = 0._r8
            nstend = 0._r8
            ! initialize in-cloud and in-precip quantities to zero
            qcic = 0._r8
            qiic = 0._r8
            qsic = 0._r8
            qric = 0._r8
            ncic = 0._r8
            niic = 0._r8
            nsic = 0._r8
            nric = 0._r8
            ! initialize precip at surface
            prect = 0._r8
            preci = 0._r8
            ! initialize precip fallspeeds to zero
            ums = 0._r8
            uns = 0._r8
            umr = 0._r8
            unr = 0._r8
            ! initialize limiter for output
            qcrat = 1._r8
            ! Many outputs have to be initialized here at the top to work around
            ! ifort problems, even if they are always overwritten later.
            effc = 10._r8
            lamcrad = 0._r8
            pgamrad = 0._r8
            effc_fn = 10._r8
            effi = 25._r8
            deffi = 50._r8
            qrout2 = 0._r8
            nrout2 = 0._r8
            drout2 = 0._r8
            qsout2 = 0._r8
            nsout2 = 0._r8
            dsout = 0._r8
            dsout2 = 0._r8
            freqr = 0._r8
            freqs = 0._r8
            reff_rain = 0._r8
            reff_snow = 0._r8
            refl = -9999._r8
            arefl = 0._r8
            areflz = 0._r8
            frefl = 0._r8
            csrfl = 0._r8
            acsrfl = 0._r8
            fcsrfl = 0._r8
            ncal = 0._r8
            ncai = 0._r8
            nfice = 0._r8
            !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
            ! droplet activation
            ! get provisional droplet number after activation. This is used for
            ! all microphysical process calculations, for consistency with update of
            ! droplet mass before microphysics
            ! calculate potential for droplet activation if cloud water is present
            ! tendency from activation (npccn) is read in from companion routine
            ! output activated liquid and ice (convert from #/kg -> #/m3)
            !--------------------------------------------------
            WHERE ( qc >= qsmall )
                nc = max(nc + npccn*deltat, 0._r8)
                ncal = nc*rho/lcldm ! sghan minimum in #/cm3
                ELSEWHERE
                ncal = 0._r8
            END WHERE 
            WHERE ( t < icenuct )
                ncai = naai*rho
                ELSEWHERE
                ncai = 0._r8
            END WHERE 
            !===============================================
            ! ice nucleation if activated nuclei exist at t<-5C AND rhmini + 5%
            !-------------------------------------------------------
            IF (do_cldice) THEN
                WHERE ( naai > 0._r8 .and. t < icenuct .and.           relhum*esl/esi > rhmini+0.05_r8 )
                    !if NAAI > 0. then set numice = naai (as before)
                    !note: this is gridbox averaged
                    nnuccd = (naai-ni/icldm)/mtime*icldm
                    nnuccd = max(nnuccd,0._r8)
                    nimax = naai*icldm
                    !Calc mass of new particles using new crystal mass...
                    !also this will be multiplied by mtime as nnuccd is...
                    mnuccd = nnuccd * mi0
                    ELSEWHERE
                    nnuccd = 0._r8
                    nimax = 0._r8
                    mnuccd = 0._r8
                END WHERE 
            END IF 
            !=============================================================================
            pre_vert_loop: DO k=1,nlev
                pre_col_loop: DO i=1,mgncol
                    ! calculate instantaneous precip processes (melting and homogeneous freezing)
                    ! melting of snow at +2 C
                    IF (t(i,k) > snowmelt) THEN
                        IF (qs(i,k) > 0._r8) THEN
                            ! make sure melting snow doesn't reduce temperature below threshold
                            dum = -xlf/cpp*qs(i,k)
                            IF (t(i,k)+dum < snowmelt) THEN
                                dum = (t(i,k)-snowmelt)*cpp/xlf
                                dum = dum/qs(i,k)
                                dum = max(0._r8,dum)
                                dum = min(1._r8,dum)
                                ELSE
                                dum = 1._r8
                            END IF 
                            minstsm(i,k) = dum*qs(i,k)
                            ninstsm(i,k) = dum*ns(i,k)
                            dum1 = -xlf*minstsm(i,k)/deltat
                            tlat(i,k) = tlat(i,k)+dum1
                            meltsdttot(i,k) = meltsdttot(i,k) + dum1
                            qs(i,k) = max(qs(i,k) - minstsm(i,k), 0._r8)
                            ns(i,k) = max(ns(i,k) - ninstsm(i,k), 0._r8)
                            qr(i,k) = max(qr(i,k) + minstsm(i,k), 0._r8)
                            nr(i,k) = max(nr(i,k) + ninstsm(i,k), 0._r8)
                        END IF 
                    END IF 
                    ! freezing of rain at -5 C
                    IF (t(i,k) < rainfrze) THEN
                        IF (qr(i,k) > 0._r8) THEN
                            ! make sure freezing rain doesn't increase temperature above threshold
                            dum = xlf/cpp*qr(i,k)
                            IF (t(i,k)+dum > rainfrze) THEN
                                dum = -(t(i,k)-rainfrze)*cpp/xlf
                                dum = dum/qr(i,k)
                                dum = max(0._r8,dum)
                                dum = min(1._r8,dum)
                                ELSE
                                dum = 1._r8
                            END IF 
                            minstrf(i,k) = dum*qr(i,k)
                            ninstrf(i,k) = dum*nr(i,k)
                            ! heating tendency
                            dum1 = xlf*minstrf(i,k)/deltat
                            tlat(i,k) = tlat(i,k)+dum1
                            frzrdttot(i,k) = frzrdttot(i,k) + dum1
                            qr(i,k) = max(qr(i,k) - minstrf(i,k), 0._r8)
                            nr(i,k) = max(nr(i,k) - ninstrf(i,k), 0._r8)
                            qs(i,k) = max(qs(i,k) + minstrf(i,k), 0._r8)
                            ns(i,k) = max(ns(i,k) + ninstrf(i,k), 0._r8)
                        END IF 
                    END IF 
                    ! obtain in-cloud values of cloud water/ice mixing ratios and number concentrations
                    !-------------------------------------------------------
                    ! for microphysical process calculations
                    ! units are kg/kg for mixing ratio, 1/kg for number conc
                    IF (qc(i,k).ge.qsmall) THEN
                        ! limit in-cloud values to 0.005 kg/kg
                        qcic(i,k) = min(qc(i,k)/lcldm(i,k),5.e-3_r8)
                        ncic(i,k) = max(nc(i,k)/lcldm(i,k),0._r8)
                        ! specify droplet concentration
                        IF (nccons) THEN
                            ncic(i,k) = ncnst/rho(i,k)
                        END IF 
                        ELSE
                        qcic(i,k) = 0._r8
                        ncic(i,k) = 0._r8
                    END IF 
                    IF (qi(i,k).ge.qsmall) THEN
                        ! limit in-cloud values to 0.005 kg/kg
                        qiic(i,k) = min(qi(i,k)/icldm(i,k),5.e-3_r8)
                        niic(i,k) = max(ni(i,k)/icldm(i,k),0._r8)
                        ! switch for specification of cloud ice number
                        IF (nicons) THEN
                            niic(i,k) = ninst/rho(i,k)
                        END IF 
                        ELSE
                        qiic(i,k) = 0._r8
                        niic(i,k) = 0._r8
                    END IF 
                END DO pre_col_loop
            END DO pre_vert_loop
            !========================================================================
            ! for sub-columns cldm has already been set to 1 if cloud
            ! water or ice is present, so precip_frac will be correctly set below
            ! and nothing extra needs to be done here
            precip_frac = cldm
            micro_vert_loop: DO k=1,nlev
                IF (trim(micro_mg_precip_frac_method) == 'in_cloud') THEN
                    IF (k /= 1) THEN
                        WHERE ( qc(:,k) < qsmall .and. qi(:,k) < qsmall )
                            precip_frac(:,k) = precip_frac(:,k-1)
                        END WHERE 
                    END IF 
                    ELSE IF (trim(micro_mg_precip_frac_method) == 'max_overlap') THEN
                    ! calculate precip fraction based on maximum overlap assumption
                    ! if rain or snow mix ratios are smaller than threshold,
                    ! then leave precip_frac as cloud fraction at current level
                    IF (k /= 1) THEN
                        WHERE ( qr(:,k-1) >= qsmall .or. qs(:,k-1) >= qsmall )
                            precip_frac(:,k) = max(precip_frac(:,k-1),precip_frac(:,k))
                        END WHERE 
                    END IF 
                END IF 
                DO i = 1, mgncol
                    !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                    ! get size distribution parameters based on in-cloud cloud water
                    ! these calculations also ensure consistency between number and mixing ratio
                    !cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                    ! cloud liquid
                    !-------------------------------------------
                    CALL size_dist_param_liq(mg_liq_props, qcic(i,k), ncic(i,k), rho(i,k), pgam(i,k), lamc(i,k))
                END DO 
                !========================================================================
                ! autoconversion of cloud liquid water to rain
                ! formula from Khrouditnov and Kogan (2000), modified for sub-grid distribution of qc
                ! minimum qc of 1 x 10^-8 prevents floating point error
                CALL kk2000_liq_autoconversion(microp_uniform, qcic(:,k), ncic(:,k), rho(:,k), relvar(:,k), prc(:,k), nprc(:,k), &
                nprc1(:,k))
                ! assign qric based on prognostic qr, using assumed precip fraction
                ! note: this could be moved above for consistency with qcic and qiic calculations
                qric(:,k) = qr(:,k)/precip_frac(:,k)
                nric(:,k) = nr(:,k)/precip_frac(:,k)
                ! limit in-precip mixing ratios to 10 g/kg
                qric(:,k) = min(qric(:,k),0.01_r8)
                ! add autoconversion to precip from above to get provisional rain mixing ratio
                ! and number concentration (qric and nric)
                WHERE ( qric(:,k).lt.qsmall )
                    qric(:,k) = 0._r8
                    nric(:,k) = 0._r8
                END WHERE 
                ! make sure number concentration is a positive number to avoid
                ! taking root of negative later
                nric(:,k) = max(nric(:,k),0._r8)
                ! Get size distribution parameters for cloud ice
                CALL size_dist_param_basic(mg_ice_props, qiic(:,k), niic(:,k), lami(:,k), n0i(:,k))
                !.......................................................................
                ! Autoconversion of cloud ice to snow
                ! similar to Ferrier (1994)
                IF (do_cldice) THEN
                    CALL ice_autoconversion(t(:,k), qiic(:,k), lami(:,k), n0i(:,k), dcs, prci(:,k), nprci(:,k))
                    ELSE
                    ! Add in the particles that we have already converted to snow, and
                    ! don't do any further autoconversion of ice.
                    prci(:,k) = tnd_qsnow(:,k) / cldm(:,k)
                    nprci(:,k) = tnd_nsnow(:,k) / cldm(:,k)
                END IF 
                ! note, currently we don't have this
                ! inside the do_cldice block, should be changed later
                ! assign qsic based on prognostic qs, using assumed precip fraction
                qsic(:,k) = qs(:,k)/precip_frac(:,k)
                nsic(:,k) = ns(:,k)/precip_frac(:,k)
                ! limit in-precip mixing ratios to 10 g/kg
                qsic(:,k) = min(qsic(:,k),0.01_r8)
                ! if precip mix ratio is zero so should number concentration
                WHERE ( qsic(:,k) < qsmall )
                    qsic(:,k) = 0._r8
                    nsic(:,k) = 0._r8
                END WHERE 
                ! make sure number concentration is a positive number to avoid
                ! taking root of negative later
                nsic(:,k) = max(nsic(:,k),0._r8)
                !.......................................................................
                ! get size distribution parameters for precip
                !......................................................................
                ! rain
                CALL size_dist_param_basic(mg_rain_props, qric(:,k), nric(:,k), lamr(:,k), n0r(:,k))
                WHERE ( lamr(:,k) >= qsmall )
                    ! provisional rain number and mass weighted mean fallspeed (m/s)
                    unr(:,k) = min(arn(:,k)*gamma_br_plus1/lamr(:,k)**br,9.1_r8*rhof(:,k))
                    umr(:,k) = min(arn(:,k)*gamma_br_plus4/(6._r8*lamr(:,k)**br),9.1_r8*rhof(:,k))
                    ELSEWHERE
                    umr(:,k) = 0._r8
                    unr(:,k) = 0._r8
                END WHERE 
                !......................................................................
                ! snow
                CALL size_dist_param_basic(mg_snow_props, qsic(:,k), nsic(:,k), lams(:,k), n0s(:,k))
                WHERE ( lams(:,k) > 0._r8 )
                    ! provisional snow number and mass weighted mean fallspeed (m/s)
                    ums(:,k) = min(asn(:,k)*gamma_bs_plus4/(6._r8*lams(:,k)**bs),1.2_r8*rhof(:,k))
                    uns(:,k) = min(asn(:,k)*gamma_bs_plus1/lams(:,k)**bs,1.2_r8*rhof(:,k))
                    ELSEWHERE
                    ums(:,k) = 0._r8
                    uns(:,k) = 0._r8
                END WHERE 
                IF (do_cldice) THEN
                    IF (.not. use_hetfrz_classnuc) THEN
                        ! heterogeneous freezing of cloud water
                        !----------------------------------------------
                        CALL immersion_freezing(microp_uniform, t(:,k), pgam(:,k), lamc(:,k), qcic(:,k), ncic(:,k), relvar(:,k), &
                        mnuccc(:,k), nnuccc(:,k))
                        ! make sure number of droplets frozen does not exceed available ice nuclei concentration
                        ! this prevents 'runaway' droplet freezing
                        WHERE ( qcic(:,k).ge.qsmall .and. t(:,k).lt.269.15_r8 )
                            WHERE ( nnuccc(:,k)*lcldm(:,k).gt.nnuccd(:,k) )
                                ! scale mixing ratio of droplet freezing with limit
                                mnuccc(:,k) = mnuccc(:,k)*(nnuccd(:,k)/(nnuccc(:,k)*lcldm(:,k)))
                                nnuccc(:,k) = nnuccd(:,k)/lcldm(:,k)
                            END WHERE 
                        END WHERE 
                        CALL contact_freezing(microp_uniform, t(:,k), p(:,k), rndst(:,k,:), nacon(:,k,:), pgam(:,k), lamc(:,k), &
                        qcic(:,k), ncic(:,k), relvar(:,k), mnucct(:,k), nnucct(:,k))
                        mnudep(:,k) = 0._r8
                        nnudep(:,k) = 0._r8
                        ELSE
                        ! Mass of droplets frozen is the average droplet mass, except
                        ! with two limiters: concentration must be at least 1/cm^3, and
                        ! mass must be at least the minimum defined above.
                        mi0l = qcic(:,k)/max(ncic(:,k), 1.0e6_r8/rho(:,k))
                        mi0l = max(mi0l_min, mi0l)
                        WHERE ( qcic(:,k) >= qsmall )
                            nnuccc(:,k) = frzimm(:,k)*1.0e6_r8/rho(:,k)
                            mnuccc(:,k) = nnuccc(:,k)*mi0l
                            nnucct(:,k) = frzcnt(:,k)*1.0e6_r8/rho(:,k)
                            mnucct(:,k) = nnucct(:,k)*mi0l
                            nnudep(:,k) = frzdep(:,k)*1.0e6_r8/rho(:,k)
                            mnudep(:,k) = nnudep(:,k)*mi0
                            ELSEWHERE
                            nnuccc(:,k) = 0._r8
                            mnuccc(:,k) = 0._r8
                            nnucct(:,k) = 0._r8
                            mnucct(:,k) = 0._r8
                            nnudep(:,k) = 0._r8
                            mnudep(:,k) = 0._r8
                        END WHERE 
                    END IF 
                    ELSE
                    mnuccc(:,k) = 0._r8
                    nnuccc(:,k) = 0._r8
                    mnucct(:,k) = 0._r8
                    nnucct(:,k) = 0._r8
                    mnudep(:,k) = 0._r8
                    nnudep(:,k) = 0._r8
                END IF 
                CALL snow_self_aggregation(t(:,k), rho(:,k), asn(:,k), rhosn, qsic(:,k), nsic(:,k), nsagg(:,k))
                CALL accrete_cloud_water_snow(t(:,k), rho(:,k), asn(:,k), uns(:,k), mu(:,k), qcic(:,k), ncic(:,k), qsic(:,k), &
                pgam(:,k), lamc(:,k), lams(:,k), n0s(:,k), psacws(:,k), npsacws(:,k))
                IF (do_cldice) THEN
                    CALL secondary_ice_production(t(:,k), psacws(:,k), msacwi(:,k), nsacwi(:,k))
                    ELSE
                    nsacwi(:,k) = 0.0_r8
                    msacwi(:,k) = 0.0_r8
                END IF 
                CALL accrete_rain_snow(t(:,k), rho(:,k), umr(:,k), ums(:,k), unr(:,k), uns(:,k), qric(:,k), qsic(:,k), lamr(:,k), &
                n0r(:,k), lams(:,k), n0s(:,k), pracs(:,k), npracs(:,k))
                CALL heterogeneous_rain_freezing(t(:,k), qric(:,k), nric(:,k), lamr(:,k), mnuccr(:,k), nnuccr(:,k))
                CALL accrete_cloud_water_rain(microp_uniform, qric(:,k), qcic(:,k), ncic(:,k), relvar(:,k), accre_enhan(:,k), pra(&
                :,k), npra(:,k))
                CALL self_collection_rain(rho(:,k), qric(:,k), nric(:,k), nragg(:,k))
                IF (do_cldice) THEN
                    CALL accrete_cloud_ice_snow(t(:,k), rho(:,k), asn(:,k), qiic(:,k), niic(:,k), qsic(:,k), lams(:,k), n0s(:,k), &
                    prai(:,k), nprai(:,k))
                    ELSE
                    prai(:,k) = 0._r8
                    nprai(:,k) = 0._r8
                END IF 
                CALL evaporate_sublimate_precip(t(:,k), rho(:,k), dv(:,k), mu(:,k), sc(:,k), q(:,k), qvl(:,k), qvi(:,k), lcldm(:,&
                k), precip_frac(:,k), arn(:,k), asn(:,k), qcic(:,k), qiic(:,k), qric(:,k), qsic(:,k), lamr(:,k), n0r(:,k), lams(:,&
                k), n0s(:,k), pre(:,k), prds(:,k))
                CALL bergeron_process_snow(t(:,k), rho(:,k), dv(:,k), mu(:,k), sc(:,k), qvl(:,k), qvi(:,k), asn(:,k), qcic(:,k), &
                qsic(:,k), lams(:,k), n0s(:,k), bergs(:,k))
                bergs(:,k) = bergs(:,k)*micro_mg_berg_eff_factor
                !+++PMC 12/3/12 - NEW VAPOR DEP/SUBLIMATION GOES HERE!!!
                IF (do_cldice) THEN
                    CALL ice_deposition_sublimation(t(:,k), q(:,k), qi(:,k), ni(:,k), icldm(:,k), rho(:,k), dv(:,k), qvl(:,k), &
                    qvi(:,k), berg(:,k), vap_dep(:,k), ice_sublim(:,k))
                    berg(:,k) = berg(:,k)*micro_mg_berg_eff_factor
                    WHERE ( vap_dep(:,k) < 0._r8 .and. qi(:,k) > qsmall .and. icldm(:,k) > mincld )
                        nsubi(:,k) = vap_dep(:,k) / qi(:,k) * ni(:,k) / icldm(:,k)
                        ELSEWHERE
                        nsubi(:,k) = 0._r8
                    END WHERE 
                    ! bergeron process should not reduce nc unless
                    ! all ql is removed (which is handled elsewhere)
                    !in fact, nothing in this entire file makes nsubc nonzero.
                    nsubc(:,k) = 0._r8
                END IF  !do_cldice
                !---PMC 12/3/12
                DO i=1,mgncol
                    ! conservation to ensure no negative values of cloud water/precipitation
                    ! in case microphysical process rates are large
                    !===================================================================
                    ! note: for check on conservation, processes are multiplied by omsm
                    ! to prevent problems due to round off error
                    ! conservation of qc
                    !-------------------------------------------------------------------
                    dum = ((prc(i,k)+pra(i,k)+mnuccc(i,k)+mnucct(i,k)+msacwi(i,k)+              psacws(i,k)+bergs(i,k))*lcldm(i,k)&
                    +berg(i,k))*deltat
                    IF (dum.gt.qc(i,k)) THEN
                        ratio = qc(i,k)/deltat/((prc(i,k)+pra(i,k)+mnuccc(i,k)+mnucct(i,k)+                 msacwi(i,k)+psacws(i,&
                        k)+bergs(i,k))*lcldm(i,k)+berg(i,k))*omsm
                        prc(i,k) = prc(i,k)*ratio
                        pra(i,k) = pra(i,k)*ratio
                        mnuccc(i,k) = mnuccc(i,k)*ratio
                        mnucct(i,k) = mnucct(i,k)*ratio
                        msacwi(i,k) = msacwi(i,k)*ratio
                        psacws(i,k) = psacws(i,k)*ratio
                        bergs(i,k) = bergs(i,k)*ratio
                        berg(i,k) = berg(i,k)*ratio
                        qcrat(i,k) = ratio
                        ELSE
                        qcrat(i,k) = 1._r8
                    END IF 
                    !PMC 12/3/12: ratio is also frac of step w/ liquid.
                    !thus we apply berg for "ratio" of timestep and vapor
                    !deposition for the remaining frac of the timestep.
                    IF (qc(i,k) >= qsmall) THEN
                        vap_dep(i,k) = vap_dep(i,k)*(1._r8-qcrat(i,k))
                    END IF 
                END DO 
                DO i=1,mgncol
                    !=================================================================
                    ! apply limiter to ensure that ice/snow sublimation and rain evap
                    ! don't push conditions into supersaturation, and ice deposition/nucleation don't
                    ! push conditions into sub-saturation
                    ! note this is done after qc conservation since we don't know how large
                    ! vap_dep is before then
                    ! estimates are only approximate since other process terms haven't been limited
                    ! for conservation yet
                    ! first limit ice deposition/nucleation vap_dep + mnuccd
                    dum1 = vap_dep(i,k) + mnuccd(i,k)
                    IF (dum1 > 1.e-20_r8) THEN
                        dum = (q(i,k)-qvi(i,k))/(1._r8 + xxls_squared*qvi(i,k)/(cpp*rv*t(i,k)**2))/deltat
                        dum = max(dum,0._r8)
                        IF (dum1 > dum) THEN
                            ! Allocate the limited "dum" tendency to mnuccd and vap_dep
                            ! processes. Don't divide by cloud fraction; these are grid-
                            ! mean rates.
                            dum1 = mnuccd(i,k) / (vap_dep(i,k)+mnuccd(i,k))
                            mnuccd(i,k) = dum*dum1
                            vap_dep(i,k) = dum - mnuccd(i,k)
                        END IF 
                    END IF 
                END DO 
                DO i=1,mgncol
                    !===================================================================
                    ! conservation of nc
                    !-------------------------------------------------------------------
                    dum = (nprc1(i,k)+npra(i,k)+nnuccc(i,k)+nnucct(i,k)+              npsacws(i,k)-nsubc(i,k))*lcldm(i,k)*deltat
                    IF (dum.gt.nc(i,k)) THEN
                        ratio = nc(i,k)/deltat/((nprc1(i,k)+npra(i,k)+nnuccc(i,k)+nnucct(i,k)+                npsacws(i,k)-nsubc(&
                        i,k))*lcldm(i,k))*omsm
                        nprc1(i,k) = nprc1(i,k)*ratio
                        npra(i,k) = npra(i,k)*ratio
                        nnuccc(i,k) = nnuccc(i,k)*ratio
                        nnucct(i,k) = nnucct(i,k)*ratio
                        npsacws(i,k) = npsacws(i,k)*ratio
                        nsubc(i,k) = nsubc(i,k)*ratio
                    END IF 
                    mnuccri(i,k) = 0._r8
                    nnuccri(i,k) = 0._r8
                    IF (do_cldice) THEN
                        ! freezing of rain to produce ice if mean rain size is smaller than Dcs
                        IF (lamr(i,k) > qsmall .and. 1._r8/lamr(i,k) < dcs) THEN
                            mnuccri(i,k) = mnuccr(i,k)
                            nnuccri(i,k) = nnuccr(i,k)
                            mnuccr(i,k) = 0._r8
                            nnuccr(i,k) = 0._r8
                        END IF 
                    END IF 
                END DO 
                DO i=1,mgncol
                    ! conservation of rain mixing ratio
                    !-------------------------------------------------------------------
                    dum = ((-pre(i,k)+pracs(i,k)+mnuccr(i,k)+mnuccri(i,k))*precip_frac(i,k)-              (pra(i,k)+prc(i,k))&
                    *lcldm(i,k))*deltat
                    ! note that qrtend is included below because of instantaneous freezing/melt
                    IF (dum.gt.qr(i,k).and.              (-pre(i,k)+pracs(i,k)+mnuccr(i,k)+mnuccri(i,k)).ge.qsmall) THEN
                        ratio = (qr(i,k)/deltat+(pra(i,k)+prc(i,k))*lcldm(i,k))/                   precip_frac(i,k)/(-pre(i,k)&
                        +pracs(i,k)+mnuccr(i,k)+mnuccri(i,k))*omsm
                        pre(i,k) = pre(i,k)*ratio
                        pracs(i,k) = pracs(i,k)*ratio
                        mnuccr(i,k) = mnuccr(i,k)*ratio
                        mnuccri(i,k) = mnuccri(i,k)*ratio
                    END IF 
                END DO 
                DO i=1,mgncol
                    ! conservation of rain number
                    !-------------------------------------------------------------------
                    ! Add evaporation of rain number.
                    IF (pre(i,k) < 0._r8) THEN
                        dum = pre(i,k)*deltat/qr(i,k)
                        dum = max(-1._r8,dum)
                        nsubr(i,k) = dum*nr(i,k)/deltat
                        ELSE
                        nsubr(i,k) = 0._r8
                    END IF 
                END DO 
                DO i=1,mgncol
                    dum = ((-nsubr(i,k)+npracs(i,k)+nnuccr(i,k)+nnuccri(i,k)-nragg(i,k))*precip_frac(i,k)-              nprc(i,k)&
                    *lcldm(i,k))*deltat
                    IF (dum.gt.nr(i,k)) THEN
                        ratio = (nr(i,k)/deltat+nprc(i,k)*lcldm(i,k)/precip_frac(i,k))/                 (-nsubr(i,k)+npracs(i,k)&
                        +nnuccr(i,k)+nnuccri(i,k)-nragg(i,k))*omsm
                        nragg(i,k) = nragg(i,k)*ratio
                        npracs(i,k) = npracs(i,k)*ratio
                        nnuccr(i,k) = nnuccr(i,k)*ratio
                        nsubr(i,k) = nsubr(i,k)*ratio
                        nnuccri(i,k) = nnuccri(i,k)*ratio
                    END IF 
                END DO 
                IF (do_cldice) THEN
                    DO i=1,mgncol
                        ! conservation of qi
                        !-------------------------------------------------------------------
                        dum = ((-mnuccc(i,k)-mnucct(i,k)-mnudep(i,k)-msacwi(i,k))*lcldm(i,k)+(prci(i,k)+                 prai(i,k)&
                        )*icldm(i,k)-mnuccri(i,k)*precip_frac(i,k)                 -ice_sublim(i,k)-vap_dep(i,k)-berg(i,k)-mnuccd(&
                        i,k))*deltat
                        IF (dum.gt.qi(i,k)) THEN
                            ratio = (qi(i,k)/deltat+vap_dep(i,k)+berg(i,k)+mnuccd(i,k)+                    (mnuccc(i,k)+mnucct(i,&
                            k)+mnudep(i,k)+msacwi(i,k))*lcldm(i,k)+                    mnuccri(i,k)*precip_frac(i,k))/            &
                                    ((prci(i,k)+prai(i,k))*icldm(i,k)-ice_sublim(i,k))*omsm
                            prci(i,k) = prci(i,k)*ratio
                            prai(i,k) = prai(i,k)*ratio
                            ice_sublim(i,k) = ice_sublim(i,k)*ratio
                        END IF 
                    END DO 
                END IF 
                IF (do_cldice) THEN
                    DO i=1,mgncol
                        ! conservation of ni
                        !-------------------------------------------------------------------
                        IF (use_hetfrz_classnuc) THEN
                            tmpfrz = nnuccc(i,k)
                            ELSE
                            tmpfrz = 0._r8
                        END IF 
                        dum = ((-nnucct(i,k)-tmpfrz-nnudep(i,k)-nsacwi(i,k))*lcldm(i,k)+(nprci(i,k)+                 nprai(i,k)&
                        -nsubi(i,k))*icldm(i,k)-nnuccri(i,k)*precip_frac(i,k)-                 nnuccd(i,k))*deltat
                        IF (dum.gt.ni(i,k)) THEN
                            ratio = (ni(i,k)/deltat+nnuccd(i,k)+                    (nnucct(i,k)+tmpfrz+nnudep(i,k)+nsacwi(i,k))&
                            *lcldm(i,k)+                    nnuccri(i,k)*precip_frac(i,k))/                    ((nprci(i,k)+nprai(&
                            i,k)-nsubi(i,k))*icldm(i,k))*omsm
                            nprci(i,k) = nprci(i,k)*ratio
                            nprai(i,k) = nprai(i,k)*ratio
                            nsubi(i,k) = nsubi(i,k)*ratio
                        END IF 
                    END DO 
                END IF 
                DO i=1,mgncol
                    ! conservation of snow mixing ratio
                    !-------------------------------------------------------------------
                    dum = (-(prds(i,k)+pracs(i,k)+mnuccr(i,k))*precip_frac(i,k)-(prai(i,k)+prci(i,k))*icldm(i,k)              -(&
                    bergs(i,k)+psacws(i,k))*lcldm(i,k))*deltat
                    IF (dum.gt.qs(i,k).and.-prds(i,k).ge.qsmall) THEN
                        ratio = (qs(i,k)/deltat+(prai(i,k)+prci(i,k))*icldm(i,k)+                 (bergs(i,k)+psacws(i,k))*lcldm(&
                        i,k)+(pracs(i,k)+mnuccr(i,k))*precip_frac(i,k))/                 precip_frac(i,k)/(-prds(i,k))*omsm
                        prds(i,k) = prds(i,k)*ratio
                    END IF 
                END DO 
                DO i=1,mgncol
                    ! conservation of snow number
                    !-------------------------------------------------------------------
                    ! calculate loss of number due to sublimation
                    ! for now neglect sublimation of ns
                    nsubs(i,k) = 0._r8
                    dum = ((-nsagg(i,k)-nsubs(i,k)-nnuccr(i,k))*precip_frac(i,k)-nprci(i,k)*icldm(i,k))*deltat
                    IF (dum.gt.ns(i,k)) THEN
                        ratio = (ns(i,k)/deltat+nnuccr(i,k)*                 precip_frac(i,k)+nprci(i,k)*icldm(i,k))/precip_frac(&
                        i,k)/                 (-nsubs(i,k)-nsagg(i,k))*omsm
                        nsubs(i,k) = nsubs(i,k)*ratio
                        nsagg(i,k) = nsagg(i,k)*ratio
                    END IF 
                END DO 
                DO i=1,mgncol
                    ! next limit ice and snow sublimation and rain evaporation
                    ! get estimate of q and t at end of time step
                    ! don't include other microphysical processes since they haven't
                    ! been limited via conservation checks yet
                    IF ((pre(i,k)+prds(i,k))*precip_frac(i,k)+ice_sublim(i,k) < -1.e-20_r8) THEN
                        qtmp = q(i,k)-(ice_sublim(i,k)+vap_dep(i,k)+mnuccd(i,k)+                 (pre(i,k)+prds(i,k))*precip_frac(&
                        i,k))*deltat
                        ttmp = t(i,k)+((pre(i,k)*precip_frac(i,k))*xxlv+                 (prds(i,k)*precip_frac(i,k)+vap_dep(i,k)&
                        +ice_sublim(i,k)+mnuccd(i,k))*xxls)*deltat/cpp
                        ! use rhw to allow ice supersaturation
                        CALL qsat_water(ttmp, p(i,k), esn, qvn)
                        ! modify ice/precip evaporation rate if q > qsat
                        IF (qtmp > qvn) THEN
                            dum1 = pre(i,k)*precip_frac(i,k)/((pre(i,k)+prds(i,k))*precip_frac(i,k)+ice_sublim(i,k))
                            dum2 = prds(i,k)*precip_frac(i,k)/((pre(i,k)+prds(i,k))*precip_frac(i,k)+ice_sublim(i,k))
                            ! recalculate q and t after vap_dep and mnuccd but without evap or sublim
                            qtmp = q(i,k)-(vap_dep(i,k)+mnuccd(i,k))*deltat
                            ttmp = t(i,k)+((vap_dep(i,k)+mnuccd(i,k))*xxls)*deltat/cpp
                            ! use rhw to allow ice supersaturation
                            CALL qsat_water(ttmp, p(i,k), esn, qvn)
                            dum = (qtmp-qvn)/(1._r8 + xxlv_squared*qvn/(cpp*rv*ttmp**2))
                            dum = min(dum,0._r8)
                            ! modify rates if needed, divide by precip_frac to get local (in-precip) value
                            pre(i,k) = dum*dum1/deltat/precip_frac(i,k)
                            ! do separately using RHI for prds and ice_sublim
                            CALL qsat_ice(ttmp, p(i,k), esn, qvn)
                            dum = (qtmp-qvn)/(1._r8 + xxls_squared*qvn/(cpp*rv*ttmp**2))
                            dum = min(dum,0._r8)
                            ! modify rates if needed, divide by precip_frac to get local (in-precip) value
                            prds(i,k) = dum*dum2/deltat/precip_frac(i,k)
                            ! don't divide ice_sublim by cloud fraction since it is grid-averaged
                            dum1 = (1._r8-dum1-dum2)
                            ice_sublim(i,k) = dum*dum1/deltat
                        END IF 
                    END IF 
                END DO 
                ! Big "administration" loop enforces conservation, updates variables
                ! that accumulate over substeps, and sets output variables.
                DO i=1,mgncol
                    ! get tendencies due to microphysical conversion processes
                    !==========================================================
                    ! note: tendencies are multiplied by appropriate cloud/precip
                    ! fraction to get grid-scale values
                    ! note: vap_dep is already grid-average values
                    ! The net tendencies need to be added to rather than overwritten,
                    ! because they may have a value already set for instantaneous
                    ! melting/freezing.
                    qvlat(i,k) = qvlat(i,k)-(pre(i,k)+prds(i,k))*precip_frac(i,k)-             vap_dep(i,k)-ice_sublim(i,k)&
                    -mnuccd(i,k)-mnudep(i,k)*lcldm(i,k)
                    tlat(i,k) = tlat(i,k)+((pre(i,k)*precip_frac(i,k))              *xxlv+(prds(i,k)*precip_frac(i,k)+vap_dep(i,k)&
                    +ice_sublim(i,k)+mnuccd(i,k)+mnudep(i,k)*lcldm(i,k))*xxls+              ((bergs(i,k)+psacws(i,k)+mnuccc(i,k)&
                    +mnucct(i,k)+msacwi(i,k))*lcldm(i,k)+(mnuccr(i,k)+              pracs(i,k)+mnuccri(i,k))*precip_frac(i,k)&
                    +berg(i,k))*xlf)
                    qctend(i,k) = qctend(i,k)+              (-pra(i,k)-prc(i,k)-mnuccc(i,k)-mnucct(i,k)-msacwi(i,k)-              &
                    psacws(i,k)-bergs(i,k))*lcldm(i,k)-berg(i,k)
                    IF (do_cldice) THEN
                        qitend(i,k) = qitend(i,k)+                 (mnuccc(i,k)+mnucct(i,k)+mnudep(i,k)+msacwi(i,k))*lcldm(i,k)+(&
                        -prci(i,k)-                 prai(i,k))*icldm(i,k)+vap_dep(i,k)+berg(i,k)+ice_sublim(i,k)+                 &
                        mnuccd(i,k)+mnuccri(i,k)*precip_frac(i,k)
                    END IF 
                    qrtend(i,k) = qrtend(i,k)+              (pra(i,k)+prc(i,k))*lcldm(i,k)+(pre(i,k)-pracs(i,k)-              &
                    mnuccr(i,k)-mnuccri(i,k))*precip_frac(i,k)
                    qstend(i,k) = qstend(i,k)+              (prai(i,k)+prci(i,k))*icldm(i,k)+(psacws(i,k)+bergs(i,k))*lcldm(i,k)+(&
                    prds(i,k)+              pracs(i,k)+mnuccr(i,k))*precip_frac(i,k)
                    cmeout(i,k) = vap_dep(i,k) + ice_sublim(i,k) + mnuccd(i,k)
                    ! add output for cmei (accumulate)
                    cmeitot(i,k) = vap_dep(i,k) + ice_sublim(i,k) + mnuccd(i,k)
                    ! assign variables for trop_mozart, these are grid-average
                    !-------------------------------------------------------------------
                    ! evaporation/sublimation is stored here as positive term
                    evapsnow(i,k) = -prds(i,k)*precip_frac(i,k)
                    nevapr(i,k) = -pre(i,k)*precip_frac(i,k)
                    prer_evap(i,k) = -pre(i,k)*precip_frac(i,k)
                    ! change to make sure prain is positive: do not remove snow from
                    ! prain used for wet deposition
                    prain(i,k) = (pra(i,k)+prc(i,k))*lcldm(i,k)+(-pracs(i,k)-              mnuccr(i,k)-mnuccri(i,k))*precip_frac(&
                    i,k)
                    prodsnow(i,k) = (prai(i,k)+prci(i,k))*icldm(i,k)+(psacws(i,k)+bergs(i,k))*lcldm(i,k)+(pracs(i,k)+mnuccr(i,k))&
                    *precip_frac(i,k)
                    ! following are used to calculate 1st order conversion rate of cloud water
                    !    to rain and snow (1/s), for later use in aerosol wet removal routine
                    ! previously, wetdepa used (prain/qc) for this, and the qc in wetdepa may be smaller than the qc
                    !    used to calculate pra, prc, ... in this routine
                    ! qcsinksum_rate1ord = { rate of direct transfer of cloud water to rain & snow }
                    !                      (no cloud ice or bergeron terms)
                    qcsinksum_rate1ord(i,k) = (pra(i,k)+prc(i,k)+psacws(i,k))*lcldm(i,k)
                    ! Avoid zero/near-zero division.
                    qcsinksum_rate1ord(i,k) = qcsinksum_rate1ord(i,k) /              max(qc(i,k),1.0e-30_r8)
                    ! microphysics output, note this is grid-averaged
                    pratot(i,k) = pra(i,k)*lcldm(i,k)
                    prctot(i,k) = prc(i,k)*lcldm(i,k)
                    mnuccctot(i,k) = mnuccc(i,k)*lcldm(i,k)
                    mnuccttot(i,k) = mnucct(i,k)*lcldm(i,k)
                    msacwitot(i,k) = msacwi(i,k)*lcldm(i,k)
                    psacwstot(i,k) = psacws(i,k)*lcldm(i,k)
                    bergstot(i,k) = bergs(i,k)*lcldm(i,k)
                    bergtot(i,k) = berg(i,k)
                    prcitot(i,k) = prci(i,k)*icldm(i,k)
                    praitot(i,k) = prai(i,k)*icldm(i,k)
                    mnuccdtot(i,k) = mnuccd(i,k)*icldm(i,k)
                    pracstot(i,k) = pracs(i,k)*precip_frac(i,k)
                    mnuccrtot(i,k) = mnuccr(i,k)*precip_frac(i,k)
                    nctend(i,k) = nctend(i,k)+             (-nnuccc(i,k)-nnucct(i,k)-npsacws(i,k)+nsubc(i,k)              -npra(i,&
                    k)-nprc1(i,k))*lcldm(i,k)
                    IF (do_cldice) THEN
                        IF (use_hetfrz_classnuc) THEN
                            tmpfrz = nnuccc(i,k)
                            ELSE
                            tmpfrz = 0._r8
                        END IF 
                        nitend(i,k) = nitend(i,k)+ nnuccd(i,k)+                 (nnucct(i,k)+tmpfrz+nnudep(i,k)+nsacwi(i,k))&
                        *lcldm(i,k)+(nsubi(i,k)-nprci(i,k)-                 nprai(i,k))*icldm(i,k)+nnuccri(i,k)*precip_frac(i,k)
                    END IF 
                    nstend(i,k) = nstend(i,k)+(nsubs(i,k)+              nsagg(i,k)+nnuccr(i,k))*precip_frac(i,k)+nprci(i,k)*icldm(&
                    i,k)
                    nrtend(i,k) = nrtend(i,k)+              nprc(i,k)*lcldm(i,k)+(nsubr(i,k)-npracs(i,k)-nnuccr(i,k)              &
                    -nnuccri(i,k)+nragg(i,k))*precip_frac(i,k)
                    ! make sure that ni at advanced time step does not exceed
                    ! maximum (existing N + source terms*dt), which is possible if mtime < deltat
                    ! note that currently mtime = deltat
                    !================================================================
                    IF (do_cldice .and. nitend(i,k).gt.0._r8.and.ni(i,k)+nitend(i,k)*deltat.gt.nimax(i,k)) THEN
                        nitend(i,k) = max(0._r8,(nimax(i,k)-ni(i,k))/deltat)
                    END IF 
                END DO 
                ! End of "administration" loop
            END DO micro_vert_loop ! end k loop
            !-----------------------------------------------------
            ! convert rain/snow q and N for output to history, note,
            ! output is for gridbox average
            qrout = qr
            nrout = nr * rho
            qsout = qs
            nsout = ns * rho
            ! calculate precip fluxes
            ! calculate the precip flux (kg/m2/s) as mixingratio(kg/kg)*airdensity(kg/m3)*massweightedfallspeed(m/s)
            ! ---------------------------------------------------------------------
            rflx(:,2:) = rflx(:,2:) + (qric*rho*umr*precip_frac)
            sflx(:,2:) = sflx(:,2:) + (qsic*rho*ums*precip_frac)
            ! calculate n0r and lamr from rain mass and number
            ! divide by precip fraction to get in-precip (local) values of
            ! rain mass and number, divide by rhow to get rain number in kg^-1
            CALL size_dist_param_basic(mg_rain_props, qric, nric, lamr, n0r)
            ! Calculate rercld
            ! calculate mean size of combined rain and cloud water
            CALL calc_rercld(lamr, n0r, lamc, pgam, qric, qcic, ncic, rercld)
            ! Assign variables back to start-of-timestep values
            ! Some state variables are changed before the main microphysics loop
            ! to make "instantaneous" adjustments. Afterward, we must move those changes
            ! back into the tendencies.
            ! These processes:
            !  - Droplet activation (npccn, impacts nc)
            !  - Instantaneous snow melting  (minstsm/ninstsm, impacts qr/qs/nr/ns)
            !  - Instantaneous rain freezing (minstfr/ninstrf, impacts qr/qs/nr/ns)
            !================================================================================
            ! Re-apply droplet activation tendency
            nc = ncn
            nctend = nctend + npccn
            ! Re-apply rain freezing and snow melting.
            dum_2d = qs
            qs = qsn
            qstend = qstend + (dum_2d-qs)/deltat
            dum_2d = ns
            ns = nsn
            nstend = nstend + (dum_2d-ns)/deltat
            dum_2d = qr
            qr = qrn
            qrtend = qrtend + (dum_2d-qr)/deltat
            dum_2d = nr
            nr = nrn
            nrtend = nrtend + (dum_2d-nr)/deltat
            !.............................................................................
            !================================================================================
            ! modify to include snow. in prain & evap (diagnostic here: for wet dep)
            nevapr = nevapr + evapsnow
            prain = prain + prodsnow
            sed_col_loop: DO i=1,mgncol
                DO k=1,nlev
                    ! calculate sedimentation for cloud water and ice
                    !================================================================================
                    ! update in-cloud cloud mixing ratio and number concentration
                    ! with microphysical tendencies to calculate sedimentation, assign to dummy vars
                    ! note: these are in-cloud values***, hence we divide by cloud fraction
                    dumc(i,k) = (qc(i,k)+qctend(i,k)*deltat)/lcldm(i,k)
                    dumi(i,k) = (qi(i,k)+qitend(i,k)*deltat)/icldm(i,k)
                    dumnc(i,k) = max((nc(i,k)+nctend(i,k)*deltat)/lcldm(i,k),0._r8)
                    dumni(i,k) = max((ni(i,k)+nitend(i,k)*deltat)/icldm(i,k),0._r8)
                    dumr(i,k) = (qr(i,k)+qrtend(i,k)*deltat)/precip_frac(i,k)
                    dumnr(i,k) = max((nr(i,k)+nrtend(i,k)*deltat)/precip_frac(i,k),0._r8)
                    dums(i,k) = (qs(i,k)+qstend(i,k)*deltat)/precip_frac(i,k)
                    dumns(i,k) = max((ns(i,k)+nstend(i,k)*deltat)/precip_frac(i,k),0._r8)
                    ! switch for specification of droplet and crystal number
                    IF (nccons) THEN
                        dumnc(i,k) = ncnst/rho(i,k)
                    END IF 
                    ! switch for specification of cloud ice number
                    IF (nicons) THEN
                        dumni(i,k) = ninst/rho(i,k)
                    END IF 
                    ! obtain new slope parameter to avoid possible singularity
                    CALL size_dist_param_basic(mg_ice_props, dumi(i,k), dumni(i,k), lami(i,k))
                    CALL size_dist_param_liq(mg_liq_props, dumc(i,k), dumnc(i,k), rho(i,k), pgam(i,k), lamc(i,k))
                    ! calculate number and mass weighted fall velocity for droplets and cloud ice
                    !-------------------------------------------------------------------
                    IF (dumc(i,k).ge.qsmall) THEN
                        vtrmc(i,k) = acn(i,k)*gamma(4._r8+bc+pgam(i,k))/                 (lamc(i,k)**bc*gamma(pgam(i,k)+4._r8))
                        fc(k) = g*rho(i,k)*vtrmc(i,k)
                        fnc(k) = g*rho(i,k)*                 acn(i,k)*gamma(1._r8+bc+pgam(i,k))/                 (lamc(i,k)&
                        **bc*gamma(pgam(i,k)+1._r8))
                        ELSE
                        fc(k) = 0._r8
                        fnc(k) = 0._r8
                    END IF 
                    ! calculate number and mass weighted fall velocity for cloud ice
                    IF (dumi(i,k).ge.qsmall) THEN
                        vtrmi(i,k) = min(ain(i,k)*gamma_bi_plus4/(6._r8*lami(i,k)**bi),                 1.2_r8*rhof(i,k))
                        fi(k) = g*rho(i,k)*vtrmi(i,k)
                        fni(k) = g*rho(i,k)*                 min(ain(i,k)*gamma_bi_plus1/lami(i,k)**bi,1.2_r8*rhof(i,k))
                        ELSE
                        fi(k) = 0._r8
                        fni(k) = 0._r8
                    END IF 
                    ! fallspeed for rain
                    CALL size_dist_param_basic(mg_rain_props, dumr(i,k), dumnr(i,k), lamr(i,k))
                    IF (lamr(i,k).ge.qsmall) THEN
                        ! 'final' values of number and mass weighted mean fallspeed for rain (m/s)
                        unr(i,k) = min(arn(i,k)*gamma_br_plus1/lamr(i,k)**br,9.1_r8*rhof(i,k))
                        umr(i,k) = min(arn(i,k)*gamma_br_plus4/(6._r8*lamr(i,k)**br),9.1_r8*rhof(i,k))
                        fr(k) = g*rho(i,k)*umr(i,k)
                        fnr(k) = g*rho(i,k)*unr(i,k)
                        ELSE
                        fr(k) = 0._r8
                        fnr(k) = 0._r8
                    END IF 
                    ! fallspeed for snow
                    CALL size_dist_param_basic(mg_snow_props, dums(i,k), dumns(i,k), lams(i,k))
                    IF (lams(i,k).ge.qsmall) THEN
                        ! 'final' values of number and mass weighted mean fallspeed for snow (m/s)
                        ums(i,k) = min(asn(i,k)*gamma_bs_plus4/(6._r8*lams(i,k)**bs),1.2_r8*rhof(i,k))
                        uns(i,k) = min(asn(i,k)*gamma_bs_plus1/lams(i,k)**bs,1.2_r8*rhof(i,k))
                        fs(k) = g*rho(i,k)*ums(i,k)
                        fns(k) = g*rho(i,k)*uns(i,k)
                        ELSE
                        fs(k) = 0._r8
                        fns(k) = 0._r8
                    END IF 
                    ! redefine dummy variables - sedimentation is calculated over grid-scale
                    ! quantities to ensure conservation
                    dumc(i,k) = (qc(i,k)+qctend(i,k)*deltat)
                    dumnc(i,k) = max((nc(i,k)+nctend(i,k)*deltat),0._r8)
                    dumi(i,k) = (qi(i,k)+qitend(i,k)*deltat)
                    dumni(i,k) = max((ni(i,k)+nitend(i,k)*deltat),0._r8)
                    dumr(i,k) = (qr(i,k)+qrtend(i,k)*deltat)
                    dumnr(i,k) = max((nr(i,k)+nrtend(i,k)*deltat),0._r8)
                    dums(i,k) = (qs(i,k)+qstend(i,k)*deltat)
                    dumns(i,k) = max((ns(i,k)+nstend(i,k)*deltat),0._r8)
                    IF (dumc(i,k).lt.qsmall) dumnc(i,k) = 0._r8
                    IF (dumi(i,k).lt.qsmall) dumni(i,k) = 0._r8
                    IF (dumr(i,k).lt.qsmall) dumnr(i,k) = 0._r8
                    IF (dums(i,k).lt.qsmall) dumns(i,k) = 0._r8
                END DO  !!! vertical loop
                ! initialize nstep for sedimentation sub-steps
                ! calculate number of split time steps to ensure courant stability criteria
                ! for sedimentation calculations
                !-------------------------------------------------------------------
                nstep = 1 + int(max(           maxval( fi/pdel(i,:)),           maxval(fni/pdel(i,:)))           * deltat)
                ! loop over sedimentation sub-time step to ensure stability
                !==============================================================
                DO n = 1,nstep
                    IF (do_cldice) THEN
                        falouti = fi  * dumi(i,:)
                        faloutni = fni * dumni(i,:)
                        ELSE
                        falouti = 0._r8
                        faloutni = 0._r8
                    END IF 
                    ! top of model
                    k = 1
                    ! add fallout terms to microphysical tendencies
                    faltndi = falouti(k)/pdel(i,k)
                    faltndni = faloutni(k)/pdel(i,k)
                    qitend(i,k) = qitend(i,k)-faltndi/nstep
                    nitend(i,k) = nitend(i,k)-faltndni/nstep
                    ! sedimentation tendency for output
                    qisedten(i,k) = qisedten(i,k)-faltndi/nstep
                    dumi(i,k) = dumi(i,k)-faltndi*deltat/nstep
                    dumni(i,k) = dumni(i,k)-faltndni*deltat/nstep
                    DO k = 2,nlev
                        ! for cloud liquid and ice, if cloud fraction increases with height
                        ! then add flux from above to both vapor and cloud water of current level
                        ! this means that flux entering clear portion of cell from above evaporates
                        ! instantly
                        ! note: this is not an issue with precip, since we assume max overlap
                        dum1 = icldm(i,k)/icldm(i,k-1)
                        dum1 = min(dum1,1._r8)
                        faltndqie = (falouti(k)-falouti(k-1))/pdel(i,k)
                        faltndi = (falouti(k)-dum1*falouti(k-1))/pdel(i,k)
                        faltndni = (faloutni(k)-dum1*faloutni(k-1))/pdel(i,k)
                        ! add fallout terms to eulerian tendencies
                        qitend(i,k) = qitend(i,k)-faltndi/nstep
                        nitend(i,k) = nitend(i,k)-faltndni/nstep
                        ! sedimentation tendency for output
                        qisedten(i,k) = qisedten(i,k)-faltndi/nstep
                        ! add terms to to evap/sub of cloud water
                        qvlat(i,k) = qvlat(i,k)-(faltndqie-faltndi)/nstep
                        ! for output
                        qisevap(i,k) = qisevap(i,k)-(faltndqie-faltndi)/nstep
                        tlat(i,k) = tlat(i,k)+(faltndqie-faltndi)*xxls/nstep
                        dumi(i,k) = dumi(i,k)-faltndi*deltat/nstep
                        dumni(i,k) = dumni(i,k)-faltndni*deltat/nstep
                    END DO 
                    ! units below are m/s
                    ! sedimentation flux at surface is added to precip flux at surface
                    ! to get total precip (cloud + precip water) rate
                    prect(i) = prect(i)+falouti(nlev)/g/real(nstep)/1000._r8
                    preci(i) = preci(i)+falouti(nlev)/g/real(nstep)/1000._r8
                END DO 
                ! calculate number of split time steps to ensure courant stability criteria
                ! for sedimentation calculations
                !-------------------------------------------------------------------
                nstep = 1 + int(max(           maxval( fc/pdel(i,:)),           maxval(fnc/pdel(i,:)))           * deltat)
                ! loop over sedimentation sub-time step to ensure stability
                !==============================================================
                DO n = 1,nstep
                    faloutc = fc  * dumc(i,:)
                    faloutnc = fnc * dumnc(i,:)
                    ! top of model
                    k = 1
                    ! add fallout terms to microphysical tendencies
                    faltndc = faloutc(k)/pdel(i,k)
                    faltndnc = faloutnc(k)/pdel(i,k)
                    qctend(i,k) = qctend(i,k)-faltndc/nstep
                    nctend(i,k) = nctend(i,k)-faltndnc/nstep
                    ! sedimentation tendency for output
                    qcsedten(i,k) = qcsedten(i,k)-faltndc/nstep
                    dumc(i,k) = dumc(i,k)-faltndc*deltat/nstep
                    dumnc(i,k) = dumnc(i,k)-faltndnc*deltat/nstep
                    DO k = 2,nlev
                        dum = lcldm(i,k)/lcldm(i,k-1)
                        dum = min(dum,1._r8)
                        faltndqce = (faloutc(k)-faloutc(k-1))/pdel(i,k)
                        faltndc = (faloutc(k)-dum*faloutc(k-1))/pdel(i,k)
                        faltndnc = (faloutnc(k)-dum*faloutnc(k-1))/pdel(i,k)
                        ! add fallout terms to eulerian tendencies
                        qctend(i,k) = qctend(i,k)-faltndc/nstep
                        nctend(i,k) = nctend(i,k)-faltndnc/nstep
                        ! sedimentation tendency for output
                        qcsedten(i,k) = qcsedten(i,k)-faltndc/nstep
                        ! add terms to to evap/sub of cloud water
                        qvlat(i,k) = qvlat(i,k)-(faltndqce-faltndc)/nstep
                        ! for output
                        qcsevap(i,k) = qcsevap(i,k)-(faltndqce-faltndc)/nstep
                        tlat(i,k) = tlat(i,k)+(faltndqce-faltndc)*xxlv/nstep
                        dumc(i,k) = dumc(i,k)-faltndc*deltat/nstep
                        dumnc(i,k) = dumnc(i,k)-faltndnc*deltat/nstep
                    END DO 
                    prect(i) = prect(i)+faloutc(nlev)/g/real(nstep)/1000._r8
                END DO 
                ! calculate number of split time steps to ensure courant stability criteria
                ! for sedimentation calculations
                !-------------------------------------------------------------------
                nstep = 1 + int(max(           maxval( fr/pdel(i,:)),           maxval(fnr/pdel(i,:)))           * deltat)
                ! loop over sedimentation sub-time step to ensure stability
                !==============================================================
                DO n = 1,nstep
                    faloutr = fr  * dumr(i,:)
                    faloutnr = fnr * dumnr(i,:)
                    ! top of model
                    k = 1
                    ! add fallout terms to microphysical tendencies
                    faltndr = faloutr(k)/pdel(i,k)
                    faltndnr = faloutnr(k)/pdel(i,k)
                    qrtend(i,k) = qrtend(i,k)-faltndr/nstep
                    nrtend(i,k) = nrtend(i,k)-faltndnr/nstep
                    ! sedimentation tendency for output
                    qrsedten(i,k) = qrsedten(i,k)-faltndr/nstep
                    dumr(i,k) = dumr(i,k)-faltndr*deltat/real(nstep)
                    dumnr(i,k) = dumnr(i,k)-faltndnr*deltat/real(nstep)
                    DO k = 2,nlev
                        faltndr = (faloutr(k)-faloutr(k-1))/pdel(i,k)
                        faltndnr = (faloutnr(k)-faloutnr(k-1))/pdel(i,k)
                        ! add fallout terms to eulerian tendencies
                        qrtend(i,k) = qrtend(i,k)-faltndr/nstep
                        nrtend(i,k) = nrtend(i,k)-faltndnr/nstep
                        ! sedimentation tendency for output
                        qrsedten(i,k) = qrsedten(i,k)-faltndr/nstep
                        dumr(i,k) = dumr(i,k)-faltndr*deltat/real(nstep)
                        dumnr(i,k) = dumnr(i,k)-faltndnr*deltat/real(nstep)
                    END DO 
                    prect(i) = prect(i)+faloutr(nlev)/g/real(nstep)/1000._r8
                END DO 
                ! calculate number of split time steps to ensure courant stability criteria
                ! for sedimentation calculations
                !-------------------------------------------------------------------
                nstep = 1 + int(max(           maxval( fs/pdel(i,:)),           maxval(fns/pdel(i,:)))           * deltat)
                ! loop over sedimentation sub-time step to ensure stability
                !==============================================================
                DO n = 1,nstep
                    falouts = fs  * dums(i,:)
                    faloutns = fns * dumns(i,:)
                    ! top of model
                    k = 1
                    ! add fallout terms to microphysical tendencies
                    faltnds = falouts(k)/pdel(i,k)
                    faltndns = faloutns(k)/pdel(i,k)
                    qstend(i,k) = qstend(i,k)-faltnds/nstep
                    nstend(i,k) = nstend(i,k)-faltndns/nstep
                    ! sedimentation tendency for output
                    qssedten(i,k) = qssedten(i,k)-faltnds/nstep
                    dums(i,k) = dums(i,k)-faltnds*deltat/real(nstep)
                    dumns(i,k) = dumns(i,k)-faltndns*deltat/real(nstep)
                    DO k = 2,nlev
                        faltnds = (falouts(k)-falouts(k-1))/pdel(i,k)
                        faltndns = (faloutns(k)-faloutns(k-1))/pdel(i,k)
                        ! add fallout terms to eulerian tendencies
                        qstend(i,k) = qstend(i,k)-faltnds/nstep
                        nstend(i,k) = nstend(i,k)-faltndns/nstep
                        ! sedimentation tendency for output
                        qssedten(i,k) = qssedten(i,k)-faltnds/nstep
                        dums(i,k) = dums(i,k)-faltnds*deltat/real(nstep)
                        dumns(i,k) = dumns(i,k)-faltndns*deltat/real(nstep)
                    END DO  !! k loop
                    prect(i) = prect(i)+falouts(nlev)/g/real(nstep)/1000._r8
                    preci(i) = preci(i)+falouts(nlev)/g/real(nstep)/1000._r8
                END DO  !! nstep loop
                ! end sedimentation
                !ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
                ! get new update for variables that includes sedimentation tendency
                ! note : here dum variables are grid-average, NOT in-cloud
                DO k=1,nlev
                    dumc(i,k) = max(qc(i,k)+qctend(i,k)*deltat,0._r8)
                    dumi(i,k) = max(qi(i,k)+qitend(i,k)*deltat,0._r8)
                    dumnc(i,k) = max(nc(i,k)+nctend(i,k)*deltat,0._r8)
                    dumni(i,k) = max(ni(i,k)+nitend(i,k)*deltat,0._r8)
                    dumr(i,k) = max(qr(i,k)+qrtend(i,k)*deltat,0._r8)
                    dumnr(i,k) = max(nr(i,k)+nrtend(i,k)*deltat,0._r8)
                    dums(i,k) = max(qs(i,k)+qstend(i,k)*deltat,0._r8)
                    dumns(i,k) = max(ns(i,k)+nstend(i,k)*deltat,0._r8)
                    ! switch for specification of droplet and crystal number
                    IF (nccons) THEN
                        dumnc(i,k) = ncnst/rho(i,k)*lcldm(i,k)
                    END IF 
                    ! switch for specification of cloud ice number
                    IF (nicons) THEN
                        dumni(i,k) = ninst/rho(i,k)*icldm(i,k)
                    END IF 
                    IF (dumc(i,k).lt.qsmall) dumnc(i,k) = 0._r8
                    IF (dumi(i,k).lt.qsmall) dumni(i,k) = 0._r8
                    IF (dumr(i,k).lt.qsmall) dumnr(i,k) = 0._r8
                    IF (dums(i,k).lt.qsmall) dumns(i,k) = 0._r8
                    ! calculate instantaneous processes (melting, homogeneous freezing)
                    !====================================================================
                    ! melting of snow at +2 C
                    IF (t(i,k)+tlat(i,k)/cpp*deltat > snowmelt) THEN
                        IF (dums(i,k) > 0._r8) THEN
                            ! make sure melting snow doesn't reduce temperature below threshold
                            dum = -xlf/cpp*dums(i,k)
                            IF (t(i,k)+tlat(i,k)/cpp*deltat+dum.lt. snowmelt) THEN
                                dum = (t(i,k)+tlat(i,k)/cpp*deltat-snowmelt)*cpp/xlf
                                dum = dum/dums(i,k)
                                dum = max(0._r8,dum)
                                dum = min(1._r8,dum)
                                ELSE
                                dum = 1._r8
                            END IF 
                            qstend(i,k) = qstend(i,k)-dum*dums(i,k)/deltat
                            nstend(i,k) = nstend(i,k)-dum*dumns(i,k)/deltat
                            qrtend(i,k) = qrtend(i,k)+dum*dums(i,k)/deltat
                            nrtend(i,k) = nrtend(i,k)+dum*dumns(i,k)/deltat
                            dum1 = -xlf*dum*dums(i,k)/deltat
                            tlat(i,k) = tlat(i,k)+dum1
                            meltsdttot(i,k) = meltsdttot(i,k) + dum1
                        END IF 
                    END IF 
                    ! freezing of rain at -5 C
                    IF (t(i,k)+tlat(i,k)/cpp*deltat < rainfrze) THEN
                        IF (dumr(i,k) > 0._r8) THEN
                            ! make sure freezing rain doesn't increase temperature above threshold
                            dum = xlf/cpp*dumr(i,k)
                            IF (t(i,k)+tlat(i,k)/cpp*deltat+dum.gt.rainfrze) THEN
                                dum = -(t(i,k)+tlat(i,k)/cpp*deltat-rainfrze)*cpp/xlf
                                dum = dum/dumr(i,k)
                                dum = max(0._r8,dum)
                                dum = min(1._r8,dum)
                                ELSE
                                dum = 1._r8
                            END IF 
                            qrtend(i,k) = qrtend(i,k)-dum*dumr(i,k)/deltat
                            nrtend(i,k) = nrtend(i,k)-dum*dumnr(i,k)/deltat
                            ! get mean size of rain = 1/lamr, add frozen rain to either snow or cloud ice
                            ! depending on mean rain size
                            CALL size_dist_param_basic(mg_rain_props, dumr(i,k), dumnr(i,k), lamr(i,k))
                            IF (lamr(i,k) < 1._r8/dcs) THEN
                                qstend(i,k) = qstend(i,k)+dum*dumr(i,k)/deltat
                                nstend(i,k) = nstend(i,k)+dum*dumnr(i,k)/deltat
                                ELSE
                                qitend(i,k) = qitend(i,k)+dum*dumr(i,k)/deltat
                                nitend(i,k) = nitend(i,k)+dum*dumnr(i,k)/deltat
                            END IF 
                            ! heating tendency
                            dum1 = xlf*dum*dumr(i,k)/deltat
                            frzrdttot(i,k) = frzrdttot(i,k) + dum1
                            tlat(i,k) = tlat(i,k)+dum1
                        END IF 
                    END IF 
                    IF (do_cldice) THEN
                        IF (t(i,k)+tlat(i,k)/cpp*deltat > tmelt) THEN
                            IF (dumi(i,k) > 0._r8) THEN
                                ! limit so that melting does not push temperature below freezing
                                !-----------------------------------------------------------------
                                dum = -dumi(i,k)*xlf/cpp
                                IF (t(i,k)+tlat(i,k)/cpp*deltat+dum.lt.tmelt) THEN
                                    dum = (t(i,k)+tlat(i,k)/cpp*deltat-tmelt)*cpp/xlf
                                    dum = dum/dumi(i,k)
                                    dum = max(0._r8,dum)
                                    dum = min(1._r8,dum)
                                    ELSE
                                    dum = 1._r8
                                END IF 
                                qctend(i,k) = qctend(i,k)+dum*dumi(i,k)/deltat
                                ! for output
                                melttot(i,k) = dum*dumi(i,k)/deltat
                                ! assume melting ice produces droplet
                                ! mean volume radius of 8 micron
                                nctend(i,k) = nctend(i,k)+3._r8*dum*dumi(i,k)/deltat/                       (&
                                4._r8*pi*5.12e-16_r8*rhow)
                                qitend(i,k) = ((1._r8-dum)*dumi(i,k)-qi(i,k))/deltat
                                nitend(i,k) = ((1._r8-dum)*dumni(i,k)-ni(i,k))/deltat
                                tlat(i,k) = tlat(i,k)-xlf*dum*dumi(i,k)/deltat
                            END IF 
                        END IF 
                        ! homogeneously freeze droplets at -40 C
                        !-----------------------------------------------------------------
                        IF (t(i,k)+tlat(i,k)/cpp*deltat < 233.15_r8) THEN
                            IF (dumc(i,k) > 0._r8) THEN
                                ! limit so that freezing does not push temperature above threshold
                                dum = dumc(i,k)*xlf/cpp
                                IF (t(i,k)+tlat(i,k)/cpp*deltat+dum.gt.233.15_r8) THEN
                                    dum = -(t(i,k)+tlat(i,k)/cpp*deltat-233.15_r8)*cpp/xlf
                                    dum = dum/dumc(i,k)
                                    dum = max(0._r8,dum)
                                    dum = min(1._r8,dum)
                                    ELSE
                                    dum = 1._r8
                                END IF 
                                qitend(i,k) = qitend(i,k)+dum*dumc(i,k)/deltat
                                ! for output
                                homotot(i,k) = dum*dumc(i,k)/deltat
                                ! assume 25 micron mean volume radius of homogeneously frozen droplets
                                ! consistent with size of detrained ice in stratiform.F90
                                nitend(i,k) = nitend(i,k)+dum*3._r8*dumc(i,k)/(4._r8*3.14_r8*1.563e-14_r8*                       &
                                500._r8)/deltat
                                qctend(i,k) = ((1._r8-dum)*dumc(i,k)-qc(i,k))/deltat
                                nctend(i,k) = ((1._r8-dum)*dumnc(i,k)-nc(i,k))/deltat
                                tlat(i,k) = tlat(i,k)+xlf*dum*dumc(i,k)/deltat
                            END IF 
                        END IF 
                        ! remove any excess over-saturation, which is possible due to non-linearity when adding
                        ! together all microphysical processes
                        !-----------------------------------------------------------------
                        ! follow code similar to old 1 scheme
                        qtmp = q(i,k)+qvlat(i,k)*deltat
                        ttmp = t(i,k)+tlat(i,k)/cpp*deltat
                        ! use rhw to allow ice supersaturation
                        CALL qsat_water(ttmp, p(i,k), esn, qvn)
                        IF (qtmp > qvn .and. qvn > 0) THEN
                            ! expression below is approximate since there may be ice deposition
                            dum = (qtmp-qvn)/(1._r8+xxlv_squared*qvn/(cpp*rv*ttmp**2))/deltat
                            ! add to output cme
                            cmeout(i,k) = cmeout(i,k)+dum
                            ! now add to tendencies, partition between liquid and ice based on temperature
                            IF (ttmp > 268.15_r8) THEN
                                dum1 = 0.0_r8
                                ! now add to tendencies, partition between liquid and ice based on te
                                !-------------------------------------------------------
                                ELSE IF (ttmp < 238.15_r8) THEN
                                dum1 = 1.0_r8
                                ELSE
                                dum1 = (268.15_r8-ttmp)/30._r8
                            END IF 
                            dum = (qtmp-qvn)/(1._r8+(xxls*dum1+xxlv*(1._r8-dum1))**2                    *qvn/(cpp*rv*ttmp**2))/deltat
                            qctend(i,k) = qctend(i,k)+dum*(1._r8-dum1)
                            ! for output
                            qcrestot(i,k) = dum*(1._r8-dum1)
                            qitend(i,k) = qitend(i,k)+dum*dum1
                            qirestot(i,k) = dum*dum1
                            qvlat(i,k) = qvlat(i,k)-dum
                            ! for output
                            qvres(i,k) = -dum
                            tlat(i,k) = tlat(i,k)+dum*(1._r8-dum1)*xxlv+dum*dum1*xxls
                        END IF 
                    END IF 
                    ! calculate effective radius for pass to radiation code
                    !=========================================================
                    ! if no cloud water, default value is 10 micron for droplets,
                    ! 25 micron for cloud ice
                    ! update cloud variables after instantaneous processes to get effective radius
                    ! variables are in-cloud to calculate size dist parameters
                    dumc(i,k) = max(qc(i,k)+qctend(i,k)*deltat,0._r8)/lcldm(i,k)
                    dumi(i,k) = max(qi(i,k)+qitend(i,k)*deltat,0._r8)/icldm(i,k)
                    dumnc(i,k) = max(nc(i,k)+nctend(i,k)*deltat,0._r8)/lcldm(i,k)
                    dumni(i,k) = max(ni(i,k)+nitend(i,k)*deltat,0._r8)/icldm(i,k)
                    dumr(i,k) = max(qr(i,k)+qrtend(i,k)*deltat,0._r8)/precip_frac(i,k)
                    dumnr(i,k) = max(nr(i,k)+nrtend(i,k)*deltat,0._r8)/precip_frac(i,k)
                    dums(i,k) = max(qs(i,k)+qstend(i,k)*deltat,0._r8)/precip_frac(i,k)
                    dumns(i,k) = max(ns(i,k)+nstend(i,k)*deltat,0._r8)/precip_frac(i,k)
                    ! switch for specification of droplet and crystal number
                    IF (nccons) THEN
                        dumnc(i,k) = ncnst/rho(i,k)
                    END IF 
                    ! switch for specification of cloud ice number
                    IF (nicons) THEN
                        dumni(i,k) = ninst/rho(i,k)
                    END IF 
                    ! limit in-cloud mixing ratio to reasonable value of 5 g kg-1
                    dumc(i,k) = min(dumc(i,k),5.e-3_r8)
                    dumi(i,k) = min(dumi(i,k),5.e-3_r8)
                    ! limit in-precip mixing ratios
                    dumr(i,k) = min(dumr(i,k),10.e-3_r8)
                    dums(i,k) = min(dums(i,k),10.e-3_r8)
                    ! cloud ice effective radius
                    !-----------------------------------------------------------------
                    IF (do_cldice) THEN
                        IF (dumi(i,k).ge.qsmall) THEN
                            dum_2d(i,k) = dumni(i,k)
                            CALL size_dist_param_basic(mg_ice_props, dumi(i,k), dumni(i,k), lami(i,k))
                            IF (dumni(i,k) /=dum_2d(i,k)) THEN
                                ! adjust number conc if needed to keep mean size in reasonable range
                                nitend(i,k) = (dumni(i,k)*icldm(i,k)-ni(i,k))/deltat
                            END IF 
                            effi(i,k) = 1.5_r8/lami(i,k)*1.e6_r8
                            ELSE
                            effi(i,k) = 25._r8
                        END IF 
                        ! ice effective diameter for david mitchell's optics
                        deffi(i,k) = effi(i,k)*rhoi/rhows*2._r8
                        ELSE
                        ! NOTE: If CARMA is doing the ice microphysics, then the ice effective
                        ! radius has already been determined from the size distribution.
                        effi(i,k) = re_ice(i,k) * 1.e6_r8 ! m -> um
                        deffi(i,k) = effi(i,k) * 2._r8
                    END IF 
                    ! cloud droplet effective radius
                    !-----------------------------------------------------------------
                    IF (dumc(i,k).ge.qsmall) THEN
                        ! switch for specification of droplet and crystal number
                        IF (nccons) THEN
                            ! make sure nc is consistence with the constant N by adjusting tendency, need
                            ! to multiply by cloud fraction
                            ! note that nctend may be further adjusted below if mean droplet size is
                            ! out of bounds
                            nctend(i,k) = (ncnst/rho(i,k)*lcldm(i,k)-nc(i,k))/deltat
                        END IF 
                        dum = dumnc(i,k)
                        CALL size_dist_param_liq(mg_liq_props, dumc(i,k), dumnc(i,k), rho(i,k), pgam(i,k), lamc(i,k))
                        IF (dum /= dumnc(i,k)) THEN
                            ! adjust number conc if needed to keep mean size in reasonable range
                            nctend(i,k) = (dumnc(i,k)*lcldm(i,k)-nc(i,k))/deltat
                        END IF 
                        effc(i,k) = (pgam(i,k)+3._r8)/lamc(i,k)/2._r8*1.e6_r8
                        !assign output fields for shape here
                        lamcrad(i,k) = lamc(i,k)
                        pgamrad(i,k) = pgam(i,k)
                        ! recalculate effective radius for constant number, in order to separate
                        ! first and second indirect effects
                        !======================================
                        ! assume constant number of 10^8 kg-1
                        dumnc(i,k) = 1.e8_r8
                        ! Pass in "false" adjust flag to prevent number from being changed within
                        ! size distribution subroutine.
                        CALL size_dist_param_liq(mg_liq_props, dumc(i,k), dumnc(i,k), rho(i,k), pgam(i,k), lamc(i,k))
                        effc_fn(i,k) = (pgam(i,k)+3._r8)/lamc(i,k)/2._r8*1.e6_r8
                        ELSE
                        effc(i,k) = 10._r8
                        lamcrad(i,k) = 0._r8
                        pgamrad(i,k) = 0._r8
                        effc_fn(i,k) = 10._r8
                    END IF 
                    ! recalculate 'final' rain size distribution parameters
                    ! to ensure that rain size is in bounds, adjust rain number if needed
                    IF (dumr(i,k).ge.qsmall) THEN
                        dum = dumnr(i,k)
                        CALL size_dist_param_basic(mg_rain_props, dumr(i,k), dumnr(i,k), lamr(i,k))
                        IF (dum /= dumnr(i,k)) THEN
                            ! adjust number conc if needed to keep mean size in reasonable range
                            nrtend(i,k) = (dumnr(i,k)*precip_frac(i,k)-nr(i,k))/deltat
                        END IF 
                    END IF 
                    ! recalculate 'final' snow size distribution parameters
                    ! to ensure that snow size is in bounds, adjust snow number if needed
                    IF (dums(i,k).ge.qsmall) THEN
                        dum = dumns(i,k)
                        CALL size_dist_param_basic(mg_snow_props, dums(i,k), dumns(i,k), lams(i,k))
                        IF (dum /= dumns(i,k)) THEN
                            ! adjust number conc if needed to keep mean size in reasonable range
                            nstend(i,k) = (dumns(i,k)*precip_frac(i,k)-ns(i,k))/deltat
                        END IF 
                    END IF 
                END DO  ! vertical k loop
                DO k=1,nlev
                    ! if updated q (after microphysics) is zero, then ensure updated n is also zero
                    !=================================================================================
                    IF (qc(i,k)+qctend(i,k)*deltat.lt.qsmall) nctend(i,k) = -nc(i,k)/deltat
                    IF (do_cldice .and. qi(i,k)+qitend(i,k)*deltat.lt.qsmall) nitend(i,k) = -ni(i,k)/deltat
                    IF (qr(i,k)+qrtend(i,k)*deltat.lt.qsmall) nrtend(i,k) = -nr(i,k)/deltat
                    IF (qs(i,k)+qstend(i,k)*deltat.lt.qsmall) nstend(i,k) = -ns(i,k)/deltat
                END DO 
            END DO sed_col_loop ! i loop
            ! DO STUFF FOR OUTPUT:
            !==================================================
            ! qc and qi are only used for output calculations past here,
            ! so add qctend and qitend back in one more time
            qc = qc + qctend*deltat
            qi = qi + qitend*deltat
            ! averaging for snow and rain number and diameter
            !--------------------------------------------------
            ! drout2/dsout2:
            ! diameter of rain and snow
            ! dsout:
            ! scaled diameter of snow (passed to radiation in 1)
            ! reff_rain/reff_snow:
            ! calculate effective radius of rain and snow in microns for COSP using Eq. 9 of COSP v1.3 manual
            WHERE ( qrout .gt. 1.e-7_r8        .and. nrout.gt.0._r8 )
                qrout2 = qrout * precip_frac
                nrout2 = nrout * precip_frac
                ! The avg_diameter call does the actual calculation; other diameter
                ! outputs are just drout2 times constants.
                drout2 = avg_diameter(qrout, nrout, rho, rhow)
                freqr = precip_frac
                reff_rain = 1.5_r8*drout2*1.e6_r8
                ELSEWHERE
                qrout2 = 0._r8
                nrout2 = 0._r8
                drout2 = 0._r8
                freqr = 0._r8
                reff_rain = 0._r8
            END WHERE 
            WHERE ( qsout .gt. 1.e-7_r8        .and. nsout.gt.0._r8 )
                qsout2 = qsout * precip_frac
                nsout2 = nsout * precip_frac
                ! The avg_diameter call does the actual calculation; other diameter
                ! outputs are just dsout2 times constants.
                dsout2 = avg_diameter(qsout, nsout, rho, rhosn)
                freqs = precip_frac
                dsout = 3._r8*rhosn/rhows*dsout2
                reff_snow = 1.5_r8*dsout2*1.e6_r8
                ELSEWHERE
                dsout = 0._r8
                qsout2 = 0._r8
                nsout2 = 0._r8
                dsout2 = 0._r8
                freqs = 0._r8
                reff_snow = 0._r8
            END WHERE 
            ! analytic radar reflectivity
            !--------------------------------------------------
            ! formulas from Matthew Shupe, NOAA/CERES
            ! *****note: radar reflectivity is local (in-precip average)
            ! units of mm^6/m^3
            DO i = 1,mgncol
                DO k=1,nlev
                    IF (qc(i,k).ge.qsmall) THEN
                        dum = (qc(i,k)/lcldm(i,k)*rho(i,k)*1000._r8)**2                 /(0.109_r8*(nc(i,k)+nctend(i,k)*deltat)&
                        /lcldm(i,k)*rho(i,k)/1.e6_r8)*lcldm(i,k)/precip_frac(i,k)
                        ELSE
                        dum = 0._r8
                    END IF 
                    IF (qi(i,k).ge.qsmall) THEN
                        dum1 = (qi(i,k)*rho(i,k)/icldm(i,k)*1000._r8/0.1_r8)**(1._r8/0.63_r8)*icldm(i,k)/precip_frac(i,k)
                        ELSE
                        dum1 = 0._r8
                    END IF 
                    IF (qsout(i,k).ge.qsmall) THEN
                        dum1 = dum1+(qsout(i,k)*rho(i,k)*1000._r8/0.1_r8)**(1._r8/0.63_r8)
                    END IF 
                    refl(i,k) = dum+dum1
                    ! add rain rate, but for 37 GHz formulation instead of 94 GHz
                    ! formula approximated from data of Matrasov (2007)
                    ! rainrt is the rain rate in mm/hr
                    ! reflectivity (dum) is in DBz
                    IF (rainrt(i,k).ge.0.001_r8) THEN
                        dum = log10(rainrt(i,k)**6._r8)+16._r8
                        ! convert from DBz to mm^6/m^3
                        dum = 10._r8**(dum/10._r8)
                        ELSE
                        ! don't include rain rate in R calculation for values less than 0.001 mm/hr
                        dum = 0._r8
                    END IF 
                    ! add to refl
                    refl(i,k) = refl(i,k)+dum
                    !output reflectivity in Z.
                    areflz(i,k) = refl(i,k) * precip_frac(i,k)
                    ! convert back to DBz
                    IF (refl(i,k).gt.minrefl) THEN
                        refl(i,k) = 10._r8*log10(refl(i,k))
                        ELSE
                        refl(i,k) = -9999._r8
                    END IF 
                    !set averaging flag
                    IF (refl(i,k).gt.mindbz) THEN
                        arefl(i,k) = refl(i,k) * precip_frac(i,k)
                        frefl(i,k) = precip_frac(i,k)
                        ELSE
                        arefl(i,k) = 0._r8
                        areflz(i,k) = 0._r8
                        frefl(i,k) = 0._r8
                    END IF 
                    ! bound cloudsat reflectivity
                    csrfl(i,k) = min(csmax,refl(i,k))
                    !set averaging flag
                    IF (csrfl(i,k).gt.csmin) THEN
                        acsrfl(i,k) = refl(i,k) * precip_frac(i,k)
                        fcsrfl(i,k) = precip_frac(i,k)
                        ELSE
                        acsrfl(i,k) = 0._r8
                        fcsrfl(i,k) = 0._r8
                    END IF 
                END DO 
            END DO 
            !redefine fice here....
            dum_2d = qsout + qrout + qc + qi
            dumi = qsout + qi
            WHERE ( dumi .gt. qsmall .and. dum_2d .gt. qsmall )
                nfice = min(dumi/dum_2d,1._r8)
                ELSEWHERE
                nfice = 0._r8
            END WHERE 
        END SUBROUTINE micro_mg_tend
        !========================================================================
        !OUTPUT CALCULATIONS
        !========================================================================

        elemental SUBROUTINE calc_rercld(lamr, n0r, lamc, pgam, qric, qcic, ncic, rercld)
            REAL(KIND=r8), intent(in) :: lamr ! rain size parameter (slope)
            REAL(KIND=r8), intent(in) :: n0r ! rain size parameter (intercept)
            REAL(KIND=r8), intent(in) :: lamc ! size distribution parameter (slope)
            REAL(KIND=r8), intent(in) :: pgam ! droplet size parameter
            REAL(KIND=r8), intent(in) :: qric ! in-cloud rain mass mixing ratio
            REAL(KIND=r8), intent(in) :: qcic ! in-cloud cloud liquid
            REAL(KIND=r8), intent(in) :: ncic ! in-cloud droplet number concentration
            REAL(KIND=r8), intent(inout) :: rercld ! effective radius calculation for rain + cloud
            ! combined size of precip & cloud drops
            REAL(KIND=r8) :: atmp
            ! Rain drops
            IF (lamr > 0._r8) THEN
                atmp = n0r * pi / (2._r8 * lamr**3._r8)
                ELSE
                atmp = 0._r8
            END IF 
            ! Add cloud drops
            IF (lamc > 0._r8) THEN
                atmp = atmp +           ncic * pi * rising_factorial(pgam+1._r8, 2)/(4._r8 * lamc**2._r8)
            END IF 
            IF (atmp > 0._r8) THEN
                rercld = rercld + 3._r8 *(qric + qcic) / (4._r8 * rhow * atmp)
            END IF 
        END SUBROUTINE calc_rercld
        !========================================================================
        !UTILITIES
        !========================================================================

    END MODULE micro_mg2_0
