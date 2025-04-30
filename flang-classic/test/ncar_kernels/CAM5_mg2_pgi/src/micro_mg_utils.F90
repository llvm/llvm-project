
! KGEN-generated Fortran source file
!
! Filename    : micro_mg_utils.F90
! Generated at: 2015-03-31 09:44:40
! KGEN version: 0.4.5



    MODULE micro_mg_utils
        !--------------------------------------------------------------------------
        !
        ! This module contains process rates and utility functions used by the MG
        ! microphysics.
        !
        ! Original MG authors: Andrew Gettelman, Hugh Morrison
        ! Contributions from: Peter Caldwell, Xiaohong Liu and Steve Ghan
        !
        ! Separated from MG 1.5 by B. Eaton.
        ! Separated module switched to MG 2.0 and further changes by S. Santos.
        !
        ! for questions contact Hugh Morrison, Andrew Gettelman
        ! e-mail: morrison@ucar.edu, andrew@ucar.edu
        !
        !--------------------------------------------------------------------------
        !
        ! List of required external functions that must be supplied:
        !   gamma --> standard mathematical gamma function (if gamma is an
        !       intrinsic, define HAVE_GAMMA_INTRINSICS)
        !
        !--------------------------------------------------------------------------
        !
        ! Constants that must be specified in the "init" method (module variables):
        !
        ! kind            kind of reals (to verify correct linkage only) -
        ! gravit          acceleration due to gravity                    m s-2
        ! rair            dry air gas constant for air                   J kg-1 K-1
        ! rh2o            gas constant for water vapor                   J kg-1 K-1
        ! cpair           specific heat at constant pressure for dry air J kg-1 K-1
        ! tmelt           temperature of melting point for water         K
        ! latvap          latent heat of vaporization                    J kg-1
        ! latice          latent heat of fusion                          J kg-1
        !
        !--------------------------------------------------------------------------
        USE shr_spfn_mod, ONLY: gamma => shr_spfn_gamma
        IMPLICIT NONE
        PRIVATE
        PUBLIC size_dist_param_liq, rising_factorial, size_dist_param_basic, kk2000_liq_autoconversion, ice_autoconversion, &
        immersion_freezing, contact_freezing, snow_self_aggregation, accrete_cloud_water_snow, secondary_ice_production, &
        accrete_rain_snow, heterogeneous_rain_freezing, accrete_cloud_water_rain, self_collection_rain, accrete_cloud_ice_snow, &
        evaporate_sublimate_precip, bergeron_process_snow, ice_deposition_sublimation, avg_diameter
        ! 8 byte real and integer
        INTEGER, parameter, public :: r8 = selected_real_kind(12)
        INTEGER, parameter, public :: i8 = selected_int_kind(18)
        PUBLIC mghydrometeorprops
        TYPE mghydrometeorprops
            ! Density (kg/m^3)
            REAL(KIND=r8) :: rho
            ! Information for size calculations.
            ! Basic calculation of mean size is:
            !     lambda = (shape_coef*nic/qic)^(1/eff_dim)
            ! Then lambda is constrained by bounds.
            REAL(KIND=r8) :: eff_dim
            REAL(KIND=r8) :: shape_coef
            REAL(KIND=r8) :: lambda_bounds(2)
            ! Minimum average particle mass (kg).
            ! Limit is applied at the beginning of the size distribution calculations.
            REAL(KIND=r8) :: min_mean_mass
        END TYPE mghydrometeorprops

        TYPE(mghydrometeorprops), public :: mg_liq_props
        TYPE(mghydrometeorprops), public :: mg_ice_props
        TYPE(mghydrometeorprops), public :: mg_rain_props
        TYPE(mghydrometeorprops), public :: mg_snow_props
        !=================================================
        ! Public module parameters (mostly for MG itself)
        !=================================================
        ! Pi to 20 digits; more than enough to reach the limit of double precision.
        REAL(KIND=r8), parameter, public :: pi = 3.14159265358979323846_r8
        ! "One minus small number": number near unity for round-off issues.
        REAL(KIND=r8), parameter, public :: omsm   = 1._r8 - 1.e-5_r8
        ! Smallest mixing ratio considered in microphysics.
        REAL(KIND=r8), parameter, public :: qsmall = 1.e-18_r8
        ! minimum allowed cloud fraction
        REAL(KIND=r8), parameter, public :: mincld = 0.0001_r8
        REAL(KIND=r8), parameter, public :: rhosn = 250._r8 ! bulk density snow
        REAL(KIND=r8), parameter, public :: rhoi = 500._r8 ! bulk density ice
        REAL(KIND=r8), parameter, public :: rhow = 1000._r8 ! bulk density liquid
        REAL(KIND=r8), parameter, public :: rhows = 917._r8 ! bulk density water solid
        ! fall speed parameters, V = aD^b (V is in m/s)
        ! droplets
        REAL(KIND=r8), parameter, public :: bc = 2._r8
        ! snow
        REAL(KIND=r8), parameter, public :: as = 11.72_r8
        REAL(KIND=r8), parameter, public :: bs = 0.41_r8
        ! cloud ice
        REAL(KIND=r8), parameter, public :: ai = 700._r8
        REAL(KIND=r8), parameter, public :: bi = 1._r8
        ! rain
        REAL(KIND=r8), parameter, public :: ar = 841.99667_r8
        REAL(KIND=r8), parameter, public :: br = 0.8_r8
        ! mass of new crystal due to aerosol freezing and growth (kg)
        REAL(KIND=r8), parameter, public :: mi0 = 4._r8/3._r8*pi*rhoi*(10.e-6_r8)**3
        !=================================================
        ! Private module parameters
        !=================================================
        ! Signaling NaN bit pattern that represents a limiter that's turned off.
        INTEGER(KIND=i8), parameter :: limiter_off = int(z'7FF1111111111111', i8)
        ! alternate threshold used for some in-cloud mmr
        REAL(KIND=r8), parameter :: icsmall = 1.e-8_r8
        ! particle mass-diameter relationship
        ! currently we assume spherical particles for cloud ice/snow
        ! m = cD^d
        ! exponent
        ! Bounds for mean diameter for different constituents.
        ! Minimum average mass of particles.
        ! ventilation parameters
        ! for snow
        REAL(KIND=r8), parameter :: f1s = 0.86_r8
        REAL(KIND=r8), parameter :: f2s = 0.28_r8
        ! for rain
        REAL(KIND=r8), parameter :: f1r = 0.78_r8
        REAL(KIND=r8), parameter :: f2r = 0.308_r8
        ! collection efficiencies
        ! aggregation of cloud ice and snow
        REAL(KIND=r8), parameter :: eii = 0.5_r8
        ! immersion freezing parameters, bigg 1953
        REAL(KIND=r8), parameter :: bimm = 100._r8
        REAL(KIND=r8), parameter :: aimm = 0.66_r8
        ! Mass of each raindrop created from autoconversion.
        REAL(KIND=r8), parameter :: droplet_mass_25um = 4._r8/3._r8*pi*rhow*(25.e-6_r8)**3
        !=========================================================
        ! Constants set in initialization
        !=========================================================
        ! Set using arguments to micro_mg_init
        REAL(KIND=r8) :: rv ! water vapor gas constant
        REAL(KIND=r8) :: cpp ! specific heat of dry air
        REAL(KIND=r8) :: tmelt ! freezing point of water (K)
        ! latent heats of:
        REAL(KIND=r8) :: xxlv ! vaporization
        ! freezing
        REAL(KIND=r8) :: xxls ! sublimation
        ! additional constants to help speed up code
        REAL(KIND=r8) :: gamma_bs_plus3
        REAL(KIND=r8) :: gamma_half_br_plus5
        REAL(KIND=r8) :: gamma_half_bs_plus5
        !=========================================================
        ! Utilities that are cheaper if the compiler knows that
        ! some argument is an integer.
        !=========================================================

        INTERFACE rising_factorial
            MODULE PROCEDURE rising_factorial_r8
            MODULE PROCEDURE rising_factorial_integer
        END INTERFACE rising_factorial

        INTERFACE var_coef
            MODULE PROCEDURE var_coef_r8
            MODULE PROCEDURE var_coef_integer
        END INTERFACE var_coef
        !==========================================================================
            PUBLIC kgen_read_externs_micro_mg_utils

        ! read interface
        PUBLIC kgen_read
        INTERFACE kgen_read
            MODULE PROCEDURE kgen_read_mghydrometeorprops
        END INTERFACE kgen_read

        CONTAINS

        ! write subroutines

        ! module extern variables

        SUBROUTINE kgen_read_externs_micro_mg_utils(kgen_unit)
            INTEGER, INTENT(IN) :: kgen_unit
            READ(UNIT=kgen_unit) rv
            READ(UNIT=kgen_unit) cpp
            READ(UNIT=kgen_unit) tmelt
            READ(UNIT=kgen_unit) xxlv
            READ(UNIT=kgen_unit) xxls
            READ(UNIT=kgen_unit) gamma_bs_plus3
            READ(UNIT=kgen_unit) gamma_half_br_plus5
            READ(UNIT=kgen_unit) gamma_half_bs_plus5
            CALL kgen_read_mghydrometeorprops(mg_liq_props, kgen_unit)
            CALL kgen_read_mghydrometeorprops(mg_ice_props, kgen_unit)
            CALL kgen_read_mghydrometeorprops(mg_rain_props, kgen_unit)
            CALL kgen_read_mghydrometeorprops(mg_snow_props, kgen_unit)
        END SUBROUTINE kgen_read_externs_micro_mg_utils

        SUBROUTINE kgen_read_mghydrometeorprops(var, kgen_unit, printvar)
            INTEGER, INTENT(IN) :: kgen_unit
            CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
            TYPE(mghydrometeorprops), INTENT(out) :: var
            READ(UNIT=kgen_unit) var%rho
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%rho **", var%rho
            END IF
            READ(UNIT=kgen_unit) var%eff_dim
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%eff_dim **", var%eff_dim
            END IF
            READ(UNIT=kgen_unit) var%shape_coef
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%shape_coef **", var%shape_coef
            END IF
            READ(UNIT=kgen_unit) var%lambda_bounds
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%lambda_bounds **", var%lambda_bounds
            END IF
            READ(UNIT=kgen_unit) var%min_mean_mass
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%min_mean_mass **", var%min_mean_mass
            END IF
        END SUBROUTINE
        !==========================================================================
        ! Initialize module variables.
        !
        ! "kind" serves no purpose here except to check for unlikely linking
        ! issues; always pass in the kind for a double precision real.
        !
        ! "errstring" is the only output; it is blank if there is no error, or set
        ! to a message if there is an error.
        !
        ! Check the list at the top of this module for descriptions of all other
        ! arguments.

        ! Constructor for a constituent property object.

        !========================================================================
        !FORMULAS
        !========================================================================
        ! Use gamma function to implement rising factorial extended to the reals.

        pure FUNCTION rising_factorial_r8(x, n) RESULT ( res )
            REAL(KIND=r8), intent(in) :: x
            REAL(KIND=r8), intent(in) :: n
            REAL(KIND=r8) :: res
            res = gamma(x+n)/gamma(x)
        END FUNCTION rising_factorial_r8
        ! Rising factorial can be performed much cheaper if n is a small integer.

        pure FUNCTION rising_factorial_integer(x, n) RESULT ( res )
            REAL(KIND=r8), intent(in) :: x
            INTEGER, intent(in) :: n
            REAL(KIND=r8) :: res
            INTEGER :: i
            REAL(KIND=r8) :: factor
            res = 1._r8
            factor = x
            DO i = 1, n
                res = res * factor
                factor = factor + 1._r8
            END DO 
        END FUNCTION rising_factorial_integer
        ! Calculate correction due to latent heat for evaporation/sublimation

        elemental FUNCTION calc_ab(t, qv, xxl) RESULT ( ab )
            REAL(KIND=r8), intent(in) :: t ! Temperature
            REAL(KIND=r8), intent(in) :: qv ! Saturation vapor pressure
            REAL(KIND=r8), intent(in) :: xxl ! Latent heat
            REAL(KIND=r8) :: ab
            REAL(KIND=r8) :: dqsdt
            dqsdt = xxl*qv / (rv * t**2)
            ab = 1._r8 + dqsdt*xxl/cpp
        END FUNCTION calc_ab
        ! get cloud droplet size distribution parameters

        elemental SUBROUTINE size_dist_param_liq(props, qcic, ncic, rho, pgam, lamc)
            TYPE(mghydrometeorprops), intent(in) :: props
            REAL(KIND=r8), intent(in) :: qcic
            REAL(KIND=r8), intent(inout) :: ncic
            REAL(KIND=r8), intent(in) :: rho
            REAL(KIND=r8), intent(out) :: pgam
            REAL(KIND=r8), intent(out) :: lamc
            TYPE(mghydrometeorprops) :: props_loc
            IF (qcic > qsmall) THEN
                ! Local copy of properties that can be modified.
                ! (Elemental routines that operate on arrays can't modify scalar
                ! arguments.)
                props_loc = props
                ! Get pgam from fit to observations of martin et al. 1994
                pgam = 0.0005714_r8*1.e-6_r8*ncic*rho + 0.2714_r8
                pgam = 1._r8/(pgam**2) - 1._r8
                pgam = max(pgam, 2._r8)
                ! Set coefficient for use in size_dist_param_basic.
                ! The 3D case is so common and optimizable that we specialize it:
                IF (props_loc%eff_dim == 3._r8) THEN
                    props_loc%shape_coef = pi / 6._r8 * props_loc%rho *              rising_factorial(pgam+1._r8, 3)
                    ELSE
                    props_loc%shape_coef = pi / 6._r8 * props_loc%rho *              rising_factorial(pgam+1._r8, &
                    props_loc%eff_dim)
                END IF 
                ! Limit to between 2 and 50 microns mean size.
                props_loc%lambda_bounds = (pgam+1._r8)*1._r8/[50.e-6_r8, 2.e-6_r8]
                CALL size_dist_param_basic(props_loc, qcic, ncic, lamc)
                ELSE
                ! pgam not calculated in this case, so set it to a value likely to
                ! cause an error if it is accidentally used
                ! (gamma function undefined for negative integers)
                pgam = -100._r8
                lamc = 0._r8
            END IF 
        END SUBROUTINE size_dist_param_liq
        ! Basic routine for getting size distribution parameters.

        elemental SUBROUTINE size_dist_param_basic(props, qic, nic, lam, n0)
            TYPE(mghydrometeorprops), intent(in) :: props
            REAL(KIND=r8), intent(in) :: qic
            REAL(KIND=r8), intent(inout) :: nic
            REAL(KIND=r8), intent(out) :: lam
            REAL(KIND=r8), intent(out), optional :: n0
            IF (qic > qsmall) THEN
                ! add upper limit to in-cloud number concentration to prevent
                ! numerical error
                IF (limiter_is_on(props%min_mean_mass)) THEN
                    nic = min(nic, qic / props%min_mean_mass)
                END IF 
                ! lambda = (c n/q)^(1/d)
                lam = (props%shape_coef * nic/qic)**(1._r8/props%eff_dim)
                ! check for slope
                ! adjust vars
                IF (lam < props%lambda_bounds(1)) THEN
                    lam = props%lambda_bounds(1)
                    nic = lam**(props%eff_dim) * qic/props%shape_coef
                    ELSE IF (lam > props%lambda_bounds(2)) THEN
                    lam = props%lambda_bounds(2)
                    nic = lam**(props%eff_dim) * qic/props%shape_coef
                END IF 
                ELSE
                lam = 0._r8
            END IF 
            IF (present(n0)) n0 = nic * lam
        END SUBROUTINE size_dist_param_basic

        elemental real(r8) FUNCTION avg_diameter(q, n, rho_air, rho_sub)
            ! Finds the average diameter of particles given their density, and
            ! mass/number concentrations in the air.
            ! Assumes that diameter follows an exponential distribution.
            REAL(KIND=r8), intent(in) :: q ! mass mixing ratio
            REAL(KIND=r8), intent(in) :: n ! number concentration (per volume)
            REAL(KIND=r8), intent(in) :: rho_air ! local density of the air
            REAL(KIND=r8), intent(in) :: rho_sub ! density of the particle substance
            avg_diameter = (pi * rho_sub * n/(q*rho_air))**(-1._r8/3._r8)
        END FUNCTION avg_diameter

        elemental FUNCTION var_coef_r8(relvar, a) RESULT ( res )
            ! Finds a coefficient for process rates based on the relative variance
            ! of cloud water.
            REAL(KIND=r8), intent(in) :: relvar
            REAL(KIND=r8), intent(in) :: a
            REAL(KIND=r8) :: res
            res = rising_factorial(relvar, a) / relvar**a
        END FUNCTION var_coef_r8

        elemental FUNCTION var_coef_integer(relvar, a) RESULT ( res )
            ! Finds a coefficient for process rates based on the relative variance
            ! of cloud water.
            REAL(KIND=r8), intent(in) :: relvar
            INTEGER, intent(in) :: a
            REAL(KIND=r8) :: res
            res = rising_factorial(relvar, a) / relvar**a
        END FUNCTION var_coef_integer
        !========================================================================
        !MICROPHYSICAL PROCESS CALCULATIONS
        !========================================================================
        !========================================================================
        ! Initial ice deposition and sublimation loop.
        ! Run before the main loop
        ! This subroutine written by Peter Caldwell

        elemental SUBROUTINE ice_deposition_sublimation(t, qv, qi, ni, icldm, rho, dv, qvl, qvi, berg, vap_dep, ice_sublim)
            !INPUT VARS:
            !===============================================
            REAL(KIND=r8), intent(in) :: t
            REAL(KIND=r8), intent(in) :: qv
            REAL(KIND=r8), intent(in) :: qi
            REAL(KIND=r8), intent(in) :: ni
            REAL(KIND=r8), intent(in) :: icldm
            REAL(KIND=r8), intent(in) :: rho
            REAL(KIND=r8), intent(in) :: dv
            REAL(KIND=r8), intent(in) :: qvl
            REAL(KIND=r8), intent(in) :: qvi
            !OUTPUT VARS:
            !===============================================
            REAL(KIND=r8), intent(out) :: vap_dep !ice deposition (cell-ave value)
            REAL(KIND=r8), intent(out) :: ice_sublim !ice sublimation (cell-ave value)
            REAL(KIND=r8), intent(out) :: berg !bergeron enhancement (cell-ave value)
            !INTERNAL VARS:
            !===============================================
            REAL(KIND=r8) :: ab
            REAL(KIND=r8) :: epsi
            REAL(KIND=r8) :: qiic
            REAL(KIND=r8) :: niic
            REAL(KIND=r8) :: lami
            REAL(KIND=r8) :: n0i
            IF (qi>=qsmall) THEN
                !GET IN-CLOUD qi, ni
                !===============================================
                qiic = qi/icldm
                niic = ni/icldm
                !Compute linearized condensational heating correction
                ab = calc_ab(t, qvi, xxls)
                !Get slope and intercept of gamma distn for ice.
                CALL size_dist_param_basic(mg_ice_props, qiic, niic, lami, n0i)
                !Get depletion timescale=1/eps
                epsi = 2._r8*pi*n0i*rho*dv/(lami*lami)
                !Compute deposition/sublimation
                vap_dep = epsi/ab*(qv - qvi)
                !Make this a grid-averaged quantity
                vap_dep = vap_dep*icldm
                !Split into deposition or sublimation.
                IF (t < tmelt .and. vap_dep>0._r8) THEN
                    ice_sublim = 0._r8
                    ELSE
                    ! make ice_sublim negative for consistency with other evap/sub processes
                    ice_sublim = min(vap_dep,0._r8)
                    vap_dep = 0._r8
                END IF 
                !sublimation occurs @ any T. Not so for berg.
                IF (t < tmelt) THEN
                    !Compute bergeron rate assuming cloud for whole step.
                    berg = max(epsi/ab*(qvl - qvi), 0._r8)
                    ELSE !T>frz
                    berg = 0._r8
                END IF  !T<frz
                ELSE !where qi<qsmall
                berg = 0._r8
                vap_dep = 0._r8
                ice_sublim = 0._r8
            END IF  !qi>qsmall
        END SUBROUTINE ice_deposition_sublimation
        !========================================================================
        ! autoconversion of cloud liquid water to rain
        ! formula from Khrouditnov and Kogan (2000), modified for sub-grid distribution of qc
        ! minimum qc of 1 x 10^-8 prevents floating point error

        elemental SUBROUTINE kk2000_liq_autoconversion(microp_uniform, qcic, ncic, rho, relvar, prc, nprc, nprc1)
            LOGICAL, intent(in) :: microp_uniform
            REAL(KIND=r8), intent(in) :: qcic
            REAL(KIND=r8), intent(in) :: ncic
            REAL(KIND=r8), intent(in) :: rho
            REAL(KIND=r8), intent(in) :: relvar
            REAL(KIND=r8), intent(out) :: prc
            REAL(KIND=r8), intent(out) :: nprc
            REAL(KIND=r8), intent(out) :: nprc1
            REAL(KIND=r8) :: prc_coef
            ! Take variance into account, or use uniform value.
            IF (.not. microp_uniform) THEN
                prc_coef = var_coef(relvar, 2.47_r8)
                ELSE
                prc_coef = 1._r8
            END IF 
            IF (qcic >= icsmall) THEN
                ! nprc is increase in rain number conc due to autoconversion
                ! nprc1 is decrease in cloud droplet conc due to autoconversion
                ! assume exponential sub-grid distribution of qc, resulting in additional
                ! factor related to qcvar below
                ! switch for sub-columns, don't include sub-grid qc
                prc = prc_coef *           1350._r8 * qcic**2.47_r8 * (ncic*1.e-6_r8*rho)**(-1.79_r8)
                nprc = prc * (1._r8/droplet_mass_25um)
                nprc1 = prc*ncic/qcic
                ELSE
                prc = 0._r8
                nprc = 0._r8
                nprc1 = 0._r8
            END IF 
        END SUBROUTINE kk2000_liq_autoconversion
        !========================================================================
        ! Autoconversion of cloud ice to snow
        ! similar to Ferrier (1994)

        elemental SUBROUTINE ice_autoconversion(t, qiic, lami, n0i, dcs, prci, nprci)
            REAL(KIND=r8), intent(in) :: t
            REAL(KIND=r8), intent(in) :: qiic
            REAL(KIND=r8), intent(in) :: lami
            REAL(KIND=r8), intent(in) :: n0i
            REAL(KIND=r8), intent(in) :: dcs
            REAL(KIND=r8), intent(out) :: prci
            REAL(KIND=r8), intent(out) :: nprci
            ! Assume autoconversion timescale of 180 seconds.
            REAL(KIND=r8), parameter :: ac_time = 180._r8
            ! Average mass of an ice particle.
            REAL(KIND=r8) :: m_ip
            ! Ratio of autoconversion diameter to average diameter.
            REAL(KIND=r8) :: d_rat
            IF (t <= tmelt .and. qiic >= qsmall) THEN
                d_rat = lami*dcs
                ! Rate of ice particle conversion (number).
                nprci = n0i/(lami*ac_time)*exp(-d_rat)
                m_ip = (rhoi*pi/6._r8) / lami**3
                ! Rate of mass conversion.
                ! Note that this is:
                ! m n (d^3 + 3 d^2 + 6 d + 6)
                prci = m_ip * nprci *           (((d_rat + 3._r8)*d_rat + 6._r8)*d_rat + 6._r8)
                ELSE
                prci = 0._r8
                nprci = 0._r8
            END IF 
        END SUBROUTINE ice_autoconversion
        ! immersion freezing (Bigg, 1953)
        !===================================

        elemental SUBROUTINE immersion_freezing(microp_uniform, t, pgam, lamc, qcic, ncic, relvar, mnuccc, nnuccc)
            LOGICAL, intent(in) :: microp_uniform
            ! Temperature
            REAL(KIND=r8), intent(in) :: t
            ! Cloud droplet size distribution parameters
            REAL(KIND=r8), intent(in) :: pgam
            REAL(KIND=r8), intent(in) :: lamc
            ! MMR and number concentration of in-cloud liquid water
            REAL(KIND=r8), intent(in) :: qcic
            REAL(KIND=r8), intent(in) :: ncic
            ! Relative variance of cloud water
            REAL(KIND=r8), intent(in) :: relvar
            ! Output tendencies
            REAL(KIND=r8), intent(out) :: mnuccc ! MMR
            REAL(KIND=r8), intent(out) :: nnuccc ! Number
            ! Coefficients that will be omitted for sub-columns
            REAL(KIND=r8) :: dum
            IF (.not. microp_uniform) THEN
                dum = var_coef(relvar, 2)
                ELSE
                dum = 1._r8
            END IF 
            IF (qcic >= qsmall .and. t < 269.15_r8) THEN
                nnuccc = pi/6._r8*ncic*rising_factorial(pgam+1._r8, 3)*           bimm*(exp(aimm*(tmelt - t))-1._r8)/lamc**3
                mnuccc = dum * nnuccc *           pi/6._r8*rhow*           rising_factorial(pgam+4._r8, 3)/lamc**3
                ELSE
                mnuccc = 0._r8
                nnuccc = 0._r8
            END IF  ! qcic > qsmall and t < 4 deg C
        END SUBROUTINE immersion_freezing
        ! contact freezing (-40<T<-3 C) (Young, 1974) with hooks into simulated dust
        !===================================================================
        ! dust size and number in multiple bins are read in from companion routine

        pure SUBROUTINE contact_freezing(microp_uniform, t, p, rndst, nacon, pgam, lamc, qcic, ncic, relvar, mnucct, nnucct)
            LOGICAL, intent(in) :: microp_uniform
            REAL(KIND=r8), intent(in) :: t(:) ! Temperature
            REAL(KIND=r8), intent(in) :: p(:) ! Pressure
            REAL(KIND=r8), intent(in) :: rndst(:,:) ! Radius (for multiple dust bins)
            REAL(KIND=r8), intent(in) :: nacon(:,:) ! Number (for multiple dust bins)
            ! Size distribution parameters for cloud droplets
            REAL(KIND=r8), intent(in) :: pgam(:)
            REAL(KIND=r8), intent(in) :: lamc(:)
            ! MMR and number concentration of in-cloud liquid water
            REAL(KIND=r8), intent(in) :: qcic(:)
            REAL(KIND=r8), intent(in) :: ncic(:)
            ! Relative cloud water variance
            REAL(KIND=r8), intent(in) :: relvar(:)
            ! Output tendencies
            REAL(KIND=r8), intent(out) :: mnucct(:) ! MMR
            REAL(KIND=r8), intent(out) :: nnucct(:) ! Number
            REAL(KIND=r8) :: tcnt ! scaled relative temperature
            REAL(KIND=r8) :: viscosity ! temperature-specific viscosity (kg/m/s)
            REAL(KIND=r8) :: mfp ! temperature-specific mean free path (m)
            ! Dimension these according to number of dust bins, inferred from rndst size
            REAL(KIND=r8) :: nslip(size(rndst,2)) ! slip correction factors
            REAL(KIND=r8) :: ndfaer(size(rndst,2)) ! aerosol diffusivities (m^2/sec)
            ! Coefficients not used for subcolumns
            REAL(KIND=r8) :: dum
            REAL(KIND=r8) :: dum1
            ! Common factor between mass and number.
            REAL(KIND=r8) :: contact_factor
            INTEGER :: i
            DO i = 1,size(t)
                IF (qcic(i) >= qsmall .and. t(i) < 269.15_r8) THEN
                    IF (.not. microp_uniform) THEN
                        dum = var_coef(relvar(i), 4._r8/3._r8)
                        dum1 = var_coef(relvar(i), 1._r8/3._r8)
                        ELSE
                        dum = 1._r8
                        dum1 = 1._r8
                    END IF 
                    tcnt = (270.16_r8-t(i))**1.3_r8
                    viscosity = 1.8e-5_r8*(t(i)/298.0_r8)**0.85_r8 ! Viscosity (kg/m/s)
                    mfp = 2.0_r8*viscosity/                      (p(i)*sqrt( 8.0_r8*28.96e-3_r8/(pi*8.314409_r8*t(i)) )) ! Mean free path (m)
                    ! Note that these two are vectors.
                    nslip = 1.0_r8+(mfp/rndst(i,:))*(1.257_r8+(0.4_r8*exp(-(1.1_r8*rndst(i,:)/mfp)))) ! Slip correction factor
                    ndfaer = 1.381e-23_r8*t(i)*nslip/(6._r8*pi*viscosity*rndst(i,:)) ! aerosol diffusivity (m2/s)
                    contact_factor = dot_product(ndfaer,nacon(i,:)*tcnt) * pi *              ncic(i) * (pgam(i) + 1._r8) / lamc(i)
                    mnucct(i) = dum * contact_factor *              pi/3._r8*rhow*rising_factorial(pgam(i)+2._r8, 3)/lamc(i)**3
                    nnucct(i) = dum1 * 2._r8 * contact_factor
                    ELSE
                    mnucct(i) = 0._r8
                    nnucct(i) = 0._r8
                END IF  ! qcic > qsmall and t < 4 deg C
            END DO 
        END SUBROUTINE contact_freezing
        ! snow self-aggregation from passarelli, 1978, used by reisner, 1998
        !===================================================================
        ! this is hard-wired for bs = 0.4 for now
        ! ignore self-collection of cloud ice

        elemental SUBROUTINE snow_self_aggregation(t, rho, asn, rhosn, qsic, nsic, nsagg)
            REAL(KIND=r8), intent(in) :: t ! Temperature
            REAL(KIND=r8), intent(in) :: rho ! Density
            REAL(KIND=r8), intent(in) :: asn ! fall speed parameter for snow
            REAL(KIND=r8), intent(in) :: rhosn ! density of snow
            ! In-cloud snow
            REAL(KIND=r8), intent(in) :: qsic ! MMR
            REAL(KIND=r8), intent(in) :: nsic ! Number
            ! Output number tendency
            REAL(KIND=r8), intent(out) :: nsagg
            IF (qsic >= qsmall .and. t <= tmelt) THEN
                nsagg = -1108._r8*eii/(4._r8*720._r8*rhosn)*asn*qsic*nsic*rho*          ((qsic/nsic)*(1._r8/(rhosn*pi)))**((&
                bs-1._r8)/3._r8)
                ELSE
                nsagg = 0._r8
            END IF 
        END SUBROUTINE snow_self_aggregation
        ! accretion of cloud droplets onto snow/graupel
        !===================================================================
        ! here use continuous collection equation with
        ! simple gravitational collection kernel
        ! ignore collisions between droplets/cloud ice
        ! since minimum size ice particle for accretion is 50 - 150 micron

        elemental SUBROUTINE accrete_cloud_water_snow(t, rho, asn, uns, mu, qcic, ncic, qsic, pgam, lamc, lams, n0s, psacws, &
        npsacws)
            REAL(KIND=r8), intent(in) :: t ! Temperature
            REAL(KIND=r8), intent(in) :: rho ! Density
            REAL(KIND=r8), intent(in) :: asn ! Fallspeed parameter (snow)
            REAL(KIND=r8), intent(in) :: uns ! Current fallspeed   (snow)
            REAL(KIND=r8), intent(in) :: mu ! Viscosity
            ! In-cloud liquid water
            REAL(KIND=r8), intent(in) :: qcic ! MMR
            REAL(KIND=r8), intent(in) :: ncic ! Number
            ! In-cloud snow
            REAL(KIND=r8), intent(in) :: qsic ! MMR
            ! Cloud droplet size parameters
            REAL(KIND=r8), intent(in) :: pgam
            REAL(KIND=r8), intent(in) :: lamc
            ! Snow size parameters
            REAL(KIND=r8), intent(in) :: lams
            REAL(KIND=r8), intent(in) :: n0s
            ! Output tendencies
            REAL(KIND=r8), intent(out) :: psacws ! Mass mixing ratio
            REAL(KIND=r8), intent(out) :: npsacws ! Number concentration
            REAL(KIND=r8) :: dc0 ! Provisional mean droplet size
            REAL(KIND=r8) :: dum
            REAL(KIND=r8) :: eci ! collection efficiency for riming of snow by droplets
            ! Fraction of cloud droplets accreted per second
            REAL(KIND=r8) :: accrete_rate
            ! ignore collision of snow with droplets above freezing
            IF (qsic >= qsmall .and. t <= tmelt .and. qcic >= qsmall) THEN
                ! put in size dependent collection efficiency
                ! mean diameter of snow is area-weighted, since
                ! accretion is function of crystal geometric area
                ! collection efficiency is approximation based on stoke's law (Thompson et al. 2004)
                dc0 = (pgam+1._r8)/lamc
                dum = dc0*dc0*uns*rhow*lams/(9._r8*mu)
                eci = dum*dum/((dum+0.4_r8)*(dum+0.4_r8))
                eci = max(eci,0._r8)
                eci = min(eci,1._r8)
                ! no impact of sub-grid distribution of qc since psacws
                ! is linear in qc
                accrete_rate = pi/4._r8*asn*rho*n0s*eci*gamma_bs_plus3 / lams**(bs+3._r8)
                psacws = accrete_rate*qcic
                npsacws = accrete_rate*ncic
                ELSE
                psacws = 0._r8
                npsacws = 0._r8
            END IF 
        END SUBROUTINE accrete_cloud_water_snow
        ! add secondary ice production due to accretion of droplets by snow
        !===================================================================
        ! (Hallet-Mossop process) (from Cotton et al., 1986)

        elemental SUBROUTINE secondary_ice_production(t, psacws, msacwi, nsacwi)
            REAL(KIND=r8), intent(in) :: t ! Temperature
            ! Accretion of cloud water to snow tendencies
            REAL(KIND=r8), intent(inout) :: psacws ! MMR
            ! Output (ice) tendencies
            REAL(KIND=r8), intent(out) :: msacwi ! MMR
            REAL(KIND=r8), intent(out) :: nsacwi ! Number
            IF ((t < 270.16_r8) .and. (t >= 268.16_r8)) THEN
                nsacwi = 3.5e8_r8*(270.16_r8-t)/2.0_r8*psacws
                ELSE IF ((t < 268.16_r8) .and. (t >= 265.16_r8)) THEN
                nsacwi = 3.5e8_r8*(t-265.16_r8)/3.0_r8*psacws
                ELSE
                nsacwi = 0.0_r8
            END IF 
            msacwi = min(nsacwi*mi0, psacws)
            psacws = psacws - msacwi
        END SUBROUTINE secondary_ice_production
        ! accretion of rain water by snow
        !===================================================================
        ! formula from ikawa and saito, 1991, used by reisner et al., 1998

        elemental SUBROUTINE accrete_rain_snow(t, rho, umr, ums, unr, uns, qric, qsic, lamr, n0r, lams, n0s, pracs, npracs)
            REAL(KIND=r8), intent(in) :: t ! Temperature
            REAL(KIND=r8), intent(in) :: rho ! Density
            ! Fallspeeds
            ! mass-weighted
            REAL(KIND=r8), intent(in) :: umr ! rain
            REAL(KIND=r8), intent(in) :: ums ! snow
            ! number-weighted
            REAL(KIND=r8), intent(in) :: unr ! rain
            REAL(KIND=r8), intent(in) :: uns ! snow
            ! In cloud MMRs
            REAL(KIND=r8), intent(in) :: qric ! rain
            REAL(KIND=r8), intent(in) :: qsic ! snow
            ! Size distribution parameters
            ! rain
            REAL(KIND=r8), intent(in) :: lamr
            REAL(KIND=r8), intent(in) :: n0r
            ! snow
            REAL(KIND=r8), intent(in) :: lams
            REAL(KIND=r8), intent(in) :: n0s
            ! Output tendencies
            REAL(KIND=r8), intent(out) :: pracs ! MMR
            REAL(KIND=r8), intent(out) :: npracs ! Number
            ! Collection efficiency for accretion of rain by snow
            REAL(KIND=r8), parameter :: ecr = 1.0_r8
            ! Ratio of average snow diameter to average rain diameter.
            REAL(KIND=r8) :: d_rat
            ! Common factor between mass and number expressions
            REAL(KIND=r8) :: common_factor
            IF (qric >= icsmall .and. qsic >= icsmall .and. t <= tmelt) THEN
                common_factor = pi*ecr*rho*n0r*n0s/(lamr**3 * lams)
                d_rat = lamr/lams
                pracs = common_factor*pi*rhow*           sqrt((1.2_r8*umr-0.95_r8*ums)**2 + 0.08_r8*ums*umr) / lamr**3 *          &
                 ((0.5_r8*d_rat + 2._r8)*d_rat + 5._r8)
                npracs = common_factor*0.5_r8*           sqrt(1.7_r8*(unr-uns)**2 + 0.3_r8*unr*uns) *           ((d_rat + 1._r8)&
                *d_rat + 1._r8)
                ELSE
                pracs = 0._r8
                npracs = 0._r8
            END IF 
        END SUBROUTINE accrete_rain_snow
        ! heterogeneous freezing of rain drops
        !===================================================================
        ! follows from Bigg (1953)

        elemental SUBROUTINE heterogeneous_rain_freezing(t, qric, nric, lamr, mnuccr, nnuccr)
            REAL(KIND=r8), intent(in) :: t ! Temperature
            ! In-cloud rain
            REAL(KIND=r8), intent(in) :: qric ! MMR
            REAL(KIND=r8), intent(in) :: nric ! Number
            REAL(KIND=r8), intent(in) :: lamr ! size parameter
            ! Output tendencies
            REAL(KIND=r8), intent(out) :: mnuccr ! MMR
            REAL(KIND=r8), intent(out) :: nnuccr ! Number
            IF (t < 269.15_r8 .and. qric >= qsmall) THEN
                nnuccr = pi*nric*bimm*           (exp(aimm*(tmelt - t))-1._r8)/lamr**3
                mnuccr = nnuccr * 20._r8*pi*rhow/lamr**3
                ELSE
                mnuccr = 0._r8
                nnuccr = 0._r8
            END IF 
        END SUBROUTINE heterogeneous_rain_freezing
        ! accretion of cloud liquid water by rain
        !===================================================================
        ! formula from Khrouditnov and Kogan (2000)
        ! gravitational collection kernel, droplet fall speed neglected

        elemental SUBROUTINE accrete_cloud_water_rain(microp_uniform, qric, qcic, ncic, relvar, accre_enhan, pra, npra)
            LOGICAL, intent(in) :: microp_uniform
            ! In-cloud rain
            REAL(KIND=r8), intent(in) :: qric ! MMR
            ! Cloud droplets
            REAL(KIND=r8), intent(in) :: qcic ! MMR
            REAL(KIND=r8), intent(in) :: ncic ! Number
            ! SGS variability
            REAL(KIND=r8), intent(in) :: relvar
            REAL(KIND=r8), intent(in) :: accre_enhan
            ! Output tendencies
            REAL(KIND=r8), intent(out) :: pra ! MMR
            REAL(KIND=r8), intent(out) :: npra ! Number
            ! Coefficient that varies for subcolumns
            REAL(KIND=r8) :: pra_coef
            IF (.not. microp_uniform) THEN
                pra_coef = accre_enhan * var_coef(relvar, 1.15_r8)
                ELSE
                pra_coef = 1._r8
            END IF 
            IF (qric >= qsmall .and. qcic >= qsmall) THEN
                ! include sub-grid distribution of cloud water
                pra = pra_coef * 67._r8*(qcic*qric)**1.15_r8
                npra = pra*ncic/qcic
                ELSE
                pra = 0._r8
                npra = 0._r8
            END IF 
        END SUBROUTINE accrete_cloud_water_rain
        ! Self-collection of rain drops
        !===================================================================
        ! from Beheng(1994)

        elemental SUBROUTINE self_collection_rain(rho, qric, nric, nragg)
            REAL(KIND=r8), intent(in) :: rho ! Air density
            ! Rain
            REAL(KIND=r8), intent(in) :: qric ! MMR
            REAL(KIND=r8), intent(in) :: nric ! Number
            ! Output number tendency
            REAL(KIND=r8), intent(out) :: nragg
            IF (qric >= qsmall) THEN
                nragg = -8._r8*nric*qric*rho
                ELSE
                nragg = 0._r8
            END IF 
        END SUBROUTINE self_collection_rain
        ! Accretion of cloud ice by snow
        !===================================================================
        ! For this calculation, it is assumed that the Vs >> Vi
        ! and Ds >> Di for continuous collection

        elemental SUBROUTINE accrete_cloud_ice_snow(t, rho, asn, qiic, niic, qsic, lams, n0s, prai, nprai)
            REAL(KIND=r8), intent(in) :: t ! Temperature
            REAL(KIND=r8), intent(in) :: rho ! Density
            REAL(KIND=r8), intent(in) :: asn ! Snow fallspeed parameter
            ! Cloud ice
            REAL(KIND=r8), intent(in) :: qiic ! MMR
            REAL(KIND=r8), intent(in) :: niic ! Number
            REAL(KIND=r8), intent(in) :: qsic ! Snow MMR
            ! Snow size parameters
            REAL(KIND=r8), intent(in) :: lams
            REAL(KIND=r8), intent(in) :: n0s
            ! Output tendencies
            REAL(KIND=r8), intent(out) :: prai ! MMR
            REAL(KIND=r8), intent(out) :: nprai ! Number
            ! Fraction of cloud ice particles accreted per second
            REAL(KIND=r8) :: accrete_rate
            IF (qsic >= qsmall .and. qiic >= qsmall .and. t <= tmelt) THEN
                accrete_rate = pi/4._r8 * eii * asn * rho * n0s * gamma_bs_plus3/           lams**(bs+3._r8)
                prai = accrete_rate * qiic
                nprai = accrete_rate * niic
                ELSE
                prai = 0._r8
                nprai = 0._r8
            END IF 
        END SUBROUTINE accrete_cloud_ice_snow
        ! calculate evaporation/sublimation of rain and snow
        !===================================================================
        ! note: evaporation/sublimation occurs only in cloud-free portion of grid cell
        ! in-cloud condensation/deposition of rain and snow is neglected
        ! except for transfer of cloud water to snow through bergeron process

        elemental SUBROUTINE evaporate_sublimate_precip(t, rho, dv, mu, sc, q, qvl, qvi, lcldm, precip_frac, arn, asn, qcic, qiic,&
         qric, qsic, lamr, n0r, lams, n0s, pre, prds)
            REAL(KIND=r8), intent(in) :: t ! temperature
            REAL(KIND=r8), intent(in) :: rho ! air density
            REAL(KIND=r8), intent(in) :: dv ! water vapor diffusivity
            REAL(KIND=r8), intent(in) :: mu ! viscosity
            REAL(KIND=r8), intent(in) :: sc ! schmidt number
            REAL(KIND=r8), intent(in) :: q ! humidity
            REAL(KIND=r8), intent(in) :: qvl ! saturation humidity (water)
            REAL(KIND=r8), intent(in) :: qvi ! saturation humidity (ice)
            REAL(KIND=r8), intent(in) :: lcldm ! liquid cloud fraction
            REAL(KIND=r8), intent(in) :: precip_frac ! precipitation fraction (maximum overlap)
            ! fallspeed parameters
            REAL(KIND=r8), intent(in) :: arn ! rain
            REAL(KIND=r8), intent(in) :: asn ! snow
            ! In-cloud MMRs
            REAL(KIND=r8), intent(in) :: qcic ! cloud liquid
            REAL(KIND=r8), intent(in) :: qiic ! cloud ice
            REAL(KIND=r8), intent(in) :: qric ! rain
            REAL(KIND=r8), intent(in) :: qsic ! snow
            ! Size parameters
            ! rain
            REAL(KIND=r8), intent(in) :: lamr
            REAL(KIND=r8), intent(in) :: n0r
            ! snow
            REAL(KIND=r8), intent(in) :: lams
            REAL(KIND=r8), intent(in) :: n0s
            ! Output tendencies
            REAL(KIND=r8), intent(out) :: pre
            REAL(KIND=r8), intent(out) :: prds
            REAL(KIND=r8) :: qclr ! water vapor mixing ratio in clear air
            REAL(KIND=r8) :: ab ! correction to account for latent heat
            REAL(KIND=r8) :: eps ! 1/ sat relaxation timescale
            REAL(KIND=r8) :: dum
            ! set temporary cloud fraction to zero if cloud water + ice is very small
            ! this will ensure that evaporation/sublimation of precip occurs over
            ! entire grid cell, since min cloud fraction is specified otherwise
            IF (qcic+qiic < 1.e-6_r8) THEN
                dum = 0._r8
                ELSE
                dum = lcldm
            END IF 
            ! only calculate if there is some precip fraction > cloud fraction
            IF (precip_frac > dum) THEN
                ! calculate q for out-of-cloud region
                qclr = (q-dum*qvl)/(1._r8-dum)
                ! evaporation of rain
                IF (qric >= qsmall) THEN
                    ab = calc_ab(t, qvl, xxlv)
                    eps = 2._r8*pi*n0r*rho*dv*              (f1r/(lamr*lamr)+              f2r*(arn*rho/mu)**0.5_r8*              &
                    sc**(1._r8/3._r8)*gamma_half_br_plus5/              (lamr**(5._r8/2._r8+br/2._r8)))
                    pre = eps*(qclr-qvl)/ab
                    ! only evaporate in out-of-cloud region
                    ! and distribute across precip_frac
                    pre = min(pre*(precip_frac-dum),0._r8)
                    pre = pre/precip_frac
                    ELSE
                    pre = 0._r8
                END IF 
                ! sublimation of snow
                IF (qsic >= qsmall) THEN
                    ab = calc_ab(t, qvi, xxls)
                    eps = 2._r8*pi*n0s*rho*dv*              (f1s/(lams*lams)+              f2s*(asn*rho/mu)**0.5_r8*              &
                    sc**(1._r8/3._r8)*gamma_half_bs_plus5/              (lams**(5._r8/2._r8+bs/2._r8)))
                    prds = eps*(qclr-qvi)/ab
                    ! only sublimate in out-of-cloud region and distribute over precip_frac
                    prds = min(prds*(precip_frac-dum),0._r8)
                    prds = prds/precip_frac
                    ELSE
                    prds = 0._r8
                END IF 
                ELSE
                prds = 0._r8
                pre = 0._r8
            END IF 
        END SUBROUTINE evaporate_sublimate_precip
        ! bergeron process - evaporation of droplets and deposition onto snow
        !===================================================================

        elemental SUBROUTINE bergeron_process_snow(t, rho, dv, mu, sc, qvl, qvi, asn, qcic, qsic, lams, n0s, bergs)
            REAL(KIND=r8), intent(in) :: t ! temperature
            REAL(KIND=r8), intent(in) :: rho ! air density
            REAL(KIND=r8), intent(in) :: dv ! water vapor diffusivity
            REAL(KIND=r8), intent(in) :: mu ! viscosity
            REAL(KIND=r8), intent(in) :: sc ! schmidt number
            REAL(KIND=r8), intent(in) :: qvl ! saturation humidity (water)
            REAL(KIND=r8), intent(in) :: qvi ! saturation humidity (ice)
            ! fallspeed parameter for snow
            REAL(KIND=r8), intent(in) :: asn
            ! In-cloud MMRs
            REAL(KIND=r8), intent(in) :: qcic ! cloud liquid
            REAL(KIND=r8), intent(in) :: qsic ! snow
            ! Size parameters for snow
            REAL(KIND=r8), intent(in) :: lams
            REAL(KIND=r8), intent(in) :: n0s
            ! Output tendencies
            REAL(KIND=r8), intent(out) :: bergs
            REAL(KIND=r8) :: ab ! correction to account for latent heat
            REAL(KIND=r8) :: eps ! 1/ sat relaxation timescale
            IF (qsic >= qsmall.and. qcic >= qsmall .and. t < tmelt) THEN
                ab = calc_ab(t, qvi, xxls)
                eps = 2._r8*pi*n0s*rho*dv*           (f1s/(lams*lams)+           f2s*(asn*rho/mu)**0.5_r8*           sc**(&
                1._r8/3._r8)*gamma_half_bs_plus5/           (lams**(5._r8/2._r8+bs/2._r8)))
                bergs = eps*(qvl-qvi)/ab
                ELSE
                bergs = 0._r8
            END IF 
        END SUBROUTINE bergeron_process_snow
        !========================================================================
        !UTILITIES
        !========================================================================


        pure FUNCTION limiter_is_on(lim)
            REAL(KIND=r8), intent(in) :: lim
            LOGICAL :: limiter_is_on
            limiter_is_on = transfer(lim, limiter_off) /= limiter_off
        END FUNCTION limiter_is_on
    END MODULE micro_mg_utils
