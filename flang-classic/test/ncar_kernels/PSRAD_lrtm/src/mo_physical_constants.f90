
! KGEN-generated Fortran source file
!
! Filename    : mo_physical_constants.f90
! Generated at: 2015-02-19 15:30:36
! KGEN version: 0.4.4



    MODULE mo_physical_constants
        USE mo_kind, ONLY: wp
        IMPLICIT NONE
        PUBLIC
        ! Natural constants
        ! -----------------
        !
        ! WMO/SI values
        !> [1/mo]    Avogadro constant
        !! [J/K]     Boltzmann constant
        !! [J/K/mol] molar/universal/ideal gas constant
        !! [W/m2/K4] Stephan-Boltzmann constant
        !
        !> Molar weights
        !! -------------
        !!
        !! Pure species
        !>[g/mol] CO2 (National Institute for
        !!  Standards and Technology (NIST))
        !! [g/mol] CH4
        !! [g/mol] O3
        !! [g/mol] O2
        !! [g/mol] N2O
        !! [g/mol] CFC11
        !! [g/mol] CFC12
        REAL(KIND=wp), parameter :: amw   = 18.0154_wp !! [g/mol] H2O
        !
        !> Mixed species
        REAL(KIND=wp), parameter :: amd   = 28.970_wp !> [g/mol] dry air
        !
        !> Auxiliary constants
        ! ppmv2gg converts ozone from volume mixing ratio in ppmv
        ! to mass mixing ratio in g/g
        !
        !> Earth and Earth orbit constants
        !! -------------------------------
        !!
        !! [m]    average radius
        !! [1/m]
        !! [1/s]  angular velocity
        !
        ! WMO/SI value
        REAL(KIND=wp), parameter :: grav  = 9.80665_wp !> [m/s2] av. gravitational acceleration
        !! [s2/m]
        !
        !> [m/m]  ratio of atm. scale height
        !                                               !!        to Earth radius
        ! seconds per day
        !
        !> Thermodynamic constants for the dry and moist atmosphere
        !! --------------------------------------------------------
        !
        !> Dry air
        !> [J/K/kg] gas constant
        !! [J/K/kg] specific heat at constant pressure
        !! [J/K/kg] specific heat at constant volume
        !! [m^2/s]  kinematic viscosity of dry air
        !! [m^2/s]  scalar conductivity of dry air
        !! [J/m/s/K]thermal conductivity of dry air
        !! [N*s/m2] dyn viscosity of dry air at tmelt
        !
        !> H2O
        !! - gas
        !> [J/K/kg] gas constant for water vapor
        !! [J/K/kg] specific heat at constant pressure
        !! [J/K/kg] specific heat at constant volume
        !! [m^2/s]  diff coeff of H2O vapor in dry air at tmelt
        !> - liquid / water
        !> [kg/m3]  density of liquid water
        !> H2O related constants  (liquid, ice, snow), phase change constants
        ! echam values
        ! density of sea water in kg/m3
        ! density of ice in kg/m3
        ! density of snow in kg/m3
        ! density ratio (ice/water)
        ! specific heat for liquid water J/K/kg
        ! specific heat for sea water J/K/kg
        ! specific heat for ice J/K/kg
        ! specific heat for snow J/K/kg
        ! thermal conductivity of ice in W/K/m
        ! thermal conductivity of snow in W/K/m
        ! echam values end
        !
        !REAL(wp), PARAMETER :: clw   = 4186.84_wp       !! [J/K/kg] specific heat of water
        !                                                  !!  see below
        !> - phase changes
        !> [J/kg]   latent heat for vaporisation
        !! [J/kg]   latent heat for sublimation
        !! [J/kg]   latent heat for fusion
        !! [K]      melting temperature of ice/snow
        !
        !> Auxiliary constants
        !> [ ]
        ! the next 2 values not as parameters due to ECHAM-dyn
        !! [ ]
        !! [ ]
        !! [ ]
        !! [K]
        !! [K]
        !! [K*kg/J]
        !! [K*kg/J]
        !! cp_d / cp_l - 1
        !
        !> specific heat capacity of liquid water
        !
        !> [ ]
        !! [ ]
        !! [ ]
        !
        !> [Pa]  reference pressure for Exner function
        !> Auxiliary constants used in ECHAM
        ! Constants used for computation of saturation mixing ratio
        ! over liquid water (*c_les*) or ice(*c_ies*)
        !
        !
        !
        !
        !
        !
        !
        !> Variables for computing cloud cover in RH scheme
        !
        !> vertical profile parameters (vpp) of CH4 and N2O
        !
        !> constants for radiation module
        !> lw sfc default emissivity factor
        !
        !---------------------------
        ! Specifications, thresholds, and derived constants for the following subroutines:
        ! s_lake, s_licetemp, s_sicetemp, meltpond, meltpond_ice, update_albedo_ice_meltpond
        !
        ! mixed-layer depth of lakes in m
        ! mixed-layer depth of ocean in m
        ! minimum ice thickness in m
        ! minimum ice thickness of pond ice in m
        ! threshold ice thickness for pond closing in m
        ! minimum pond depth for pond fraction in m
        ! albedo of pond ice
        !
        ! heat capacity of lake mixed layer
        !                                                         !  in J/K/m2
        ! heat capacity of upper ice layer
        ! heat capacity of upper pond ice layer
        !
        ! [J/m3]
        ! [J/m3]
        ! [m/K]
        ! [K/m]
        ! cooling below tmelt required to form dice
        !---------------------------
        !
        !------------below are parameters for ocean model---------------
        ! coefficients in linear EOS
        ! thermal expansion coefficient (kg/m3/K)
        ! haline contraction coefficient (kg/m3/psu)
        !
        ! density reference values, to be constant in Boussinesq ocean models
        ! reference density [kg/m^3]
        ! inverse reference density [m^3/kg]
        ! reference salinity [psu]
        !
        !Conversion from pressure [p] to pressure [bar]
        !                                                    !used in ocean thermodynamics
        !
        ! [Pa]     sea level pressure
        !
        !----------below are parameters for sea-ice model---------------
        ! heat conductivity snow     [J  / (m s K)]
        ! heat conductivity ice      [J  / (m s K)]
        ! density of sea ice         [kg / m3]
        ! density of snow            [kg / m3]
        ! Heat capacity of ice       [J / (kg K)]
        ! Temperature ice bottom     [C]
        ! Sea-ice bulk salinity      [ppt]
        ! Constant in linear freezing-
        !                                             ! point relationship         [C/ppt]
        ! = - (sea-ice liquidus
        !                                             ! (aka melting) temperature) [C]
        !REAL(wp), PARAMETER :: muS = -(-0.0575 + 1.710523E-3*Sqrt(Sice) - 2.154996E-4*Sice) * Sice
        ! Albedo of snow (not melting)
        ! Albedo of snow (melting)
        ! Albedo of ice (not melting)
        ! Albedo of ice (melting)
        ! albedo of the ocean
        !REAL(wp), PARAMETER :: I_0     =    0.3       ! Ice-surface penetrating shortwave fraction
        ! Ice-surface penetrating shortwave fraction
        !------------------------------------------------------------

    ! read subroutines
    END MODULE mo_physical_constants
