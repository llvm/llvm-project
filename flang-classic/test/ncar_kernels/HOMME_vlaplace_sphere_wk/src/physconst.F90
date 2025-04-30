
! KGEN-generated Fortran source file
!
! Filename    : physconst.F90
! Generated at: 2015-04-12 19:17:34
! KGEN version: 0.4.9



    MODULE physconst
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        ! Physical constants.  Use CCSM shared values whenever available.
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        USE shr_const_mod, ONLY: shr_const_rearth
        ! Dimensions and chunk bounds
        IMPLICIT NONE
        PRIVATE
        ! Constants based off share code or defined in physconst
        ! Avogadro's number (molecules/kmole)
        ! Boltzman's constant (J/K/molecule)
        ! sec in calendar day ~ sec
        ! specific heat of dry air (J/K/kg)
        ! specific heat of fresh h2o (J/K/kg)
        ! Von Karman constant
        ! Latent heat of fusion (J/kg)
        ! Latent heat of vaporization (J/kg)
        ! 3.14...
        ! Standard pressure (Pascals)
        ! Universal gas constant (J/K/kmol)
        ! Density of liquid water (STP)
        !special value
        ! Stefan-Boltzmann's constant (W/m^2/K^4)
        ! Triple point temperature of water (K)
        ! Speed of light in a vacuum (m/s)
        ! Planck's constant (J.s)
        ! Molecular weights
        ! molecular weight co2
        ! molecular weight n2o
        ! molecular weight ch4
        ! molecular weight cfc11
        ! molecular weight cfc12
        ! molecular weight O3
        ! modifiable physical constants for aquaplanet
        ! gravitational acceleration (m/s**2)
        ! sec in siderial day ~ sec
        ! molecular weight h2o
        ! specific heat of water vapor (J/K/kg)
        ! molecular weight dry air
        ! radius of earth (m)
        ! Freezing point of water (K)
        !---------------  Variables below here are derived from those above -----------------------
        ! reciprocal of gravit
        REAL(KIND=r8), public :: ra           = 1._r8/shr_const_rearth ! reciprocal of earth radius
        ! earth rot ~ rad/sec
        ! Water vapor gas constant ~ J/K/kg
        ! Dry air gas constant     ~ J/K/kg
        ! ratio of h2o to dry air molecular weights
        ! (rh2o/rair) - 1
        ! CPWV/CPDAIR - 1.0
        ! density of dry air at STP  ~ kg/m^3
        ! R/Cp
        ! Coriolis expansion coeff -> omega/sqrt(0.375)
        !---------------  Variables below here are for WACCM-X -----------------------
        ! composition dependent specific heat at constant pressure
        ! composition dependent gas "constant"
        ! rairv/cpairv
        ! composition dependent atmosphere mean mass
        ! molecular viscosity      kg/m/s
        ! molecular conductivity   J/m/s/K
        !---------------  Variables below here are for turbulent mountain stress -----------------------
        !================================================================================================
            PUBLIC kgen_read_externs_physconst
        CONTAINS

        ! write subroutines
        ! No subroutines

        ! module extern variables

        SUBROUTINE kgen_read_externs_physconst(kgen_unit)
            INTEGER, INTENT(IN) :: kgen_unit
            READ(UNIT=kgen_unit) ra
        END SUBROUTINE kgen_read_externs_physconst

        !================================================================================================

        !==============================================================================
        ! Read namelist variables.

        !===============================================================================

    END MODULE physconst
