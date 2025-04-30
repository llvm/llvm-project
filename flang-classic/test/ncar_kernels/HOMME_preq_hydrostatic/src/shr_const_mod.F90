
! KGEN-generated Fortran source file
!
! Filename    : shr_const_mod.F90
! Generated at: 2015-04-12 19:37:49
! KGEN version: 0.4.9



    MODULE shr_const_mod
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, only : shr_kind_in
        USE shr_kind_mod, only : shr_kind_r8
        INTEGER(KIND=shr_kind_in), parameter, private :: r8 = shr_kind_r8 ! rename for local readability only
        !----------------------------------------------------------------------------
        ! physical constants (all data public)
        !----------------------------------------------------------------------------
        PUBLIC
        ! pi
        ! sec in calendar day ~ sec
        ! sec in siderial day ~ sec
        ! earth rot ~ rad/sec
        ! radius of earth ~ m
        ! acceleration of gravity ~ m/s^2
        ! Stefan-Boltzmann constant ~ W/m^2/K^4
        REAL(KIND=r8), parameter :: shr_const_boltz   = 1.38065e-23_r8 ! Boltzmann's constant ~ J/K/molecule
        REAL(KIND=r8), parameter :: shr_const_avogad  = 6.02214e26_r8 ! Avogadro's number ~ molecules/kmole
        REAL(KIND=r8), parameter :: shr_const_rgas    = shr_const_avogad*shr_const_boltz ! Universal gas constant ~ J/K/kmole
        REAL(KIND=r8), parameter :: shr_const_mwdair  = 28.966_r8 ! molecular weight dry air ~ kg/kmole
        ! molecular weight water vapor
        REAL(KIND=r8), parameter :: shr_const_rdair   = shr_const_rgas/shr_const_mwdair ! Dry air gas constant     ~ J/K/kg
        ! Water vapor gas constant ~ J/K/kg
        ! RWV/RDAIR - 1.0
        ! Von Karman constant
        ! standard pressure ~ pascals
        ! ratio of 13C/12C in Pee Dee Belemnite (C isotope standard)
        ! triple point of fresh water        ~ K
        ! freezing T of fresh water          ~ K
        ! freezing T of salt water  ~ K
        ! density of dry air at STP  ~ kg/m^3
        ! density of fresh water     ~ kg/m^3
        ! density of sea water       ~ kg/m^3
        ! density of ice             ~ kg/m^3
        ! specific heat of dry air   ~ J/kg/K
        ! specific heat of water vap ~ J/kg/K
        ! CPWV/CPDAIR - 1.0
        ! specific heat of fresh h2o ~ J/kg/K
        ! specific heat of sea h2o   ~ J/kg/K
        ! specific heat of fresh ice ~ J/kg/K
        ! latent heat of fusion      ~ J/kg
        ! latent heat of evaporation ~ J/kg
        ! latent heat of sublimation ~ J/kg
        ! ocn ref salinity (psu)
        ! ice ref salinity (psu)
        ! special missing value
        ! min spval tolerance
        ! max spval tolerance
        CONTAINS

        ! write subroutines
        ! No subroutines
        ! No module extern variables
        !-----------------------------------------------------------------------------

        !-----------------------------------------------------------------------------
    END MODULE shr_const_mod
