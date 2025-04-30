
! KGEN-generated Fortran source file
!
! Filename    : chem_mods.F90
! Generated at: 2015-07-15 10:35:30
! KGEN version: 0.4.13



    MODULE chem_mods
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        !--------------------------------------------------------------
        ! ... Basic chemistry parameters and arrays
        !--------------------------------------------------------------
        IMPLICIT NONE
        INTEGER, parameter :: nzcnt = 1509 ! number of photolysis reactions
        ! number of total reactions
        ! number of gas phase reactions
        ! number of absorbing column densities
        ! number of "gas phase" species
        ! number of "fixed" species
        ! number of relationship species
        ! number of group members
        ! number of non-zero matrix entries
        ! number of species with external forcing
        ! number of species in explicit class
        ! number of species in hov class
        ! number of species in ebi class
        ! number of species in implicit class
        ! number of species in rodas class
        ! index of total atm density in invariant array
        ! index of water vapor density
        ! loop length for implicit chemistry

    ! write subroutines
    ! No subroutines
    ! No module extern variables
    END MODULE chem_mods
