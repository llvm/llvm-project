
! KGEN-generated Fortran source file
!
! Filename    : chem_mods.F90
! Generated at: 2015-05-13 11:02:22
! KGEN version: 0.4.10



    MODULE chem_mods
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        !--------------------------------------------------------------
        ! ... Basic chemistry parameters and arrays
        !--------------------------------------------------------------
        IMPLICIT NONE
        INTEGER, parameter :: extcnt = 18
        INTEGER, parameter :: gas_pcnst = 158
        INTEGER, parameter :: rxntot = 449
        INTEGER, parameter :: clscnt4 = 135
        INTEGER, parameter :: nzcnt = 1509
        INTEGER, parameter :: nfs = 2
        INTEGER, parameter :: indexm = 1 ! number of photolysis reactions
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
        INTEGER :: cls_rxt_cnt(4,5) = 0
        INTEGER :: clsmap(gas_pcnst,5) = 0
        INTEGER :: permute(gas_pcnst,5) = 0
        PUBLIC kgen_read_externs_chem_mods
    CONTAINS

    ! write subroutines

    ! module extern variables

    SUBROUTINE kgen_read_externs_chem_mods(kgen_unit)
        INTEGER, INTENT(IN) :: kgen_unit
        READ(UNIT=kgen_unit) cls_rxt_cnt
        READ(UNIT=kgen_unit) clsmap
        READ(UNIT=kgen_unit) permute
    END SUBROUTINE kgen_read_externs_chem_mods

    END MODULE chem_mods
