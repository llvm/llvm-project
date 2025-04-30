
! KGEN-generated Fortran source file
!
! Filename    : shr_kind_mod.F90
! Generated at: 2015-03-31 09:44:40
! KGEN version: 0.4.5



    MODULE shr_kind_mod
        !----------------------------------------------------------------------------
        ! precision/kind constants add data public
        !----------------------------------------------------------------------------
        PUBLIC
        INTEGER, parameter :: shr_kind_r8 = selected_real_kind(12) ! 8 byte real
        ! 4 byte real
        ! native real
        ! 8 byte integer
        ! 4 byte integer
        INTEGER, parameter :: shr_kind_in = kind(1) ! native integer
        ! short char
        ! mid-sized char
        ! long char
        ! extra-long char
        ! extra-extra-long char

    ! write subroutines
    ! No subroutines
    ! No module extern variables
    END MODULE shr_kind_mod
