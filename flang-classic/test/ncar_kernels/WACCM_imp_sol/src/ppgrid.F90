
! KGEN-generated Fortran source file
!
! Filename    : ppgrid.F90
! Generated at: 2015-05-13 11:02:22
! KGEN version: 0.4.10



    MODULE ppgrid
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        !-----------------------------------------------------------------------
        !
        ! Purpose:
        ! Initialize physics grid resolution parameters
        !  for a chunked data structure
        !
        ! Author:
        !
        !-----------------------------------------------------------------------
        IMPLICIT NONE
        PRIVATE
        PUBLIC pcols
        PUBLIC pver
        ! Grid point resolution parameters
        INTEGER :: pcols ! number of columns (max)
        ! number of sub-columns (max)
        INTEGER :: pver ! number of vertical levels
        ! pver + 1
        PARAMETER (pcols     = 16)
        PARAMETER (pver      = 70)
        !
        ! start, end indices for chunks owned by a given MPI task
        ! (set in phys_grid_init).
        !
        !
        !

    ! write subroutines
    ! No subroutines
    ! No module extern variables
    END MODULE ppgrid
