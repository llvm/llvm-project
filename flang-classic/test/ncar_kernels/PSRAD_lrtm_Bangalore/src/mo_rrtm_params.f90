
! KGEN-generated Fortran source file
!
! Filename    : mo_rrtm_params.f90
! Generated at: 2015-02-19 15:30:37
! KGEN version: 0.4.4



    MODULE mo_rrtm_params
        USE mo_kind, ONLY: wp
        IMPLICIT NONE
        PUBLIC
        !! -----------------------------------------------------------------------------------------
        !!
        !!  Shared parameters
        !!
        !< number of original g-intervals per spectral band
        INTEGER, parameter :: maxxsec= 4 !< maximum number of cross-section molecules (cfcs)
        INTEGER, parameter :: maxinpx= 38
        !< number of last band (lw and sw share band 16)
        !< number of spectral bands in sw model
        !< total number of gpts
        !< first band in sw
        !< last band in sw
        INTEGER, parameter :: nbndlw = 16 !< number of spectral bands in lw model
        INTEGER, parameter :: ngptlw = 140 !< total number of reduced g-intervals for rrtmg_lw
        !
        ! These pressures are chosen such that the ln of the first pressure
        ! has only a few non-zero digits (i.e. ln(PREF(1)) = 6.96000) and
        ! each subsequent ln(pressure) differs from the previous one by 0.2.
        !
        REAL(KIND=wp), parameter :: preflog(59) = (/        6.9600e+00_wp, 6.7600e+00_wp, 6.5600e+00_wp, 6.3600e+00_wp, &
        6.1600e+00_wp,        5.9600e+00_wp, 5.7600e+00_wp, 5.5600e+00_wp, 5.3600e+00_wp, 5.1600e+00_wp,        4.9600e+00_wp, &
        4.7600e+00_wp, 4.5600e+00_wp, 4.3600e+00_wp, 4.1600e+00_wp,        3.9600e+00_wp, 3.7600e+00_wp, 3.5600e+00_wp, &
        3.3600e+00_wp, 3.1600e+00_wp,        2.9600e+00_wp, 2.7600e+00_wp, 2.5600e+00_wp, 2.3600e+00_wp, 2.1600e+00_wp,        &
        1.9600e+00_wp, 1.7600e+00_wp, 1.5600e+00_wp, 1.3600e+00_wp, 1.1600e+00_wp,        9.6000e-01_wp, 7.6000e-01_wp, &
        5.6000e-01_wp, 3.6000e-01_wp, 1.6000e-01_wp,        -4.0000e-02_wp,-2.4000e-01_wp,-4.4000e-01_wp,-6.4000e-01_wp,&
        -8.4000e-01_wp,        -1.0400e+00_wp,-1.2400e+00_wp,-1.4400e+00_wp,-1.6400e+00_wp,-1.8400e+00_wp,        -2.0400e+00_wp,&
        -2.2400e+00_wp,-2.4400e+00_wp,-2.6400e+00_wp,-2.8400e+00_wp,        -3.0400e+00_wp,-3.2400e+00_wp,-3.4400e+00_wp,&
        -3.6400e+00_wp,-3.8400e+00_wp,        -4.0400e+00_wp,-4.2400e+00_wp,-4.4400e+00_wp,-4.6400e+00_wp /)
        !
        ! These are the temperatures associated with the respective pressures
        !
        REAL(KIND=wp), parameter :: tref(59) = (/        2.9420e+02_wp, 2.8799e+02_wp, 2.7894e+02_wp, 2.6925e+02_wp, &
        2.5983e+02_wp,        2.5017e+02_wp, 2.4077e+02_wp, 2.3179e+02_wp, 2.2306e+02_wp, 2.1578e+02_wp,        2.1570e+02_wp, &
        2.1570e+02_wp, 2.1570e+02_wp, 2.1706e+02_wp, 2.1858e+02_wp,        2.2018e+02_wp, 2.2174e+02_wp, 2.2328e+02_wp, &
        2.2479e+02_wp, 2.2655e+02_wp,        2.2834e+02_wp, 2.3113e+02_wp, 2.3401e+02_wp, 2.3703e+02_wp, 2.4022e+02_wp,        &
        2.4371e+02_wp, 2.4726e+02_wp, 2.5085e+02_wp, 2.5457e+02_wp, 2.5832e+02_wp,        2.6216e+02_wp, 2.6606e+02_wp, &
        2.6999e+02_wp, 2.7340e+02_wp, 2.7536e+02_wp,        2.7568e+02_wp, 2.7372e+02_wp, 2.7163e+02_wp, 2.6955e+02_wp, &
        2.6593e+02_wp,        2.6211e+02_wp, 2.5828e+02_wp, 2.5360e+02_wp, 2.4854e+02_wp, 2.4348e+02_wp,        2.3809e+02_wp, &
        2.3206e+02_wp, 2.2603e+02_wp, 2.2000e+02_wp, 2.1435e+02_wp,        2.0887e+02_wp, 2.0340e+02_wp, 1.9792e+02_wp, &
        1.9290e+02_wp, 1.8809e+02_wp,        1.8329e+02_wp, 1.7849e+02_wp, 1.7394e+02_wp, 1.7212e+02_wp /)

    ! read subroutines
    END MODULE mo_rrtm_params
