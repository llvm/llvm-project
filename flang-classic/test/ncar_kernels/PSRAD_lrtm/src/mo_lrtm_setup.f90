
! KGEN-generated Fortran source file
!
! Filename    : mo_lrtm_setup.f90
! Generated at: 2015-02-19 15:30:32
! KGEN version: 0.4.4



    MODULE mo_lrtm_setup
        USE mo_kind, ONLY: wp
        USE mo_rrtm_params, ONLY: ngptlw
        USE mo_rrtm_params, ONLY: nbndlw
        IMPLICIT NONE
        !
        ! spectra information that is entered at run time
        !
        !< Weights for combining original gpts into reduced gpts
        !< Number of cross-section molecules
        !< Flag for active cross-sections in calculation
        INTEGER, parameter :: ngc(nbndlw) = (/           10,12,16,14,16,8,12,8,12,6,8,8,4,2,2,2/) !< The number of new g-intervals in each band
        INTEGER, parameter :: ngs(nbndlw) = (/           10,22,38,52,68,76,88,96,108,114,122,130,134,136,138,140/) !< The cumulative sum of new g-intervals for each band
        !< The index of each new gpt relative to the orignal
        ! band 1
        ! band 2
        ! band 3
        ! band 4
        ! band 5
        ! band 6
        ! band 7
        ! band 8
        ! band 9
        ! band 10
        ! band 11
        ! band 12
        ! band 13
        ! band 14
        ! band 15
        ! band 16
        !< The number of original gs combined to make new pts
        ! band 1
        ! band 2
        ! band 3
        ! band 4
        ! band 5
        ! band 6
        ! band 7
        ! band 8
        ! band 9
        ! band 10
        ! band 11
        ! band 12
        ! band 13
        ! band 14
        ! band 15
        ! band 16
        INTEGER, parameter :: ngb(ngptlw) = (/        1,1,1,1,1,1,1,1,1,1,        2,2,2,2,2,2,2,2,2,2,2,2,        3,3,3,3,3,3,3,3,&
        3,3,3,3,3,3,3,3,        4,4,4,4,4,4,4,4,4,4,4,4,4,4,        5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,        6,6,6,6,6,6,6,6,      &
          7,7,7,7,7,7,7,7,7,7,7,7,        8,8,8,8,8,8,8,8,        9,9,9,9,9,9,9,9,9,9,9,9,        10,10,10,10,10,10,        11,11,&
        11,11,11,11,11,11,        12,12,12,12,12,12,12,12,        13,13,13,13,        14,14,        15,15,        16,16/) !< The band index for each new g-interval
        ! band 1
        ! band 2
        ! band 3
        ! band 4
        ! band 5
        ! band 6
        ! band 7
        ! band 8
        ! band 9
        ! band 10
        ! band 11
        ! band 12
        ! band 13
        ! band 14
        ! band 15
        ! band 16
        !< RRTM weights for the original 16 g-intervals
        INTEGER, parameter :: nspa(nbndlw) = (/        1,1,9,9,9,1,9,1,9,1,1,9,9,1,9,9/) !< Number of reference atmospheres for lower atmosphere
        INTEGER, parameter :: nspb(nbndlw) = (/        1,1,5,5,5,0,1,1,1,1,1,0,0,1,0,0/) !< Number of reference atmospheres for upper atmosphere
        ! < Number of g intervals in each band
        !< Spectral band lower boundary in wavenumbers
        !< Spectral band upper boundary in wavenumbers
        REAL(KIND=wp), parameter :: delwave(nbndlw)  = (/        340._wp, 150._wp, 130._wp,  70._wp, 120._wp, 160._wp,        &
        100._wp, 100._wp, 210._wp,  90._wp, 320._wp, 280._wp,        170._wp, 130._wp, 220._wp, 650._wp/) !< Spectral band width in wavenumbers
        CONTAINS

        ! read subroutines
        ! **************************************************************************

        !***************************************************************************

        !***************************************************************************

        !***************************************************************************

        !***************************************************************************

        !***************************************************************************

        !***************************************************************************

        !***************************************************************************

        !***************************************************************************

        !***************************************************************************

        !***************************************************************************

        !***************************************************************************

        !***************************************************************************

        !***************************************************************************

        !***************************************************************************

        !***************************************************************************

        !***************************************************************************

        !***************************************************************************
    END MODULE mo_lrtm_setup
