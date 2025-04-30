
! KGEN-generated Fortran source file
!
! Filename    : parrrtm.f90
! Generated at: 2015-07-06 23:28:43
! KGEN version: 0.4.13



    MODULE parrrtm
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        !      use parkind ,only : jpim, jprb
        IMPLICIT NONE
        !------------------------------------------------------------------
        ! rrtmg_lw main parameters
        !
        ! Initial version:  JJMorcrette, ECMWF, Jul 1998
        ! Revised: MJIacono, AER, Jun 2006
        ! Revised: MJIacono, AER, Aug 2007
        !------------------------------------------------------------------
        !  name     type     purpose
        ! -----  :  ----   : ----------------------------------------------
        ! mxlay  :  integer: maximum number of layers
        ! mg     :  integer: number of original g-intervals per spectral band
        ! nbndlw :  integer: number of spectral bands
        ! maxxsec:  integer: maximum number of cross-section molecules
        !                    (e.g. cfcs)
        ! maxinpx:  integer:
        ! ngptlw :  integer: total number of reduced g-intervals for rrtmg_lw
        ! ngNN   :  integer: number of reduced g-intervals per spectral band
        ! ngsNN  :  integer: cumulative number of g-intervals per band
        !------------------------------------------------------------------
        INTEGER, parameter :: nbndlw = 16
        INTEGER, parameter :: maxxsec= 4
        INTEGER, parameter :: mxmol  = 38
        INTEGER, parameter :: maxinpx= 38
        INTEGER, parameter :: nmol   = 7
        ! Use for 140 g-point model
        INTEGER, parameter :: ngptlw = 140
        ! Use for 256 g-point model
        !      integer, parameter :: ngptlw = 256
        ! Use for 140 g-point model
        INTEGER, parameter :: ng1  = 10
        INTEGER, parameter :: ng2  = 12
        INTEGER, parameter :: ng3  = 16
        INTEGER, parameter :: ng4  = 14
        INTEGER, parameter :: ng5  = 16
        INTEGER, parameter :: ng6  = 8
        INTEGER, parameter :: ng7  = 12
        INTEGER, parameter :: ng8  = 8
        INTEGER, parameter :: ng9  = 12
        INTEGER, parameter :: ng10 = 6
        INTEGER, parameter :: ng11 = 8
        INTEGER, parameter :: ng12 = 8
        INTEGER, parameter :: ng13 = 4
        INTEGER, parameter :: ng14 = 2
        INTEGER, parameter :: ng15 = 2
        INTEGER, parameter :: ng16 = 2
        INTEGER, parameter :: ngs1  = 10
        INTEGER, parameter :: ngs2  = 22
        INTEGER, parameter :: ngs3  = 38
        INTEGER, parameter :: ngs4  = 52
        INTEGER, parameter :: ngs5  = 68
        INTEGER, parameter :: ngs6  = 76
        INTEGER, parameter :: ngs7  = 88
        INTEGER, parameter :: ngs8  = 96
        INTEGER, parameter :: ngs9  = 108
        INTEGER, parameter :: ngs10 = 114
        INTEGER, parameter :: ngs11 = 122
        INTEGER, parameter :: ngs12 = 130
        INTEGER, parameter :: ngs13 = 134
        INTEGER, parameter :: ngs14 = 136
        INTEGER, parameter :: ngs15 = 138
        ! Use for 256 g-point model
        !      integer, parameter :: ng1  = 16
        !      integer, parameter :: ng2  = 16
        !      integer, parameter :: ng3  = 16
        !      integer, parameter :: ng4  = 16
        !      integer, parameter :: ng5  = 16
        !      integer, parameter :: ng6  = 16
        !      integer, parameter :: ng7  = 16
        !      integer, parameter :: ng8  = 16
        !      integer, parameter :: ng9  = 16
        !      integer, parameter :: ng10 = 16
        !      integer, parameter :: ng11 = 16
        !      integer, parameter :: ng12 = 16
        !      integer, parameter :: ng13 = 16
        !      integer, parameter :: ng14 = 16
        !      integer, parameter :: ng15 = 16
        !      integer, parameter :: ng16 = 16
        !      integer, parameter :: ngs1  = 16
        !      integer, parameter :: ngs2  = 32
        !      integer, parameter :: ngs3  = 48
        !      integer, parameter :: ngs4  = 64
        !      integer, parameter :: ngs5  = 80
        !      integer, parameter :: ngs6  = 96
        !      integer, parameter :: ngs7  = 112
        !      integer, parameter :: ngs8  = 128
        !      integer, parameter :: ngs9  = 144
        !      integer, parameter :: ngs10 = 160
        !      integer, parameter :: ngs11 = 176
        !      integer, parameter :: ngs12 = 192
        !      integer, parameter :: ngs13 = 208
        !      integer, parameter :: ngs14 = 224
        !      integer, parameter :: ngs15 = 240
        !      integer, parameter :: ngs16 = 256

    ! write subroutines
    ! No subroutines
    ! No module extern variables
    END MODULE parrrtm
