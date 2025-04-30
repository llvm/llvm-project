
! KGEN-generated Fortran source file
!
! Filename    : parrrtm.f90
! Generated at: 2015-07-26 20:37:04
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
        ! Use for 140 g-point model
        INTEGER, parameter :: ngptlw = 140
        ! Use for 256 g-point model
        !      integer, parameter :: ngptlw = 256
        ! Use for 140 g-point model
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
