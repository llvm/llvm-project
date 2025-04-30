
! KGEN-generated Fortran source file
!
! Filename    : parrrsw.f90
! Generated at: 2015-07-27 00:38:36
! KGEN version: 0.4.13



    MODULE parrrsw
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        !      use parkind ,only : jpim, jprb
        IMPLICIT NONE
        !------------------------------------------------------------------
        ! rrtmg_sw main parameters
        !
        ! Initial version:  JJMorcrette, ECMWF, jul1998
        ! Revised: MJIacono, AER, jun2006
        !------------------------------------------------------------------
        !  name     type     purpose
        ! -----  :  ----   : ----------------------------------------------
        ! mxlay  :  integer: maximum number of layers
        ! mg     :  integer: number of original g-intervals per spectral band
        ! nbndsw :  integer: number of spectral bands
        ! naerec :  integer: number of aerosols (iaer=6, ecmwf aerosol option)
        ! ngptsw :  integer: total number of reduced g-intervals for rrtmg_lw
        ! ngNN   :  integer: number of reduced g-intervals per spectral band
        ! ngsNN  :  integer: cumulative number of g-intervals per band
        !------------------------------------------------------------------
        ! Settings for single column mode.
        ! For GCM use, set nlon to number of longitudes, and
        ! mxlay to number of model layers
        !jplay, klev
        !jpg
        !jpsw, ksw
        !jpaer
        ! Use for 112 g-point model
        INTEGER, parameter :: ngptsw = 112 !jpgpt
        ! Use for 224 g-point model
        !      integer, parameter :: ngptsw = 224   !jpgpt
        ! may need to rename these - from v2.6
        INTEGER, parameter :: jpb1     = 16 !istart
        INTEGER, parameter :: jpb2     = 29 !iend
        ! ^
        ! Use for 112 g-point model
        ! Use for 224 g-point model
        !      integer, parameter :: ng16 = 16
        !      integer, parameter :: ng17 = 16
        !      integer, parameter :: ng18 = 16
        !      integer, parameter :: ng19 = 16
        !      integer, parameter :: ng20 = 16
        !      integer, parameter :: ng21 = 16
        !      integer, parameter :: ng22 = 16
        !      integer, parameter :: ng23 = 16
        !      integer, parameter :: ng24 = 16
        !      integer, parameter :: ng25 = 16
        !      integer, parameter :: ng26 = 16
        !      integer, parameter :: ng27 = 16
        !      integer, parameter :: ng28 = 16
        !      integer, parameter :: ng29 = 16
        !      integer, parameter :: ngs16 = 16
        !      integer, parameter :: ngs17 = 32
        !      integer, parameter :: ngs18 = 48
        !      integer, parameter :: ngs19 = 64
        !      integer, parameter :: ngs20 = 80
        !      integer, parameter :: ngs21 = 96
        !      integer, parameter :: ngs22 = 112
        !      integer, parameter :: ngs23 = 128
        !      integer, parameter :: ngs24 = 144
        !      integer, parameter :: ngs25 = 160
        !      integer, parameter :: ngs26 = 176
        !      integer, parameter :: ngs27 = 192
        !      integer, parameter :: ngs28 = 208
        !      integer, parameter :: ngs29 = 224
        ! Source function solar constant
        ! W/m2

    ! write subroutines
    ! No subroutines
    ! No module extern variables
    END MODULE parrrsw
