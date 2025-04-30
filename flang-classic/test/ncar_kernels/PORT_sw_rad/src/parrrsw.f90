
! KGEN-generated Fortran source file
!
! Filename    : parrrsw.f90
! Generated at: 2015-07-07 00:48:23
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
        INTEGER, parameter :: nbndsw = 14 !jpsw, ksw
        !jpaer
        INTEGER, parameter :: mxmol  = 38
        INTEGER, parameter :: nmol   = 7
        ! Use for 112 g-point model
        INTEGER, parameter :: ngptsw = 112 !jpgpt
        ! Use for 224 g-point model
        !      integer, parameter :: ngptsw = 224   !jpgpt
        ! may need to rename these - from v2.6
        INTEGER, parameter :: jpband   = 29
        INTEGER, parameter :: jpb1     = 16 !istart
        INTEGER, parameter :: jpb2     = 29 !iend
        ! ^
        ! Use for 112 g-point model
        INTEGER, parameter :: ng16 = 6
        INTEGER, parameter :: ng17 = 12
        INTEGER, parameter :: ng18 = 8
        INTEGER, parameter :: ng19 = 8
        INTEGER, parameter :: ng20 = 10
        INTEGER, parameter :: ng21 = 10
        INTEGER, parameter :: ng22 = 2
        INTEGER, parameter :: ng23 = 10
        INTEGER, parameter :: ng24 = 8
        INTEGER, parameter :: ng25 = 6
        INTEGER, parameter :: ng26 = 6
        INTEGER, parameter :: ng27 = 8
        INTEGER, parameter :: ng28 = 6
        INTEGER, parameter :: ng29 = 12
        INTEGER, parameter :: ngs16 = 6
        INTEGER, parameter :: ngs17 = 18
        INTEGER, parameter :: ngs18 = 26
        INTEGER, parameter :: ngs19 = 34
        INTEGER, parameter :: ngs20 = 44
        INTEGER, parameter :: ngs21 = 54
        INTEGER, parameter :: ngs22 = 56
        INTEGER, parameter :: ngs23 = 66
        INTEGER, parameter :: ngs24 = 74
        INTEGER, parameter :: ngs25 = 80
        INTEGER, parameter :: ngs26 = 86
        INTEGER, parameter :: ngs27 = 94
        INTEGER, parameter :: ngs28 = 100
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
