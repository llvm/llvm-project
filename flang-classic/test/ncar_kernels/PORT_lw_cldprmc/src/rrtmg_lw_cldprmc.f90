
! KGEN-generated Fortran source file
!
! Filename    : rrtmg_lw_cldprmc.f90
! Generated at: 2015-07-26 20:16:59
! KGEN version: 0.4.13



    MODULE rrtmg_lw_cldprmc
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        !  --------------------------------------------------------------------------
        ! |                                                                          |
        ! |  Copyright 2002-2007, Atmospheric & Environmental Research, Inc. (AER).  |
        ! |  This software may be used, copied, or redistributed as long as it is    |
        ! |  not sold and this copyright notice is reproduced on each copy made.     |
        ! |  This model is provided as is without any express or implied warranties. |
        ! |                       (http://www.rtweb.aer.com/)                        |
        ! |                                                                          |
        !  --------------------------------------------------------------------------
        ! --------- Modules ----------
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !      use parkind, only : jpim, jprb
        USE parrrtm, ONLY: ngptlw
        USE rrlw_cld, ONLY: absice0
        USE rrlw_cld, ONLY: absice1
        USE rrlw_cld, ONLY: absice2
        USE rrlw_cld, ONLY: absice3
        USE rrlw_cld, ONLY: absliq0
        USE rrlw_cld, ONLY: absliq1
        USE rrlw_wvn, ONLY: ngb
        USE rrlw_vsn, ONLY: hvrclc
        IMPLICIT NONE
        CONTAINS

        ! write subroutines
        ! No subroutines
        ! No module extern variables
        ! ------------------------------------------------------------------------------

        SUBROUTINE cldprmc(ncol, nlayers, inflag, iceflag, liqflag, cldfmc, ciwpmc, clwpmc, reicmc, dgesmc, relqmc, ncbands, &
        taucmc)
            ! ------------------------------------------------------------------------------
            ! Purpose:  Compute the cloud optical depth(s) for each cloudy layer.
            ! ------- Input -------
            INTEGER, intent(in) :: ncol ! total number of columns
            INTEGER, intent(in) :: nlayers ! total number of layers
            INTEGER, intent(in) :: inflag ! see definitions
            INTEGER, intent(in) :: iceflag ! see definitions
            INTEGER, intent(in) :: liqflag ! see definitions
            !    Dimensions: (ncol)
            REAL(KIND=r8), intent(in) :: cldfmc(:,:,:) ! cloud fraction [mcica]
            !    Dimensions: (ncol,ngptlw,nlayers)
            REAL(KIND=r8), intent(in) :: ciwpmc(:,:,:) ! cloud ice water path [mcica]
            !    Dimensions: (ncol,ngptlw,nlayers)
            REAL(KIND=r8), intent(in) :: clwpmc(:,:,:) ! cloud liquid water path [mcica]
            !    Dimensions: (ncol,ngptlw,nlayers)
            REAL(KIND=r8), intent(in) :: relqmc(:,:) ! liquid particle effective radius (microns)
            !    Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(in) :: reicmc(:,:) ! ice particle effective radius (microns)
            !    Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(in) :: dgesmc(:,:) ! ice particle generalized effective size (microns)
            !    Dimensions: (ncol,nlayers)
            ! ------- Output -------
            INTEGER, intent(out) :: ncbands(:) ! number of cloud spectral bands
            !    Dimensions: (ncol)
            REAL(KIND=r8), intent(inout) :: taucmc(:,:,:) ! cloud optical depth [mcica]
            !    Dimensions: (ncol,ngptlw,nlayers)
            ! ------- Local -------
            INTEGER :: lay ! Layer index
            INTEGER :: ib ! spectral band index
            INTEGER :: ig ! g-point interval index
            REAL(KIND=r8) :: abscoice(ngptlw) ! ice absorption coefficients
            REAL(KIND=r8) :: abscoliq(ngptlw) ! liquid absorption coefficients
            REAL(KIND=r8) :: cwp ! cloud water path
            REAL(KIND=r8) :: radice ! cloud ice effective radius (microns)
            REAL(KIND=r8) :: dgeice ! cloud ice generalized effective size
            REAL(KIND=r8) :: factor !
            REAL(KIND=r8) :: fint !
            REAL(KIND=r8) :: radliq ! cloud liquid droplet radius (microns)
            ! epsilon
            REAL(KIND=r8), parameter :: cldmin = 1.e-80_r8 ! minimum value for cloud quantities
            ! ------- Definitions -------
            !     Explanation of the method for each value of INFLAG.  Values of
            !     0 or 1 for INFLAG do not distingish being liquid and ice clouds.
            !     INFLAG = 2 does distinguish between liquid and ice clouds, and
            !     requires further user input to specify the method to be used to
            !     compute the aborption due to each.
            !     INFLAG = 0:  For each cloudy layer, the cloud fraction and (gray)
            !                  optical depth are input.
            !     INFLAG = 1:  For each cloudy layer, the cloud fraction and cloud
            !                  water path (g/m2) are input.  The (gray) cloud optical
            !                  depth is computed as in CAM3.
            !     INFLAG = 2:  For each cloudy layer, the cloud fraction, cloud
            !                  water path (g/m2), and cloud ice fraction are input.
            !       ICEFLAG = 0:  The ice effective radius (microns) is input and the
            !                     optical depths due to ice clouds are computed as in CAM3.
            !       ICEFLAG = 1:  The ice effective radius (microns) is input and the
            !                     optical depths due to ice clouds are computed as in
            !                     Ebert and Curry, JGR, 97, 3831-3836 (1992).  The
            !                     spectral regions in this work have been matched with
            !                     the spectral bands in RRTM to as great an extent
            !                     as possible:
            !                     E&C 1      IB = 5      RRTM bands 9-16
            !                     E&C 2      IB = 4      RRTM bands 6-8
            !                     E&C 3      IB = 3      RRTM bands 3-5
            !                     E&C 4      IB = 2      RRTM band 2
            !                     E&C 5      IB = 1      RRTM band 1
            !       ICEFLAG = 2:  The ice effective radius (microns) is input and the
            !                     optical properties due to ice clouds are computed from
            !                     the optical properties stored in the RT code,
            !                     STREAMER v3.0 (Reference: Key. J., Streamer
            !                     User's Guide, Cooperative Institute for
            !                     Meteorological Satellite Studies, 2001, 96 pp.).
            !                     Valid range of values for re are between 5.0 and
            !                     131.0 micron.
            !       ICEFLAG = 3: The ice generalized effective size (dge) is input
            !                    and the optical properties, are calculated as in
            !                    Q. Fu, J. Climate, (1998). Q. Fu provided high resolution
            !                    tables which were appropriately averaged for the
            !                    bands in RRTM_LW.  Linear interpolation is used to
            !                    get the coefficients from the stored tables.
            !                    Valid range of values for dge are between 5.0 and
            !                    140.0 micron.
            !       LIQFLAG = 0:  The optical depths due to water clouds are computed as
            !                     in CAM3.
            !       LIQFLAG = 1:  The water droplet effective radius (microns) is input
            !                     and the optical depths due to water clouds are computed
            !                     as in Hu and Stamnes, J., Clim., 6, 728-742, (1993).
            !                     The values for absorption coefficients appropriate for
            !                     the spectral bands in RRTM have been obtained for a
            !                     range of effective radii by an averaging procedure
            !                     based on the work of J. Pinto (private communication).
            !                     Linear interpolation is used to get the absorption
            !                     coefficients for the input effective radius.
            INTEGER :: iplon,index
      hvrclc = '$Revision: 1.5 $'
      ncbands = 1
            ! This initialization is done in rrtmg_lw_subcol.F90.
            !      do lay = 1, nlayers
            !         do ig = 1, ngptlw
            !            taucmc(ig,lay) = 0.0_r8
            !         enddo
            !      enddo
            ! Main layer loop
      do iplon=1,ncol
       do lay = 1, nlayers
        do ig = 1, ngptlw
          cwp = ciwpmc(iplon,ig,lay) + clwpmc(iplon,ig,lay)
          if (cldfmc(iplon,ig,lay) .ge. cldmin .and. &
             (cwp .ge. cldmin .or. taucmc(iplon,ig,lay) .ge. cldmin)) then
                            ! Ice clouds and water clouds combined.
            if (inflag .eq. 0) then
                                ! Cloud optical depth already defined in taucmc, return to main program
               return
            elseif(inflag .eq. 1) then 
                stop 'INFLAG = 1 OPTION NOT AVAILABLE WITH MCICA'
                                !               cwp = ciwpmc(ig,lay) + clwpmc(ig,lay)
                                !               taucmc(ig,lay) = abscld1 * cwp
                                ! Separate treatement of ice clouds and water clouds.
            elseif(inflag .eq. 2) then
               radice = reicmc(iplon,lay)
                                ! Calculation of absorption coefficients due to ice clouds.
               if (ciwpmc(iplon,ig,lay) .eq. 0.0_r8) then
                  abscoice(ig) = 0.0_r8
               elseif (iceflag .eq. 0) then
                  if (radice .lt. 10.0_r8) stop 'ICE RADIUS TOO SMALL'
                  abscoice(ig) = absice0(1) + absice0(2)/radice
               elseif (iceflag .eq. 1) then
                                    ! mji - turn off limits to mimic CAM3
                                    !                  if (radice .lt. 13.0_r8 .or. radice .gt. 130._r8) stop &
                                    !                      'ICE RADIUS OUT OF BOUNDS'
                  ncbands(iplon) = 5
                  ib = ngb(ig)
                  abscoice(ig) = absice1(1,ib) + absice1(2,ib)/radice
                                    ! For iceflag=2 option, combine with iceflag=0 option to handle out of bounds
                                    ! particle sizes.
                                    ! Use iceflag=2 option for ice particle effective radii from 5.0 and 131.0 microns
                                    ! and use iceflag=0 option for ice particles greater than 131.0 microns.
                                    ! *** NOTE: Transition between two methods has not been smoothed.
               elseif (iceflag .eq. 2) then
                  if (radice .lt. 5.0_r8) stop 'ICE RADIUS OUT OF BOUNDS'
                  if (radice .ge. 5.0_r8 .and. radice .le. 131._r8) then
                     ncbands(iplon) = 16
                     factor = (radice - 2._r8)/3._r8
                     index = int(factor)
                     if (index .eq. 43) index = 42
                     fint = factor - float(index)
                     ib = ngb(ig)
                     abscoice(ig) = &
                         absice2(index,ib) + fint * &
                         (absice2(index+1,ib) - (absice2(index,ib))) 
                  elseif (radice .gt. 131._r8) then
                     abscoice(ig) = absice0(1) + absice0(2)/radice
                  endif
                                    ! For iceflag=3 option, combine with iceflag=0 option to handle large particle sizes.
                                    ! Use iceflag=3 option for ice particle effective radii from 3.2 and 91.0 microns
                                    ! (generalized effective size, dge, from 5 to 140 microns), and use iceflag=0 option
                                    ! for ice particle effective radii greater than 91.0 microns (dge = 140 microns).
                                    ! *** NOTE: Fu parameterization requires particle size in generalized effective size.
                                    ! *** NOTE: Transition between two methods has not been smoothed.
               elseif (iceflag .eq. 3) then
                  dgeice = dgesmc(iplon,lay)
                  if (dgeice .lt. 5.0_r8) stop 'ICE GENERALIZED EFFECTIVE SIZE OUT OF BOUNDS'
                  if (dgeice .ge. 5.0_r8 .and. dgeice .le. 140._r8) then
                     ncbands(iplon) = 16
                     factor = (dgeice - 2._r8)/3._r8
                     index = int(factor)
                     if (index .eq. 46) index = 45
                     fint = factor - float(index)
                     ib = ngb(ig)
                     abscoice(ig) = &
                         absice3(index,ib) + fint * &
                         (absice3(index+1,ib) - (absice3(index,ib)))
                  elseif (dgeice .gt. 140._r8) then
                     abscoice(ig) = absice0(1) + absice0(2)/radice
                  endif
               endif
                                ! Calculation of absorption coefficients due to water clouds.
               if (clwpmc(iplon,ig,lay) .eq. 0.0_r8) then
                  abscoliq(ig) = 0.0_r8
               elseif (liqflag .eq. 0) then
                   abscoliq(ig) = absliq0
               elseif (liqflag .eq. 1) then
                  radliq = relqmc(iplon,lay)
                  if (radliq .lt. 1.5_r8 .or. radliq .gt. 60._r8) stop &
                       'LIQUID EFFECTIVE RADIUS OUT OF BOUNDS'
                  index = radliq - 1.5_r8
                  if (index .eq. 58) index = 57
                  if (index .eq. 0) index = 1
                  fint = radliq - 1.5_r8 - index
                  ib = ngb(ig)
                  abscoliq(ig) = &
                        absliq1(index,ib) + fint * &
                        (absliq1(index+1,ib) - (absliq1(index,ib)))
               endif
               taucmc(iplon,ig,lay) = ciwpmc(iplon,ig,lay) * abscoice(ig) + &
                                clwpmc(iplon,ig,lay) * abscoliq(ig)
            endif
         endif
         enddo
      enddo
    enddo
        END SUBROUTINE cldprmc
    END MODULE rrtmg_lw_cldprmc
