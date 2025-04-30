
! KGEN-generated Fortran source file
!
! Filename    : rrtmg_lw_rtrnmc.f90
! Generated at: 2015-07-26 20:37:04
! KGEN version: 0.4.13



    MODULE rrtmg_lw_rtrnmc
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
        USE parrrtm, ONLY: nbndlw
        USE rrlw_con, ONLY: fluxfac
        USE rrlw_con, ONLY: heatfac
        USE rrlw_wvn, ONLY: ngb
        USE rrlw_wvn, ONLY: ngs
        USE rrlw_wvn, ONLY: delwave
        USE rrlw_tbl, ONLY: bpade
        USE rrlw_tbl, ONLY: tblint
        USE rrlw_tbl, ONLY: tfn_tbl
        USE rrlw_tbl, ONLY: exp_tbl
        USE rrlw_tbl, ONLY: tau_tbl
        USE rrlw_vsn, ONLY: hvrrtc
        IMPLICIT NONE
        PUBLIC rtrnmc
        CONTAINS

        ! write subroutines
        ! No subroutines
        ! No module extern variables
        !-----------------------------------------------------------------------------

        SUBROUTINE rtrnmc(ncol, nlayers, istart, iend, iout, pz, semiss, ncbands, cldfmc, taucmc, planklay, planklev, plankbnd, &
        pwvcm, fracs, taut, totuflux, totdflux, fnet, htr, totuclfl, totdclfl, fnetc, htrc, totufluxs, totdfluxs)
            !-----------------------------------------------------------------------------
            !
            !  Original version:   E. J. Mlawer, et al. RRTM_V3.0
            !  Revision for GCMs:  Michael J. Iacono; October, 2002
            !  Revision for F90:  Michael J. Iacono; June, 2006
            !
            !  This program calculates the upward fluxes, downward fluxes, and
            !  heating rates for an arbitrary clear or cloudy atmosphere.  The input
            !  to this program is the atmospheric profile, all Planck function
            !  information, and the cloud fraction by layer.  A variable diffusivity
            !  angle (SECDIFF) is used for the angle integration.  Bands 2-3 and 5-9
            !  use a value for SECDIFF that varies from 1.50 to 1.80 as a function of
            !  the column water vapor, and other bands use a value of 1.66.  The Gaussian
            !  weight appropriate to this angle (WTDIFF=0.5) is applied here.  Note that
            !  use of the emissivity angle for the flux integration can cause errors of
            !  1 to 4 W/m2 within cloudy layers.
            !  Clouds are treated with the McICA stochastic approach and maximum-random
            !  cloud overlap.
            !***************************************************************************
            ! ------- Declarations -------
            ! ----- Input -----
            INTEGER, intent(in) :: ncol ! total number of columns
            INTEGER, intent(in) :: nlayers ! total number of layers
            INTEGER, intent(in) :: istart ! beginning band of calculation
            INTEGER, intent(in) :: iend ! ending band of calculation
            INTEGER, intent(in) :: iout ! output option flag
            ! Atmosphere
            REAL(KIND=r8), intent(in) :: pz(:,0:) ! level (interface) pressures (hPa, mb)
            !    Dimensions: (ncol,0:nlayers)
            REAL(KIND=r8), intent(in) :: pwvcm(:) ! precipitable water vapor (cm)
            !    Dimensions: (ncol)
            REAL(KIND=r8), intent(in) :: semiss(:,:) ! lw surface emissivity
            !    Dimensions: (ncol,nbndlw)
            REAL(KIND=r8), intent(in) :: planklay(:,:,:) !
            !    Dimensions: (ncol,nlayers,nbndlw)
            REAL(KIND=r8), intent(in) :: planklev(:,0:,:) !
            !    Dimensions: (ncol,0:nlayers,nbndlw)
            REAL(KIND=r8), intent(in) :: plankbnd(:,:) !
            !    Dimensions: (ncol,nbndlw)
            REAL(KIND=r8), intent(in) :: fracs(:,:,:) !
            !    Dimensions: (ncol,nlayers,ngptw)
            REAL(KIND=r8), intent(in) :: taut(:,:,:) ! gaseous + aerosol optical depths
            !    Dimensions: (ncol,nlayers,ngptlw)
            ! Clouds
            INTEGER, intent(in) :: ncbands(:) ! number of cloud spectral bands
            !    Dimensions: (ncol)
            REAL(KIND=r8), intent(in) :: cldfmc(:,:,:) ! layer cloud fraction [mcica]
            !    Dimensions: (ncol,ngptlw,nlayers)
            REAL(KIND=r8), intent(in) :: taucmc(:,:,:) ! layer cloud optical depth [mcica]
            !    Dimensions: (ncol,ngptlw,nlayers)
            ! ----- Output -----
            REAL(KIND=r8), intent(out) :: totuflux(:,0:) ! upward longwave flux (w/m2)
            !    Dimensions: (ncol,0:nlayers)
            REAL(KIND=r8), intent(out) :: totdflux(:,0:) ! downward longwave flux (w/m2)
            !    Dimensions: (ncol,0:nlayers)
            REAL(KIND=r8), intent(out) :: fnet(:,0:) ! net longwave flux (w/m2)
            !    Dimensions: (ncol,0:nlayers)
            REAL(KIND=r8), intent(out) :: htr(:,0:) ! longwave heating rate (k/day)
            !    Dimensions: (ncol,0:nlayers)
            REAL(KIND=r8), intent(out) :: totuclfl(:,0:) ! clear sky upward longwave flux (w/m2)
            !    Dimensions: (ncol,0:nlayers)
            REAL(KIND=r8), intent(out) :: totdclfl(:,0:) ! clear sky downward longwave flux (w/m2)
            !    Dimensions: (ncol,0:nlayers)
            REAL(KIND=r8), intent(out) :: fnetc(:,0:) ! clear sky net longwave flux (w/m2)
            !    Dimensions: (ncol,0:nlayers)
            REAL(KIND=r8), intent(out) :: htrc(:,0:) ! clear sky longwave heating rate (k/day)
            !    Dimensions: (ncol,0:nlayers)
            REAL(KIND=r8), intent(out) :: totufluxs(:,:,0:) ! upward longwave flux spectral (w/m2)
            !    Dimensions: (ncol,nbndlw, 0:nlayers)
            REAL(KIND=r8), intent(out) :: totdfluxs(:,:,0:) ! downward longwave flux spectral (w/m2)
            !    Dimensions: (ncol,nbndlw, 0:nlayers)
            ! ----- Local -----
            ! Declarations for radiative transfer
            REAL(KIND=r8) :: abscld(nlayers,ngptlw)
            REAL(KIND=r8) :: atot(nlayers)
            REAL(KIND=r8) :: atrans(nlayers)
            REAL(KIND=r8) :: bbugas(nlayers)
            REAL(KIND=r8) :: bbutot(nlayers)
            REAL(KIND=r8) :: clrurad(0:nlayers)
            REAL(KIND=r8) :: clrdrad(0:nlayers)
            REAL(KIND=r8) :: efclfrac(nlayers,ngptlw)
            REAL(KIND=r8) :: uflux(0:nlayers)
            REAL(KIND=r8) :: dflux(0:nlayers)
            REAL(KIND=r8) :: urad(0:nlayers)
            REAL(KIND=r8) :: drad(0:nlayers)
            REAL(KIND=r8) :: uclfl(0:nlayers)
            REAL(KIND=r8) :: dclfl(0:nlayers)
            REAL(KIND=r8) :: odcld(nlayers,ngptlw)
            REAL(KIND=r8) :: secdiff(nbndlw) ! secant of diffusivity angle
            REAL(KIND=r8) :: a0(nbndlw)
            REAL(KIND=r8) :: a1(nbndlw)
            REAL(KIND=r8) :: a2(nbndlw) ! diffusivity angle adjustment coefficients
            REAL(KIND=r8) :: wtdiff
            REAL(KIND=r8) :: rec_6
            REAL(KIND=r8) :: transcld
            REAL(KIND=r8) :: radld
            REAL(KIND=r8) :: radclrd
            REAL(KIND=r8) :: plfrac
            REAL(KIND=r8) :: blay
            REAL(KIND=r8) :: dplankup
            REAL(KIND=r8) :: dplankdn
            REAL(KIND=r8) :: odepth
            REAL(KIND=r8) :: odtot
            REAL(KIND=r8) :: odepth_rec
            REAL(KIND=r8) :: gassrc
            REAL(KIND=r8) :: odtot_rec
            REAL(KIND=r8) :: bbdtot
            REAL(KIND=r8) :: bbd
            REAL(KIND=r8) :: tblind
            REAL(KIND=r8) :: tfactot
            REAL(KIND=r8) :: tfacgas
            REAL(KIND=r8) :: transc
            REAL(KIND=r8) :: tausfac
            REAL(KIND=r8) :: rad0
            REAL(KIND=r8) :: reflect
            REAL(KIND=r8) :: radlu
            REAL(KIND=r8) :: radclru
            INTEGER :: icldlyr(nlayers) ! flag for cloud in layer
            INTEGER :: ibnd
            INTEGER :: lay
            INTEGER :: ig
            INTEGER :: ib
            INTEGER :: iband
            INTEGER :: lev
            INTEGER :: l ! loop indices
            INTEGER :: igc ! g-point interval counter
            INTEGER :: iclddn ! flag for cloud in down path
            INTEGER :: ittot
            INTEGER :: itgas
            INTEGER :: itr ! lookup table indices
            ! ------- Definitions -------
            ! input
            !    nlayers                      ! number of model layers
            !    ngptlw                       ! total number of g-point subintervals
            !    nbndlw                       ! number of longwave spectral bands
            !    ncbands                      ! number of spectral bands for clouds
            !    secdiff                      ! diffusivity angle
            !    wtdiff                       ! weight for radiance to flux conversion
            !    pavel                        ! layer pressures (mb)
            !    pz                           ! level (interface) pressures (mb)
            !    tavel                        ! layer temperatures (k)
            !    tz                           ! level (interface) temperatures(mb)
            !    tbound                       ! surface temperature (k)
            !    cldfrac                      ! layer cloud fraction
            !    taucloud                     ! layer cloud optical depth
            !    itr                          ! integer look-up table index
            !    icldlyr                      ! flag for cloudy layers
            !    iclddn                       ! flag for cloud in column at any layer
            !    semiss                       ! surface emissivities for each band
            !    reflect                      ! surface reflectance
            !    bpade                        ! 1/(pade constant)
            !    tau_tbl                      ! clear sky optical depth look-up table
            !    exp_tbl                      ! exponential look-up table for transmittance
            !    tfn_tbl                      ! tau transition function look-up table
            ! local
            !    atrans                       ! gaseous absorptivity
            !    abscld                       ! cloud absorptivity
            !    atot                         ! combined gaseous and cloud absorptivity
            !    odclr                        ! clear sky (gaseous) optical depth
            !    odcld                        ! cloud optical depth
            !    odtot                        ! optical depth of gas and cloud
            !    tfacgas                      ! gas-only pade factor, used for planck fn
            !    tfactot                      ! gas and cloud pade factor, used for planck fn
            !    bbdgas                       ! gas-only planck function for downward rt
            !    bbugas                       ! gas-only planck function for upward rt
            !    bbdtot                       ! gas and cloud planck function for downward rt
            !    bbutot                       ! gas and cloud planck function for upward calc.
            !    gassrc                       ! source radiance due to gas only
            !    efclfrac                     ! effective cloud fraction
            !    radlu                        ! spectrally summed upward radiance
            !    radclru                      ! spectrally summed clear sky upward radiance
            !    urad                         ! upward radiance by layer
            !    clrurad                      ! clear sky upward radiance by layer
            !    radld                        ! spectrally summed downward radiance
            !    radclrd                      ! spectrally summed clear sky downward radiance
            !    drad                         ! downward radiance by layer
            !    clrdrad                      ! clear sky downward radiance by layer
            ! output
            !    totuflux                     ! upward longwave flux (w/m2)
            !    totdflux                     ! downward longwave flux (w/m2)
            !    fnet                         ! net longwave flux (w/m2)
            !    htr                          ! longwave heating rate (k/day)
            !    totuclfl                     ! clear sky upward longwave flux (w/m2)
            !    totdclfl                     ! clear sky downward longwave flux (w/m2)
            !    fnetc                        ! clear sky net longwave flux (w/m2)
            !    htrc                         ! clear sky longwave heating rate (k/day)
            ! This secant and weight corresponds to the standard diffusivity
            ! angle.  This initial value is redefined below for some bands.
      data wtdiff /0.5_r8/
      data rec_6 /0.166667_r8/
            ! Reset diffusivity angle for Bands 2-3 and 5-9 to vary (between 1.50
            ! and 1.80) as a function of total column water vapor.  The function
            ! has been defined to minimize flux and cooling rate errors in these bands
            ! over a wide range of precipitable water values.
      data a0 / 1.66_r8,  1.55_r8,  1.58_r8,  1.66_r8, &
                1.54_r8, 1.454_r8,  1.89_r8,  1.33_r8, &
               1.668_r8,  1.66_r8,  1.66_r8,  1.66_r8, &
                1.66_r8,  1.66_r8,  1.66_r8,  1.66_r8 /
      data a1 / 0.00_r8,  0.25_r8,  0.22_r8,  0.00_r8, &
                0.13_r8, 0.446_r8, -0.10_r8,  0.40_r8, &
              -0.006_r8,  0.00_r8,  0.00_r8,  0.00_r8, &
                0.00_r8,  0.00_r8,  0.00_r8,  0.00_r8 /
      data a2 / 0.00_r8, -12.0_r8, -11.7_r8,  0.00_r8, &
               -0.72_r8,-0.243_r8,  0.19_r8,-0.062_r8, &
               0.414_r8,  0.00_r8,  0.00_r8,  0.00_r8, &
                0.00_r8,  0.00_r8,  0.00_r8,  0.00_r8 /
            INTEGER :: iplon
      hvrrtc = '$Revision: 1.3 $'
  do iplon=1,ncol
      do ibnd = 1,nbndlw
         if (ibnd.eq.1 .or. ibnd.eq.4 .or. ibnd.ge.10) then
           secdiff(ibnd) = 1.66_r8
         else
           secdiff(ibnd) = a0(ibnd) + a1(ibnd)*exp(a2(ibnd)*pwvcm(iplon))
         endif
      enddo
      if (pwvcm(iplon).lt.1.0) secdiff(6) = 1.80_r8
      if (pwvcm(iplon).gt.7.1) secdiff(7) = 1.50_r8
      urad(0) = 0.0_r8
      drad(0) = 0.0_r8
      totuflux(iplon,0) = 0.0_r8
      totdflux(iplon,0) = 0.0_r8
      clrurad(0) = 0.0_r8
      clrdrad(0) = 0.0_r8
      totuclfl(iplon,0) = 0.0_r8
      totdclfl(iplon,0) = 0.0_r8
      do lay = 1, nlayers
         urad(lay) = 0.0_r8
         drad(lay) = 0.0_r8
         totuflux(iplon,lay) = 0.0_r8
         totdflux(iplon,lay) = 0.0_r8
         clrurad(lay) = 0.0_r8
         clrdrad(lay) = 0.0_r8
         totuclfl(iplon,lay) = 0.0_r8
         totdclfl(iplon,lay) = 0.0_r8
         icldlyr(lay) = 0
                    ! Change to band loop?
         do ig = 1, ngptlw
            if (cldfmc(iplon,ig,lay) .eq. 1._r8) then
               ib = ngb(ig)
               odcld(lay,ig) = secdiff(ib) * taucmc(iplon,ig,lay)
               transcld = exp(-odcld(lay,ig))
               abscld(lay,ig) = 1._r8 - transcld
               efclfrac(lay,ig) = abscld(lay,ig) * cldfmc(iplon,ig,lay)
               icldlyr(lay) = 1
            else
               odcld(lay,ig) = 0.0_r8
               abscld(lay,ig) = 0.0_r8
               efclfrac(lay,ig) = 0.0_r8
            endif
         enddo
      enddo
      igc = 1
                ! Loop over frequency bands.
      do iband = istart, iend
                    ! Reinitialize g-point counter for each band if output for each band is requested.
         if (iout.gt.0.and.iband.ge.2) igc = ngs(iband-1)+1
                    ! Loop over g-channels.
 1000    continue
                    ! Radiative transfer starts here.
         radld = 0._r8
         radclrd = 0._r8
         iclddn = 0
                    ! Downward radiative transfer loop.
         do lev = nlayers, 1, -1
               plfrac = fracs(iplon,lev,igc)
               blay = planklay(iplon,lev,iband)
               dplankup = planklev(iplon,lev,iband) - blay
               dplankdn = planklev(iplon,lev-1,iband) - blay
               odepth = secdiff(iband) * taut(iplon,lev,igc)
               if (odepth .lt. 0.0_r8) odepth = 0.0_r8
                        !  Cloudy layer
               if (icldlyr(lev).eq.1) then
                  iclddn = 1
                  odtot = odepth + odcld(lev,igc)
                  if (odtot .lt. 0.06_r8) then
                     atrans(lev) = odepth - 0.5_r8*odepth*odepth
                     odepth_rec = rec_6*odepth
                     gassrc = plfrac*(blay+dplankdn*odepth_rec)*atrans(lev)
                     atot(lev) =  odtot - 0.5_r8*odtot*odtot
                     odtot_rec = rec_6*odtot
                     bbdtot =  plfrac * (blay+dplankdn*odtot_rec)
                     bbd = plfrac*(blay+dplankdn*odepth_rec)
                     radld = radld - radld * (atrans(lev) + &
                         efclfrac(lev,igc) * (1. - atrans(lev))) + &
                         gassrc + cldfmc(iplon,igc,lev) * &
                         (bbdtot * atot(lev) - gassrc)
                     drad( lev-1) = drad(lev-1) + radld
                     bbugas(lev) =  plfrac * (blay+dplankup*odepth_rec)
                     bbutot(lev) =  plfrac * (blay+dplankup*odtot_rec)
                  elseif (odepth .le. 0.06_r8) then
                     atrans(lev) = odepth - 0.5_r8*odepth*odepth
                     odepth_rec = rec_6*odepth
                     gassrc = plfrac*(blay+dplankdn*odepth_rec)*atrans(lev)
                     odtot = odepth + odcld(lev,igc)
                     tblind = odtot/(bpade+odtot)
                     ittot = tblint*tblind + 0.5_r8
                     tfactot = tfn_tbl(ittot)
                     bbdtot = plfrac * (blay + tfactot*dplankdn)
                     bbd = plfrac*(blay+dplankdn*odepth_rec)
                     atot(lev) = 1. - exp_tbl(ittot)
                     radld = radld - radld * (atrans(lev) + &
                         efclfrac(lev,igc) * (1._r8 - atrans(lev))) + &
                         gassrc + cldfmc(iplon,igc,lev) * &
                         (bbdtot * atot(lev) - gassrc)
                     drad(lev-1) = drad(lev-1) + radld
                     bbugas(lev) = plfrac * (blay + dplankup*odepth_rec)
                     bbutot(lev) = plfrac * (blay + tfactot * dplankup)
                  else
                     tblind = odepth/(bpade+odepth)
                     itgas = tblint*tblind+0.5_r8
                     odepth = tau_tbl(itgas)
                     atrans(lev) = 1._r8 - exp_tbl(itgas)
                     tfacgas = tfn_tbl(itgas)
                     gassrc = atrans(lev) * plfrac * (blay + tfacgas*dplankdn)
                     odtot = odepth + odcld(lev,igc)
                     tblind = odtot/(bpade+odtot)
                     ittot = tblint*tblind + 0.5_r8
                     tfactot = tfn_tbl(ittot)
                     bbdtot = plfrac * (blay + tfactot*dplankdn)
                     bbd = plfrac*(blay+tfacgas*dplankdn)
                     atot(lev) = 1._r8 - exp_tbl(ittot)
                  radld = radld - radld * (atrans(lev) + &
                    efclfrac(lev,igc) * (1._r8 - atrans(lev))) + &
                    gassrc + cldfmc(iplon,igc,lev) * &
                    (bbdtot * atot(lev) - gassrc)
                  drad(lev-1) = drad(lev-1) + radld
                  bbugas(lev) = plfrac * (blay + tfacgas * dplankup)
                  bbutot(lev) = plfrac * (blay + tfactot * dplankup)
                  endif
                            !  Clear layer
               else
                  if (odepth .le. 0.06_r8) then
                     atrans(lev) = odepth-0.5_r8*odepth*odepth
                     odepth = rec_6*odepth
                     bbd = plfrac*(blay+dplankdn*odepth)
                     bbugas(lev) = plfrac*(blay+dplankup*odepth)
                  else
                     tblind = odepth/(bpade+odepth)
                     itr = tblint*tblind+0.5_r8
                     transc = exp_tbl(itr)
                     atrans(lev) = 1._r8-transc
                     tausfac = tfn_tbl(itr)
                     bbd = plfrac*(blay+tausfac*dplankdn)
                     bbugas(lev) = plfrac * (blay + tausfac * dplankup)
                  endif   
                  radld = radld + (bbd-radld)*atrans(lev)
                  drad(lev-1) = drad(lev-1) + radld
               endif
                        !  Set clear sky stream to total sky stream as long as layers
                        !  remain clear.  Streams diverge when a cloud is reached (iclddn=1),
                        !  and clear sky stream must be computed separately from that point.
                  if (iclddn.eq.1) then
                     radclrd = radclrd + (bbd-radclrd) * atrans(lev) 
                     clrdrad(lev-1) = clrdrad(lev-1) + radclrd
                  else
                     radclrd = radld
                     clrdrad(lev-1) = drad(lev-1)
                  endif
            enddo
                    ! Spectral emissivity & reflectance
                    !  Include the contribution of spectrally varying longwave emissivity
                    !  and reflection from the surface to the upward radiative transfer.
                    !  Note: Spectral and Lambertian reflection are identical for the
                    !  diffusivity angle flux integration used here.
         rad0 = fracs(iplon,1,igc) * plankbnd(iplon,iband)
                    !  Add in specular reflection of surface downward radiance.
         reflect = 1._r8 - semiss(iplon,iband)
         radlu = rad0 + reflect * radld
         radclru = rad0 + reflect * radclrd
                    ! Upward radiative transfer loop.
         urad(0) = urad(0) + radlu
         clrurad(0) = clrurad(0) + radclru
         do lev = 1, nlayers
                        !  Cloudy layer
            if (icldlyr(lev) .eq. 1) then
               gassrc = bbugas(lev) * atrans(lev)
               radlu = radlu - radlu * (atrans(lev) + &
                   efclfrac(lev,igc) * (1._r8 - atrans(lev))) + &
                   gassrc + cldfmc(iplon,igc,lev) * &
                   (bbutot(lev) * atot(lev) - gassrc)
               urad(lev) = urad(lev) + radlu
                            !  Clear layer
            else
               radlu = radlu + (bbugas(lev)-radlu)*atrans(lev)
               urad(lev) = urad(lev) + radlu
            endif
                        !  Set clear sky stream to total sky stream as long as all layers
                        !  are clear (iclddn=0).  Streams must be calculated separately at
                        !  all layers when a cloud is present (ICLDDN=1), because surface
                        !  reflectance is different for each stream.
               if (iclddn.eq.1) then
                  radclru = radclru + (bbugas(lev)-radclru)*atrans(lev) 
                  clrurad(lev) = clrurad(lev) + radclru
               else
                  radclru = radlu
                  clrurad(lev) = urad(lev)
               endif
         enddo
                    ! Increment g-point counter
         igc = igc + 1
                    ! Return to continue radiative transfer for all g-channels in present band
         if (igc .le. ngs(iband)) go to 1000
                    ! Process longwave output from band for total and clear streams.
                    ! Calculate upward, downward, and net flux.
         do lev = nlayers, 0, -1
            uflux(lev) = urad(lev)*wtdiff
            dflux(lev) = drad(lev)*wtdiff
            urad(lev) = 0.0_r8
            drad(lev) = 0.0_r8
            totuflux(iplon,lev) = totuflux(iplon,lev) + uflux(lev) * delwave(iband)
            totdflux(iplon,lev) = totdflux(iplon,lev) + dflux(lev) * delwave(iband)
            uclfl(lev) = clrurad(lev)*wtdiff
            dclfl(lev) = clrdrad(lev)*wtdiff
            clrurad(lev) = 0.0_r8
            clrdrad(lev) = 0.0_r8
            totuclfl(iplon,lev) = totuclfl(iplon,lev) + uclfl(lev) * delwave(iband)
            totdclfl(iplon,lev) = totdclfl(iplon,lev) + dclfl(lev) * delwave(iband)
            totufluxs(iplon,iband,lev) = uflux(lev) * delwave(iband)
            totdfluxs(iplon,iband,lev) = dflux(lev) * delwave(iband)
         enddo
                    ! End spectral band loop
      enddo
    enddo
    do iplon=1,ncol
                ! Calculate fluxes at surface
      totuflux(iplon,0) = totuflux(iplon,0) * fluxfac
      totdflux(iplon,0) = totdflux(iplon,0) * fluxfac
      totufluxs(iplon,:,0) = totufluxs(iplon,:,0) * fluxfac
      totdfluxs(iplon,:,0) = totdfluxs(iplon,:,0) * fluxfac
      fnet(iplon,0) = totuflux(iplon,0) - totdflux(iplon,0)
      totuclfl(iplon,0) = totuclfl(iplon,0) * fluxfac
      totdclfl(iplon,0) = totdclfl(iplon,0) * fluxfac
      fnetc(iplon,0) = totuclfl(iplon,0) - totdclfl(iplon,0)
    enddo
            ! Calculate fluxes at model levels
      do lev = 1, nlayers
      do iplon=1,ncol
         totuflux(iplon,lev) = totuflux(iplon,lev) * fluxfac
         totdflux(iplon,lev) = totdflux(iplon,lev) * fluxfac
         totufluxs(iplon,:,lev) = totufluxs(iplon,:,lev) * fluxfac
         totdfluxs(iplon,:,lev) = totdfluxs(iplon,:,lev) * fluxfac
         fnet(iplon,lev) = totuflux(iplon,lev) - totdflux(iplon,lev)
         totuclfl(iplon,lev) = totuclfl(iplon,lev) * fluxfac
         totdclfl(iplon,lev) = totdclfl(iplon,lev) * fluxfac
         fnetc(iplon,lev) = totuclfl(iplon,lev) - totdclfl(iplon,lev)
         l = lev - 1
                    ! Calculate heating rates at model layers
         htr(iplon,l)=heatfac*(fnet(iplon,l)-fnet(iplon,lev))/(pz(iplon,l)-pz(iplon,lev)) 
         htrc(iplon,l)=heatfac*(fnetc(iplon,l)-fnetc(iplon,lev))/(pz(iplon,l)-pz(iplon,lev)) 
      enddo
      enddo
            ! Set heating rate to zero in top layer
      do iplon=1,ncol
        htr(iplon,nlayers) = 0.0_r8
        htrc(iplon,nlayers) = 0.0_r8
    enddo
        END SUBROUTINE rtrnmc
    END MODULE rrtmg_lw_rtrnmc
