
! KGEN-generated Fortran source file
!
! Filename    : rrtmg_sw_spcvmc.f90
! Generated at: 2015-07-07 00:48:25
! KGEN version: 0.4.13



    MODULE rrtmg_sw_spcvmc
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
        ! ------- Modules -------
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !      use parkind, only : jpim, jprb
        USE parrrsw, ONLY: ngptsw
        USE rrsw_tbl, ONLY: od_lo
        USE rrsw_tbl, ONLY: bpade
        USE rrsw_tbl, ONLY: tblint
        USE rrsw_tbl, ONLY: exp_tbl
        USE rrsw_wvn, ONLY: ngc
        USE rrsw_wvn, ONLY: ngs
        USE rrtmg_sw_reftra, ONLY: reftra_sw
        USE rrtmg_sw_taumol, ONLY: taumol_sw
        USE rrtmg_sw_vrtqdr, ONLY: vrtqdr_sw
        IMPLICIT NONE
        CONTAINS

        ! write subroutines
        ! No subroutines
        ! No module extern variables
        ! ---------------------------------------------------------------------------

        SUBROUTINE spcvmc_sw(lchnk, ncol, nlayers, istart, iend, icpr, idelm, iout, pavel, tavel, pz, tz, tbound, palbd, palbp, &
        pcldfmc, ptaucmc, pasycmc, pomgcmc, ptaormc, ptaua, pasya, pomga, prmu0, coldry, wkl, adjflux, laytrop, layswtch, laylow, &
        jp, jt, jt1, co2mult, colch4, colco2, colh2o, colmol, coln2o, colo2, colo3, fac00, fac01, fac10, fac11, selffac, selffrac,&
         indself, forfac, forfrac, indfor, pbbfd, pbbfu, pbbcd, pbbcu, puvfd, puvcd, pnifd, pnicd, pnifu, pnicu, pbbfddir, &
        pbbcddir, puvfddir, puvcddir, pnifddir, pnicddir, pbbfsu, pbbfsd)
            ! ---------------------------------------------------------------------------
            !
            ! Purpose: Contains spectral loop to compute the shortwave radiative fluxes,
            !          using the two-stream method of H. Barker and McICA, the Monte-Carlo
            !          Independent Column Approximation, for the representation of
            !          sub-grid cloud variability (i.e. cloud overlap).
            !
            ! Interface:  *spcvmc_sw* is called from *rrtmg_sw.F90* or rrtmg_sw.1col.F90*
            !
            ! Method:
            !    Adapted from two-stream model of H. Barker;
            !    Two-stream model options (selected with kmodts in rrtmg_sw_reftra.F90):
            !        1: Eddington, 2: PIFM, Zdunkowski et al., 3: discret ordinates
            !
            ! Modifications:
            !
            ! Original: H. Barker
            ! Revision: Merge with RRTMG_SW: J.-J.Morcrette, ECMWF, Feb 2003
            ! Revision: Add adjustment for Earth/Sun distance : MJIacono, AER, Oct 2003
            ! Revision: Bug fix for use of PALBP and PALBD: MJIacono, AER, Nov 2003
            ! Revision: Bug fix to apply delta scaling to clear sky: AER, Dec 2004
            ! Revision: Code modified so that delta scaling is not done in cloudy profiles
            !           if routine cldprop is used; delta scaling can be applied by swithcing
            !           code below if cldprop is not used to get cloud properties.
            !           AER, Jan 2005
            ! Revision: Modified to use McICA: MJIacono, AER, Nov 2005
            ! Revision: Uniform formatting for RRTMG: MJIacono, AER, Jul 2006
            ! Revision: Use exponential lookup table for transmittance: MJIacono, AER,
            !           Aug 2007
            !
            ! ------------------------------------------------------------------
            ! ------- Declarations ------
            ! ------- Input -------
            INTEGER, intent(in) :: lchnk
            INTEGER, intent(in) :: nlayers
            INTEGER, intent(in) :: istart
            INTEGER, intent(in) :: iend
            INTEGER, intent(in) :: icpr
            INTEGER, intent(in) :: idelm ! delta-m scaling flag
            ! [0 = direct and diffuse fluxes are unscaled]
            ! [1 = direct and diffuse fluxes are scaled]
            INTEGER, intent(in) :: iout
            INTEGER, intent(in) :: ncol ! column loop index
            INTEGER, intent(in) :: laytrop(ncol)
            INTEGER, intent(in) :: layswtch(ncol)
            INTEGER, intent(in) :: laylow(ncol)
            INTEGER, intent(in) :: indfor(:,:)
            !   Dimensions: (ncol,nlayers)
            INTEGER, intent(in) :: indself(:,:)
            !   Dimensions: (ncol,nlayers)
            INTEGER, intent(in) :: jp(:,:)
            !   Dimensions: (ncol,nlayers)
            INTEGER, intent(in) :: jt(:,:)
            !   Dimensions: (ncol,nlayers)
            INTEGER, intent(in) :: jt1(:,:)
            !   Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(in) :: pavel(:,:) ! layer pressure (hPa, mb)
            !   Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(in) :: tavel(:,:) ! layer temperature (K)
            !   Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(in) :: pz(:,0:) ! level (interface) pressure (hPa, mb)
            !   Dimensions: (ncol,0:nlayers)
            REAL(KIND=r8), intent(in) :: tz(:,0:) ! level temperatures (hPa, mb)
            !   Dimensions: (ncol,0:nlayers)
            REAL(KIND=r8), intent(in) :: tbound(ncol) ! surface temperature (K)
            REAL(KIND=r8), intent(in) :: wkl(:,:,:) ! molecular amounts (mol/cm2)
            !   Dimensions: (ncol,mxmol,nlayers)
            REAL(KIND=r8), intent(in) :: coldry(:,:) ! dry air column density (mol/cm2)
            !   Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(in) :: colmol(:,:)
            !   Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(in) :: adjflux(:,:) ! Earth/Sun distance adjustment
            !   Dimensions: (ncol,jpband)
            REAL(KIND=r8), intent(in) :: palbd(:,:) ! surface albedo (diffuse)
            !   Dimensions: (ncol,nbndsw)
            REAL(KIND=r8), intent(in) :: palbp(:,:) ! surface albedo (direct)
            !   Dimensions: (ncol, nbndsw)
            REAL(KIND=r8), intent(in) :: prmu0(ncol) ! cosine of solar zenith angle
            REAL(KIND=r8), intent(in) :: pcldfmc(:,:,:) ! cloud fraction [mcica]
            !   Dimensions: (ncol,nlayers,ngptsw)
            REAL(KIND=r8), intent(in) :: ptaucmc(:,:,:) ! cloud optical depth [mcica]
            !   Dimensions: (ncol,nlayers,ngptsw)
            REAL(KIND=r8), intent(in) :: pasycmc(:,:,:) ! cloud asymmetry parameter [mcica]
            !   Dimensions: (ncol,nlayers,ngptsw)
            REAL(KIND=r8), intent(in) :: pomgcmc(:,:,:) ! cloud single scattering albedo [mcica]
            !   Dimensions: (ncol,nlayers,ngptsw)
            REAL(KIND=r8), intent(in) :: ptaormc(:,:,:) ! cloud optical depth, non-delta scaled [mcica]
            !   Dimensions: (ncol,nlayers,ngptsw)
            REAL(KIND=r8), intent(in) :: ptaua(:,:,:) ! aerosol optical depth
            !   Dimensions: (ncol,nlayers,nbndsw)
            REAL(KIND=r8), intent(in) :: pasya(:,:,:) ! aerosol asymmetry parameter
            !   Dimensions: (ncol,nlayers,nbndsw)
            REAL(KIND=r8), intent(in) :: pomga(:,:,:) ! aerosol single scattering albedo
            !   Dimensions: (ncol,nlayers,nbndsw)
            REAL(KIND=r8), intent(in) :: colh2o(:,:)
            !   Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(in) :: colco2(:,:)
            !   Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(in) :: colch4(:,:)
            !   Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(in) :: co2mult(:,:)
            !   Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(in) :: colo3(:,:)
            !   Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(in) :: colo2(:,:)
            !   Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(in) :: coln2o(:,:)
            !   Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(in) :: forfac(:,:)
            !   Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(in) :: forfrac(:,:)
            !   Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(in) :: selffac(:,:)
            !   Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(in) :: selffrac(:,:)
            !   Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(in) :: fac00(:,:)
            !   Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(in) :: fac01(:,:)
            !   Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(in) :: fac10(:,:)
            !   Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(in) :: fac11(:,:)
            !   Dimensions: (ncol,nlayers)
            ! ------- Output -------
            !   All Dimensions: (nlayers+1)
            REAL(KIND=r8), intent(out) :: pbbcd(:,:)
            REAL(KIND=r8), intent(out) :: pbbcu(:,:)
            REAL(KIND=r8), intent(out) :: pbbfd(:,:)
            REAL(KIND=r8), intent(out) :: pbbfu(:,:)
            REAL(KIND=r8), intent(out) :: pbbfddir(:,:)
            REAL(KIND=r8), intent(out) :: pbbcddir(:,:)
            REAL(KIND=r8), intent(out) :: puvcd(:,:)
            REAL(KIND=r8), intent(out) :: puvfd(:,:)
            REAL(KIND=r8), intent(out) :: puvcddir(:,:)
            REAL(KIND=r8), intent(out) :: puvfddir(:,:)
            REAL(KIND=r8), intent(out) :: pnicd(:,:)
            REAL(KIND=r8), intent(out) :: pnifd(:,:)
            REAL(KIND=r8), intent(out) :: pnicddir(:,:)
            REAL(KIND=r8), intent(out) :: pnifddir(:,:)
            ! Added for net near-IR flux diagnostic
            REAL(KIND=r8), intent(out) :: pnicu(:,:)
            REAL(KIND=r8), intent(out) :: pnifu(:,:)
            ! Output - inactive                                              !   All Dimensions: (nlayers+1)
            !      real(kind=r8), intent(out) :: puvcu(:)
            !      real(kind=r8), intent(out) :: puvfu(:)
            !      real(kind=r8), intent(out) :: pvscd(:)
            !      real(kind=r8), intent(out) :: pvscu(:)
            !      real(kind=r8), intent(out) :: pvsfd(:)
            !      real(kind=r8), intent(out) :: pvsfu(:)
            REAL(KIND=r8), intent(out) :: pbbfsu(:,:,:) ! shortwave spectral flux up (nswbands,nlayers+1)
            REAL(KIND=r8), intent(out) :: pbbfsd(:,:,:) ! shortwave spectral flux down (nswbands,nlayers+1)
            ! ------- Local -------
            LOGICAL :: lrtchkclr(ncol,nlayers)
            LOGICAL :: lrtchkcld(ncol,nlayers)
            INTEGER :: klev
            INTEGER :: ib1
            INTEGER :: ib2
            INTEGER :: ibm
            INTEGER :: igt
            INTEGER :: ikl
            INTEGER :: iw(ncol)
            INTEGER :: jk
            INTEGER :: jb
            INTEGER :: jg, iplon
            !      integer, parameter :: nuv = ??
            !      integer, parameter :: nvs = ??
            INTEGER :: itind(ncol)
            REAL(KIND=r8) :: ze1(ncol)
            REAL(KIND=r8) :: tblind(ncol)
            REAL(KIND=r8) :: zclear(ncol)
            REAL(KIND=r8) :: zcloud(ncol)
            REAL(KIND=r8) :: zdbt(ncol,nlayers+1)
            REAL(KIND=r8) :: zdbt_nodel(ncol,nlayers+1)
            REAL(KIND=r8) :: zgcc(ncol,nlayers)
            REAL(KIND=r8) :: zgco(ncol,nlayers)
            REAL(KIND=r8) :: zomcc(ncol,nlayers)
            REAL(KIND=r8) :: zomco(ncol,nlayers)
            REAL(KIND=r8) :: zrdndc(ncol,nlayers+1)
            REAL(KIND=r8) :: zrdnd(ncol,nlayers+1)
            REAL(KIND=r8) :: zrefc(ncol,nlayers+1)
            REAL(KIND=r8) :: zrefo(ncol,nlayers+1)
            REAL(KIND=r8) :: zref(	ncol,nlayers+1)
            REAL(KIND=r8) :: zrefdc(ncol,nlayers+1)
            REAL(KIND=r8) :: zrefdo(ncol,nlayers+1)
            REAL(KIND=r8) :: zrefd(ncol,nlayers+1)
            REAL(KIND=r8) :: zrup(ncol,nlayers+1)
            REAL(KIND=r8) :: zrupd(ncol,nlayers+1)
            REAL(KIND=r8) :: zrupc(ncol,nlayers+1)
            REAL(KIND=r8) :: zrupdc(ncol,nlayers+1)
            REAL(KIND=r8) :: ztauc(ncol,nlayers)
            REAL(KIND=r8) :: ztauo(ncol,nlayers)
            REAL(KIND=r8) :: ztdbt(ncol,nlayers+1)
            REAL(KIND=r8) :: ztrac(ncol,nlayers+1)
            REAL(KIND=r8) :: ztrao(ncol,nlayers+1)
            REAL(KIND=r8) :: ztra(ncol,nlayers+1)
            REAL(KIND=r8) :: ztradc(ncol,nlayers+1)
            REAL(KIND=r8) :: ztrado(ncol,nlayers+1)
            REAL(KIND=r8) :: ztrad(ncol,nlayers+1)
            REAL(KIND=r8) :: ztdbtc(ncol,nlayers+1)
            REAL(KIND=r8) :: zdbtc(ncol,nlayers+1)
            REAL(KIND=r8) :: zincflx(ncol,ngptsw)
            REAL(KIND=r8) :: zdbtc_nodel(ncol,nlayers+1)
            REAL(KIND=r8) :: ztdbtc_nodel(ncol,nlayers+1)
            REAL(KIND=r8) :: ztdbt_nodel(ncol,nlayers+1)
            REAL(KIND=r8) :: zdbtmc(ncol)
            REAL(KIND=r8) :: zdbtmo(ncol)
            REAL(KIND=r8) :: zf
            REAL(KIND=r8) :: repclc(ncol)
            REAL(KIND=r8) :: tauorig(ncol)
            REAL(KIND=r8) :: zwf
            !     real(kind=r8) :: zincflux                                   ! inactive
            ! Arrays from rrtmg_sw_taumoln routines
            !      real(kind=r8) :: ztaug(nlayers,16), ztaur(nlayers,16)
            !      real(kind=r8) :: zsflxzen(16)
            REAL(KIND=r8) :: ztaug(ncol,nlayers,ngptsw)
            REAL(KIND=r8) :: ztaur(ncol,nlayers,ngptsw)
            REAL(KIND=r8) :: zsflxzen(ncol,ngptsw)
            ! Arrays from rrtmg_sw_vrtqdr routine
            REAL(KIND=r8) :: zcd(ncol,nlayers+1,ngptsw)
            REAL(KIND=r8) :: zcu(ncol,nlayers+1,ngptsw)
            REAL(KIND=r8) :: zfd(ncol,nlayers+1,ngptsw)
            REAL(KIND=r8) :: zfu(ncol,nlayers+1,ngptsw)
            ! Inactive arrays
            !     real(kind=r8) :: zbbcd(nlayers+1), zbbcu(nlayers+1)
            !     real(kind=r8) :: zbbfd(nlayers+1), zbbfu(nlayers+1)
            !     real(kind=r8) :: zbbfddir(nlayers+1), zbbcddir(nlayers+1)
            ! ------------------------------------------------------------------
            ! Initializations
      ib1 = istart
      ib2 = iend
      klev = nlayers
      !djp repclc(iplon) = 1.e-12_r8
      repclc(:) = 1.e-12_r8
            !      zincflux = 0.0_r8
        do iplon=1,ncol
      do jk=1,klev+1
         pbbcd(iplon,jk)=0._r8
         pbbcu(iplon,jk)=0._r8
         pbbfd(iplon,jk)=0._r8
         pbbfu(iplon,jk)=0._r8
         pbbcddir(iplon,jk)=0._r8
         pbbfddir(iplon,jk)=0._r8
         puvcd(iplon,jk)=0._r8
         puvfd(iplon,jk)=0._r8
         puvcddir(iplon,jk)=0._r8
         puvfddir(iplon,jk)=0._r8
         pnicd(iplon,jk)=0._r8
         pnifd(iplon,jk)=0._r8
         pnicddir(iplon,jk)=0._r8
         pnifddir(iplon,jk)=0._r8
         pnicu(iplon,jk)=0._r8
         pnifu(iplon,jk)=0._r8
      enddo
        end do
        call taumol_sw(ncol,klev, &
                     colh2o, colco2, colch4, colo2, colo3, colmol, &
                     laytrop, jp, jt, jt1, &
                     fac00, fac01, fac10, fac11, &
                     selffac, selffrac, indself, forfac, forfrac,indfor, &
                     zsflxzen, ztaug, ztaur)

      jb = ib1-1                  ! ??? ! ???
			do iplon=1,ncol
					iw(iplon) =0
			end do
         do jb = ib1, ib2
			           ibm = jb-15
         igt = ngc(ibm)
                ! Reinitialize g-point counter for each band if output for each band is requested.
                !        do jk=1,klev+1
                !           zbbcd(jk)=0.0_r8
                !           zbbcu(jk)=0.0_r8
                !           zbbfd(jk)=0.0_r8
                !           zbbfu(jk)=0.0_r8
                !        enddo
                ! Top of g-point interval loop within each band (iw(iplon) is cumulative counter)

					DO IPLON=1,ncol
						if (iout.gt.0.and.ibm.ge.2) iw(iplon)= ngs(ibm-1)
					END do
         do jg = 1,igt
				  do iplon=1,ncol

	            iw(iplon) = iw(iplon)+1
                    ! Apply adjustment for correct Earth/Sun distance and zenith angle to incoming solar flux
            zincflx(iplon,iw(iplon)) = adjflux(iplon,jb) * zsflxzen(iplon,iw(iplon)) * prmu0(iplon)
                    !             zincflux = zincflux + adjflux(jb) * zsflxzen(iw(iplon)) * prmu0           ! inactive
                    ! Compute layer reflectances and transmittances for direct and diffuse sources,
                    ! first clear then cloudy
                    ! zrefc(iplon,jk)  direct albedo for clear
                    ! zrefo(iplon,jk)  direct albedo for cloud
                    ! zrefdc(iplon,jk) diffuse albedo for clear
                    ! zrefdo(iplon,jk) diffuse albedo for cloud
                    ! ztrac(iplon,jk)  direct transmittance for clear
                    ! ztrao(iplon,jk)  direct transmittance for cloudy
                    ! ztradc(iplon,jk) diffuse transmittance for clear
                    ! ztrado(iplon,jk) diffuse transmittance for cloudy
                    !
                    ! zref(iplon,jk)   direct reflectance
                    ! zrefd(iplon,jk)  diffuse reflectance
                    ! ztra(iplon,jk)   direct transmittance
                    ! ztrad(iplon,jk)  diffuse transmittance
                    !
                    ! zdbtc(iplon,jk)  clear direct beam transmittance
                    ! zdbto(jk)  cloudy direct beam transmittance
                    ! zdbt(iplon,jk)   layer mean direct beam transmittance
                    ! ztdbt(iplon,jk)  total direct beam transmittance at levels
                    ! Clear-sky
                    !   TOA direct beam
            ztdbtc(iplon,1)=1.0_r8
            ztdbtc_nodel(iplon,1)=1.0_r8
                    !   Surface values
            zdbtc(iplon,klev+1) =0.0_r8
            ztrac(iplon,klev+1) =0.0_r8
            ztradc(iplon,klev+1)=0.0_r8
            zrefc(iplon,klev+1) =palbp(iplon,ibm)
            zrefdc(iplon,klev+1)=palbd(iplon,ibm)
            zrupc(iplon,klev+1) =palbp(iplon,ibm)
            zrupdc(iplon,klev+1)=palbd(iplon,ibm)
                    ! Cloudy-sky
                    !   Surface values
            ztrao(iplon,klev+1) =0.0_r8
            ztrado(iplon,klev+1)=0.0_r8
            zrefo(iplon,klev+1) =palbp(iplon,ibm)
            zrefdo(iplon,klev+1)=palbd(iplon,ibm)
                    ! Total sky
                    !   TOA direct beam
            ztdbt(iplon,1)=1.0_r8
            ztdbt_nodel(iplon,1)=1.0_r8
                    !   Surface values
            zdbt(iplon,klev+1) =0.0_r8
            ztra(iplon,klev+1) =0.0_r8
            ztrad(iplon,klev+1)=0.0_r8
            zref(iplon,klev+1) =palbp(iplon,ibm)
            zrefd(iplon,klev+1)=palbd(iplon,ibm)
            zrup(iplon,klev+1) =palbp(iplon,ibm)
            zrupd(iplon,klev+1)=palbd(iplon,ibm)
                    ! Top of layer loop
            do jk=1,klev
                        ! Note: two-stream calculations proceed from top to bottom;
                        !   RRTMG_SW quantities are given bottom to top and are reversed here
               ikl=klev+1-jk
                        ! Set logical flag to do REFTRA calculation
                        !   Do REFTRA for all clear layers
               lrtchkclr(iplon,jk)=.true.
                        !   Do REFTRA only for cloudy layers in profile, since already done for clear layers
               lrtchkcld(iplon,jk)=.false.
               lrtchkcld(iplon,jk)=(pcldfmc(iplon,ikl,iw(iplon)) > repclc(iplon))
                        ! Clear-sky optical parameters - this section inactive
                        !   Original
                        !               ztauc(iplon,jk) = ztaur(ikl,iw(iplon)) + ztaug(ikl,iw(iplon))
                        !               zomcc(iplon,jk) = ztaur(ikl,iw(iplon)) / ztauc(iplon,jk)
                        !               zgcc(iplon,jk) = 0.0001_r8
                        !   Total sky optical parameters
                        !               ztauo(iplon,jk) = ztaur(ikl,iw(iplon)) + ztaug(ikl,iw(iplon)) + ptaucmc(ikl,iw(iplon))
                        !               zomco(iplon,jk) = ptaucmc(ikl,iw(iplon)) * pomgcmc(ikl,iw(iplon)) + ztaur(ikl,iw(iplon))
                        !               zgco (jk) = (ptaucmc(ikl,iw(iplon)) * pomgcmc(ikl,iw(iplon)) * pasycmc(ikl,iw(iplon)) + &
                        !                           ztaur(ikl,iw(iplon)) * 0.0001_r8) / zomco(iplon,jk)
                        !               zomco(iplon,jk) = zomco(iplon,jk) / ztauo(iplon,jk)
                        ! Clear-sky optical parameters including aerosols
               ztauc(iplon,jk) = ztaur(iplon,ikl,iw(iplon)) + ztaug(iplon,ikl,iw(iplon)) + ptaua(iplon,ikl,ibm)
               zomcc(iplon,jk) = ztaur(iplon,ikl,iw(iplon)) * 1.0_r8 + ptaua(iplon,ikl,ibm) * pomga(iplon,ikl,ibm)
               zgcc(iplon,jk) = pasya(iplon,ikl,ibm) * pomga(iplon,ikl,ibm) * ptaua(iplon,ikl,ibm) / zomcc(iplon,jk)
               zomcc(iplon,jk) = zomcc(iplon,jk) / ztauc(iplon,jk)
                        ! Pre-delta-scaling clear and cloudy direct beam transmittance (must use 'orig', unscaled cloud OD)
                        !   \/\/\/ This block of code is only needed for unscaled direct beam calculation
               if (idelm .eq. 0) then
                            !
                  zclear(iplon) = 1.0_r8 - pcldfmc(iplon,ikl,iw(iplon))
                  zcloud(iplon) = pcldfmc(iplon,ikl,iw(iplon))
                            ! Clear
                            !                   zdbtmc(iplon) = exp(-ztauc(iplon,jk) / prmu0)
                            ! Use exponential lookup table for transmittance, or expansion of exponential for low tau
                  ze1(iplon) = ztauc(iplon,jk) / prmu0(iplon)
                  if (ze1(iplon) .le. od_lo) then
                     zdbtmc(iplon) = 1._r8 - ze1(iplon) + 0.5_r8 * ze1(iplon) * ze1(iplon)
                  else 
                     tblind(iplon) = ze1(iplon) / (bpade + ze1(iplon))
                     itind(iplon) = tblint * tblind(iplon) + 0.5_r8
                     zdbtmc(iplon) = exp_tbl(itind(iplon))
                  endif
                  zdbtc_nodel(iplon,jk) = zdbtmc(iplon)
                  ztdbtc_nodel(iplon,jk+1) = zdbtc_nodel(iplon,jk) * ztdbtc_nodel(iplon,jk)
                            ! Clear + Cloud
                  tauorig(iplon) = ztauc(iplon,jk) + ptaormc(iplon,ikl,iw(iplon))
                            !                   zdbtmo(iplon) = exp(-tauorig(iplon) / prmu0)
                            ! Use exponential lookup table for transmittance, or expansion of exponential for low tau
                  ze1(iplon) = tauorig(iplon) / prmu0(iplon)
                  if (ze1(iplon) .le. od_lo) then
                     zdbtmo(iplon) = 1._r8 - ze1(iplon) + 0.5_r8 * ze1(iplon) * ze1(iplon)
                  else
                     tblind(iplon) = ze1(iplon) / (bpade + ze1(iplon))
                     itind(iplon) = tblint * tblind(iplon) + 0.5_r8
                     zdbtmo(iplon) = exp_tbl(itind(iplon))
                  endif
                  zdbt_nodel(iplon,jk) = zclear(iplon)*zdbtmc(iplon) + zcloud(iplon)*zdbtmo(iplon)
                  ztdbt_nodel(iplon,jk+1) = zdbt_nodel(iplon,jk) * ztdbt_nodel(iplon,jk)
               endif
                        !   /\/\/\ Above code only needed for unscaled direct beam calculation
                        ! Delta scaling - clear
               zf = zgcc(iplon,jk) * zgcc(iplon,jk)
               zwf = zomcc(iplon,jk) * zf
               ztauc(iplon,jk) = (1.0_r8 - zwf) * ztauc(iplon,jk)
               zomcc(iplon,jk) = (zomcc(iplon,jk) - zwf) / (1.0_r8 - zwf)
               zgcc (iplon,jk) = (zgcc(iplon,jk) - zf) / (1.0_r8 - zf)
                        ! Total sky optical parameters (cloud properties already delta-scaled)
                        !   Use this code if cloud properties are derived in rrtmg_sw_cldprop
               if (icpr .ge. 1) then
                  ztauo(iplon,jk) = ztauc(iplon,jk) + ptaucmc(iplon,ikl,iw(iplon))
                  zomco(iplon,jk) = ztauc(iplon,jk) * zomcc(iplon,jk) + ptaucmc(iplon,ikl,iw(iplon)) * pomgcmc(iplon,ikl,iw(iplon)) 
                  zgco (iplon,jk) = (ptaucmc(iplon,ikl,iw(iplon)) * pomgcmc(iplon,ikl,iw(iplon)) * pasycmc(iplon,ikl,iw(iplon)) + &
                              ztauc(iplon,jk) * zomcc(iplon,jk) * zgcc(iplon,jk)) / zomco(iplon,jk)
                  zomco(iplon,jk) = zomco(iplon,jk) / ztauo(iplon,jk)
                            ! Total sky optical parameters (if cloud properties not delta scaled)
                            !   Use this code if cloud properties are not derived in rrtmg_sw_cldprop
               elseif (icpr .eq. 0) then
                  ztauo(iplon,jk) = ztaur(iplon,ikl,iw(iplon)) + ztaug(iplon,ikl,iw(iplon)) + ptaua(iplon,ikl,ibm) + ptaucmc(iplon,ikl,iw(iplon))
                  zomco(iplon,jk) = ptaua(iplon,ikl,ibm) * pomga(iplon,ikl,ibm) + ptaucmc(iplon,ikl,iw(iplon)) * pomgcmc(iplon,ikl,iw(iplon)) + &
                              ztaur(iplon,ikl,iw(iplon)) * 1.0_r8
                  zgco (iplon,jk) = (ptaucmc(iplon,ikl,iw(iplon)) * pomgcmc(iplon,ikl,iw(iplon)) * pasycmc(iplon,ikl,iw(iplon)) + &
                              ptaua(iplon,ikl,ibm)*pomga(iplon,ikl,ibm)*pasya(iplon,ikl,ibm)) / zomco(iplon,jk)
                  zomco(iplon,jk) = zomco(iplon,jk) / ztauo(iplon,jk)
                            ! Delta scaling - clouds
                            !   Use only if subroutine rrtmg_sw_cldprop is not used to get cloud properties and to apply delta scaling
                  zf = zgco(iplon,jk) * zgco(iplon,jk)
                  zwf = zomco(iplon,jk) * zf
                  ztauo(iplon,jk) = (1._r8 - zwf) * ztauo(iplon,jk)
                  zomco(iplon,jk) = (zomco(iplon,jk) - zwf) / (1.0_r8 - zwf)
                  zgco (iplon,jk) = (zgco(iplon,jk) - zf) / (1.0_r8 - zf)
               endif 
                        ! End of layer loop
            enddo
				END DO
				DO iplon=1,ncol

                    ! Clear sky reflectivities
            call reftra_sw (klev,ncol, &
lrtchkclr, zgcc, prmu0, ztauc, zomcc, &
zrefc, zrefdc, ztrac, ztradc)
                    ! Total sky reflectivities
            call reftra_sw (klev, ncol, &
lrtchkcld, zgco, prmu0, ztauo, zomco, &
zrefo, zrefdo, ztrao, ztrado)
			END DO
				DO iplon=1,ncol
            do jk=1,klev
                        ! Combine clear and cloudy contributions for total sky
               ikl = klev+1-jk 
               zclear(iplon) = 1.0_r8 - pcldfmc(iplon,ikl,iw(iplon))
               zcloud(iplon) = pcldfmc(iplon,ikl,iw(iplon))
               zref(iplon,jk) = zclear(iplon)*zrefc(iplon,jk) + zcloud(iplon)*zrefo(iplon,jk)
               zrefd(iplon,jk)= zclear(iplon)*zrefdc(iplon,jk) + zcloud(iplon)*zrefdo(iplon,jk)
               ztra(iplon,jk) = zclear(iplon)*ztrac(iplon,jk) + zcloud(iplon)*ztrao(iplon,jk)
               ztrad(iplon,jk)= zclear(iplon)*ztradc(iplon,jk) + zcloud(iplon)*ztrado(iplon,jk)
                        ! Direct beam transmittance
                        ! Clear
                        !                zdbtmc(iplon) = exp(-ztauc(iplon,jk) / prmu0)
                        ! Use exponential lookup table for transmittance, or expansion of
                        ! exponential for low tau
               ze1(iplon) = ztauc(iplon,jk) / prmu0(iplon)
               if (ze1(iplon) .le. od_lo) then
                  zdbtmc(iplon) = 1._r8 - ze1(iplon) + 0.5_r8 * ze1(iplon) * ze1(iplon)
               else
                  tblind(iplon) = ze1(iplon) / (bpade + ze1(iplon))
                  itind(iplon) = tblint * tblind(iplon) + 0.5_r8
                  zdbtmc(iplon) = exp_tbl(itind(iplon))
               endif
               zdbtc(iplon,jk) = zdbtmc(iplon)
               ztdbtc(iplon,jk+1) = zdbtc(iplon,jk)*ztdbtc(iplon,jk)
                        ! Clear + Cloud
                        !                zdbtmo(iplon) = exp(-ztauo(iplon,jk) / prmu0)
                        ! Use exponential lookup table for transmittance, or expansion of
                        ! exponential for low tau
               ze1(iplon) = ztauo(iplon,jk) / prmu0(iplon)
               if (ze1(iplon) .le. od_lo) then
                  zdbtmo(iplon) = 1._r8 - ze1(iplon) + 0.5_r8 * ze1(iplon) * ze1(iplon)
               else
                  tblind(iplon) = ze1(iplon) / (bpade + ze1(iplon))
                  itind(iplon) = tblint * tblind(iplon) + 0.5_r8
                  zdbtmo(iplon) = exp_tbl(itind(iplon))
               endif
               zdbt(iplon,jk) = zclear(iplon)*zdbtmc(iplon) + zcloud(iplon)*zdbtmo(iplon)
               ztdbt(iplon,jk+1) = zdbt(iplon,jk)*ztdbt(iplon,jk)
            enddo           
                    ! Vertical quadrature for clear-sky fluxes
					END DO
!					DO iplon=1,ncol
            		call vrtqdr_sw(ncol,klev, iw, &
zrefc, zrefdc, ztrac, ztradc, &
zdbtc, zrdndc, zrupc, zrupdc, ztdbtc, &
zcd, zcu)
                    ! Vertical quadrature for cloudy fluxes
            		call vrtqdr_sw(ncol,klev, iw, &
zref, zrefd, ztra, ztrad, &
zdbt, zrdnd, zrup, zrupd, ztdbt, &
zfd, zfu)
!					END DO
					DO iplon=1,ncol
								! Upwelling and downwelling fluxes at levels
                    !   Two-stream calculations go from top to bottom;
                    !   layer indexing is reversed to go bottom to top for output arrays
            do jk=1,klev+1
               ikl=klev+2-jk
                        ! Accumulate spectral fluxes over bands - inactive
                        !               zbbfu(ikl) = zbbfu(ikl) + zincflx(iplon,iw(iplon))*zfu(iplon,jk,iw(iplon))
                        !               zbbfd(ikl) = zbbfd(ikl) + zincflx(iplon,iw(iplon))*zfd(iplon,jk,iw(iplon))
                        !               zbbcu(ikl) = zbbcu(ikl) + zincflx(iplon,iw(iplon))*zcu(iplon,jk,iw(iplon))
                        !               zbbcd(ikl) = zbbcd(ikl) + zincflx(iplon,iw(iplon))*zcd(iplon,jk,iw(iplon))
                        !               zbbfddir(ikl) = zbbfddir(ikl) + zincflx(iplon,iw(iplon))*ztdbt_nodel(iplon,jk)
                        !               zbbcddir(ikl) = zbbcddir(ikl) + zincflx(iplon,iw(iplon))*ztdbtc_nodel(iplon,jk)
               pbbfsu(iplon,ibm,ikl) = pbbfsu(iplon,ibm,ikl) + zincflx(iplon,iw(iplon))*zfu(iplon,jk,iw(iplon))
               pbbfsd(iplon,ibm,ikl) = pbbfsd(iplon,ibm,ikl) + zincflx(iplon,iw(iplon))*zfd(iplon,jk,iw(iplon))
                        ! Accumulate spectral fluxes over whole spectrum
               pbbfu(iplon,ikl) = pbbfu(iplon,ikl) + zincflx(iplon,iw(iplon))*zfu(iplon,jk,iw(iplon))
               pbbfd(iplon,ikl) = pbbfd(iplon,ikl) +zincflx(iplon,iw(iplon))*zfd(iplon,jk,iw(iplon))
               pbbcu(iplon,ikl) = pbbcu(iplon,ikl) + zincflx(iplon,iw(iplon))*zcu(iplon,jk,iw(iplon))
               pbbcd(iplon,ikl) = pbbcd(iplon,ikl) + zincflx(iplon,iw(iplon))*zcd(iplon,jk,iw(iplon))
               if (idelm .eq. 0) then
                  pbbfddir(iplon,ikl) = pbbfddir(iplon,ikl) + zincflx(iplon,iw(iplon))*ztdbt_nodel(iplon,jk)
                  pbbcddir(iplon,ikl) = pbbcddir(iplon,ikl) + zincflx(iplon,iw(iplon))*ztdbtc_nodel(iplon,jk)
               elseif (idelm .eq. 1) then
                  pbbfddir(iplon,ikl) = pbbfddir(iplon,ikl) + zincflx(iplon,iw(iplon))*ztdbt(iplon,jk)
                  pbbcddir(iplon,ikl) = pbbcddir(iplon,ikl) + zincflx(iplon,iw(iplon))*ztdbtc(iplon,jk)
               endif
                        ! Accumulate direct fluxes for UV/visible bands
               if (ibm >= 10 .and. ibm <= 13) then
                  puvcd(iplon,ikl) = puvcd(iplon,ikl) + zincflx(iplon,iw(iplon))*zcd(iplon,jk,iw(iplon))
                  puvfd(iplon,ikl) = puvfd(iplon,ikl) + zincflx(iplon,iw(iplon))*zfd(iplon,jk,iw(iplon))
                  if (idelm .eq. 0) then
                     puvfddir(iplon,ikl) = puvfddir(iplon,ikl) + zincflx(iplon,iw(iplon))*ztdbt_nodel(iplon,jk)
                     puvcddir(iplon,ikl) = puvcddir(iplon,ikl) + zincflx(iplon,iw(iplon))*ztdbtc_nodel(iplon,jk)
                  elseif (idelm .eq. 1) then
                     puvfddir(iplon,ikl) = puvfddir(iplon,ikl) + zincflx(iplon,iw(iplon))*ztdbt(iplon,jk)
                     puvcddir(iplon,ikl) = puvcddir(iplon,ikl) + zincflx(iplon,iw(iplon))*ztdbtc(iplon,jk)
                  endif
                            ! band 9 is half-NearIR and half-Visible
               else if (ibm == 9) then  
                  puvcd(iplon,ikl) = puvcd(iplon,ikl) + 0.5_r8*zincflx(iplon,iw(iplon))*zcd(iplon,jk,iw(iplon))
                  puvfd(iplon,ikl) = puvfd(iplon,ikl) + 0.5_r8*zincflx(iplon,iw(iplon))*zfd(iplon,jk,iw(iplon))
                  pnicd(iplon,ikl) = pnicd(iplon,ikl) + 0.5_r8*zincflx(iplon,iw(iplon))*zcd(iplon,jk,iw(iplon))
                  pnifd(iplon,ikl) = pnifd(iplon,ikl) + 0.5_r8*zincflx(iplon,iw(iplon))*zfd(iplon,jk,iw(iplon))
                  if (idelm .eq. 0) then
                     puvfddir(iplon,ikl) = puvfddir(iplon,ikl) + 0.5_r8*zincflx(iplon,iw(iplon))*ztdbt_nodel(iplon,jk)
                     puvcddir(iplon,ikl) = puvcddir(iplon,ikl) + 0.5_r8*zincflx(iplon,iw(iplon))*ztdbtc_nodel(iplon,jk)
                     pnifddir(iplon,ikl) = pnifddir(iplon,ikl) + 0.5_r8*zincflx(iplon,iw(iplon))*ztdbt_nodel(iplon,jk)
                     pnicddir(iplon,ikl) = pnicddir(iplon,ikl) + 0.5_r8*zincflx(iplon,iw(iplon))*ztdbtc_nodel(iplon,jk)
                  elseif (idelm .eq. 1) then
                     puvfddir(iplon,ikl) = puvfddir(iplon,ikl) + 0.5_r8*zincflx(iplon,iw(iplon))*ztdbt(iplon,jk)
                     puvcddir(iplon,ikl) = puvcddir(iplon,ikl) + 0.5_r8*zincflx(iplon,iw(iplon))*ztdbtc(iplon,jk)
                     pnifddir(iplon,ikl) = pnifddir(iplon,ikl) + 0.5_r8*zincflx(iplon,iw(iplon))*ztdbt(iplon,jk)
                     pnicddir(iplon,ikl) = pnicddir(iplon,ikl) + 0.5_r8*zincflx(iplon,iw(iplon))*ztdbtc(iplon,jk)
                  endif
                  pnicu(iplon,ikl) = pnicu(iplon,ikl) + 0.5_r8*zincflx(iplon,iw(iplon))*zcu(iplon,jk,iw(iplon))
                  pnifu(iplon,ikl) = pnifu(iplon,ikl) + 0.5_r8*zincflx(iplon,iw(iplon))*zfu(iplon,jk,iw(iplon))
                            ! Accumulate direct fluxes for near-IR bands
               else if (ibm == 14 .or. ibm <= 8) then  
                  pnicd(iplon,ikl) = pnicd(iplon,ikl) + zincflx(iplon,iw(iplon))*zcd(iplon,jk,iw(iplon))
                  pnifd(iplon,ikl) = pnifd(iplon,ikl) + zincflx(iplon,iw(iplon))*zfd(iplon,jk,iw(iplon))
                  if (idelm .eq. 0) then
                     pnifddir(iplon,ikl) = pnifddir(iplon,ikl) + zincflx(iplon,iw(iplon))*ztdbt_nodel(iplon,jk)
                     pnicddir(iplon,ikl) = pnicddir(iplon,ikl) + zincflx(iplon,iw(iplon))*ztdbtc_nodel(iplon,jk)
                  elseif (idelm .eq. 1) then
                     pnifddir(iplon,ikl) = pnifddir(iplon,ikl) + zincflx(iplon,iw(iplon))*ztdbt(iplon,jk)
                     pnicddir(iplon,ikl) = pnicddir(iplon,ikl) + zincflx(iplon,iw(iplon))*ztdbtc(iplon,jk)
                  endif
                            ! Added for net near-IR flux diagnostic
                  pnicu(iplon,ikl) = pnicu(iplon,ikl) + zincflx(iplon,iw(iplon))*zcu(iplon,jk,iw(iplon))
                  pnifu(iplon,ikl) = pnifu(iplon,ikl) + zincflx(iplon,iw(iplon))*zfu(iplon,jk,iw(iplon))
               endif
            enddo
                    ! End loop on jg, g-point interval
         enddo             
                ! End loop on jb, spectral band
      enddo                   
        end do 
        END SUBROUTINE spcvmc_sw
    END MODULE rrtmg_sw_spcvmc
