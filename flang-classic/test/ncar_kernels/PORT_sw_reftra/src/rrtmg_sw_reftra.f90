
! KGEN-generated Fortran source file
!
! Filename    : rrtmg_sw_reftra.f90
! Generated at: 2015-07-31 20:52:25
! KGEN version: 0.4.13



    MODULE rrtmg_sw_reftra
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
        USE rrsw_tbl, ONLY: od_lo
        USE rrsw_tbl, ONLY: bpade
        USE rrsw_tbl, ONLY: tblint
        USE rrsw_tbl, ONLY: exp_tbl
        USE rrsw_vsn, ONLY: hvrrft
        IMPLICIT NONE
        CONTAINS

        ! write subroutines
        ! No subroutines
        ! No module extern variables
        ! --------------------------------------------------------------------

        SUBROUTINE reftra_sw(nlayers, ncol, lrtchk, pgg, prmuz, ptau, pw, pref, prefd, ptra, ptrad)
            ! --------------------------------------------------------------------
            ! Purpose: computes the reflectivity and transmissivity of a clear or
            !   cloudy layer using a choice of various approximations.
            !
            ! Interface:  *rrtmg_sw_reftra* is called by *rrtmg_sw_spcvrt*
            !
            ! Description:
            ! explicit arguments :
            ! --------------------
            ! inputs
            ! ------
            !      lrtchk  = .t. for all layers in clear profile
            !      lrtchk  = .t. for cloudy layers in cloud profile
            !              = .f. for clear layers in cloud profile
            !      pgg     = assymetry factor
            !      prmuz   = cosine solar zenith angle
            !      ptau    = optical thickness
            !      pw      = single scattering albedo
            !
            ! outputs
            ! -------
            !      pref    : collimated beam reflectivity
            !      prefd   : diffuse beam reflectivity
            !      ptra    : collimated beam transmissivity
            !      ptrad   : diffuse beam transmissivity
            !
            !
            ! Method:
            ! -------
            !      standard delta-eddington, p.i.f.m., or d.o.m. layer calculations.
            !      kmodts  = 1 eddington (joseph et al., 1976)
            !              = 2 pifm (zdunkowski et al., 1980)
            !              = 3 discrete ordinates (liou, 1973)
            !
            !
            ! Modifications:
            ! --------------
            ! Original: J-JMorcrette, ECMWF, Feb 2003
            ! Revised for F90 reformatting: MJIacono, AER, Jul 2006
            ! Revised to add exponential lookup table: MJIacono, AER, Aug 2007
            !
            ! ------------------------------------------------------------------
            ! ------- Declarations ------
            ! ------- Input -------
            INTEGER, intent(in) :: nlayers
            INTEGER, intent(in) :: ncol


            LOGICAL, intent(in) :: lrtchk(:,:) ! Logical flag for reflectivity and
            ! and transmissivity calculation;
            !   Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: pgg(:,:) ! asymmetry parameter
            !   Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: ptau(:,:) ! optical depth
            !   Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: pw(:,:) ! single scattering albedo
            !   Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: prmuz(:) ! cosine of solar zenith angle
            ! ------- Output -------
            REAL(KIND=r8), intent(inout) :: pref(:,:) ! direct beam reflectivity
            !   Dimensions: (nlayers+1)
            REAL(KIND=r8), intent(inout) :: prefd(:,:) ! diffuse beam reflectivity
            !   Dimensions: (nlayers+1)
            REAL(KIND=r8), intent(inout) :: ptra(:,:) ! direct beam transmissivity
            !   Dimensions: (nlayers+1)
            REAL(KIND=r8), intent(inout) :: ptrad(:,:) ! diffuse beam transmissivity
            !   Dimensions: (nlayers+1)
            ! ------- Local -------
            INTEGER :: kmodts
            INTEGER :: jk
            INTEGER :: icol
            INTEGER :: itind
            REAL(KIND=r8) :: tblind
            REAL(KIND=r8) :: za
            REAL(KIND=r8) :: za1
            REAL(KIND=r8) :: za2
            REAL(KIND=r8) :: zbeta
            REAL(KIND=r8) :: zdenr
            REAL(KIND=r8) :: zdent
            REAL(KIND=r8) :: zdend
            REAL(KIND=r8) :: ze1
            REAL(KIND=r8) :: ze2
            REAL(KIND=r8) :: zem1
            REAL(KIND=r8) :: zep1
            REAL(KIND=r8) :: zem2
            REAL(KIND=r8) :: zep2
            REAL(KIND=r8) :: zemm
            REAL(KIND=r8) :: zg
            REAL(KIND=r8) :: zg3
            REAL(KIND=r8) :: zgamma1
            REAL(KIND=r8) :: zgamma2
            REAL(KIND=r8) :: zgamma3
            REAL(KIND=r8) :: zgamma4
            REAL(KIND=r8) :: zgt
            REAL(KIND=r8) :: zr1
            REAL(KIND=r8) :: zr2
            REAL(KIND=r8) :: zr3
            REAL(KIND=r8) :: zr4
            REAL(KIND=r8) :: zr5
            REAL(KIND=r8) :: zrk
            REAL(KIND=r8) :: zrp
            REAL(KIND=r8) :: zrp1
            REAL(KIND=r8) :: zrm1
            REAL(KIND=r8) :: zrk2
            REAL(KIND=r8) :: zrpp
            REAL(KIND=r8) :: zrkg
            REAL(KIND=r8) :: zsr3
            REAL(KIND=r8) :: zto1
            REAL(KIND=r8) :: zt1
            REAL(KIND=r8) :: zt2
            REAL(KIND=r8) :: zt3
            REAL(KIND=r8) :: zt4
            REAL(KIND=r8) :: zt5
            REAL(KIND=r8) :: zwcrit
            REAL(KIND=r8) :: zw
            REAL(KIND=r8) :: zwo
            REAL(KIND=r8) :: temp1, temp2
            REAL(KIND=r8), parameter :: eps = 1.e-08_r8
            !     ------------------------------------------------------------------
            ! Initialize

!DIR$ ASSUME_ALIGNED lrtchk:256, pgg:256, ptau:256, pw:256, prmuz:256, pref:256, prefd:256, ptra:256, ptrad:256

      hvrrft = '$Revision: 1.2 $'
      zsr3=sqrt(3._r8)
      zwcrit=0.9999995_r8
      kmodts=2
	do icol = 1,ncol
!DIR$ VECTOR ALWAYS ALIGNED
      do jk=1, nlayers
         if (.not.lrtchk(icol,jk)) then
            pref(icol,jk) =0._r8
            ptra(icol,jk) =1._r8
            prefd(icol,jk)=0._r8
            ptrad(icol,jk)=1._r8
         else
            zto1=ptau(icol,jk)
            zw  =pw(icol,jk)
            zg  =pgg(icol,jk)  
                        ! General two-stream expressions
            zg3= 3._r8 * zg
            if (kmodts == 1) then
               zgamma1= (7._r8 - zw * (4._r8 + zg3)) * 0.25_r8
               zgamma2=-(1._r8 - zw * (4._r8 - zg3)) * 0.25_r8
               zgamma3= (2._r8 - zg3 * prmuz(icol) ) * 0.25_r8
            else if (kmodts == 2) then  
               zgamma1= (8._r8 - zw * (5._r8 + zg3)) * 0.25_r8
               zgamma2=  3._r8 *(zw * (1._r8 - zg )) * 0.25_r8
               zgamma3= (2._r8 - zg3 * prmuz(icol) ) * 0.25_r8
            else if (kmodts == 3) then  
               zgamma1= zsr3 * (2._r8 - zw * (1._r8 + zg)) * 0.5_r8
               zgamma2= zsr3 * zw * (1._r8 - zg ) * 0.5_r8
               zgamma3= (1._r8 - zsr3 * zg * prmuz(icol) ) * 0.5_r8
            end if
            zgamma4= 1._r8 - zgamma3
                        ! Recompute original s.s.a. to test for conservative solution
            !zwo= zw / (1._r8 - (1._r8 - zw) * (zg / (1._r8 - zg))**2)
            temp1 = 1._r8 - 2._r8 * zg
            zwo= zw * (temp1 + zg**2)/(temp1 + zg**2 * zw)
            if (zwo >= zwcrit) then
                            ! Conservative scattering
               za  = zgamma1 * prmuz(icol) 
               za1 = za - zgamma3
               zgt = zgamma1 * zto1
                            ! Homogeneous reflectance and transmittance,
                            ! collimated beam
               ze1 = min ( zto1 / prmuz(icol) , 500._r8)
                            !               ze2 = exp( -ze1 )
                            ! Use exponential lookup table for transmittance, or expansion of
                            ! exponential for low tau
               if (ze1 .le. od_lo) then 
                  ze2 = 1._r8 - ze1 + 0.5_r8 * ze1 * ze1
               else
                  tblind = ze1 / (bpade + ze1)
                  itind = tblint * tblind + 0.5_r8
                  ze2 = exp_tbl(itind)
               endif
                            !
               pref(icol,jk) = (zgt - za1 * (1._r8 - ze2)) / (1._r8 + zgt)
               ptra(icol,jk) = 1._r8 - pref(icol,jk)
                            ! isotropic incidence
               prefd(icol,jk) = zgt / (1._r8 + zgt)
               ptrad(icol,jk) = 1._r8 - prefd(icol,jk)        
                            ! This is applied for consistency between total (delta-scaled) and direct (unscaled)
                            ! calculations at very low optical depths (tau < 1.e-4) when the exponential lookup
                            ! table returns a transmittance of 1.0.
               if (ze2 .eq. 1.0_r8) then 
                  pref(icol,jk) = 0.0_r8
                  ptra(icol,jk) = 1.0_r8
                  prefd(icol,jk) = 0.0_r8
                  ptrad(icol,jk) = 1.0_r8
               endif
            else
                            ! Non-conservative scattering
               za1 = zgamma1 * zgamma4 + zgamma2 * zgamma3
               za2 = zgamma1 * zgamma3 + zgamma2 * zgamma4
               zrk = sqrt ( zgamma1**2 - zgamma2**2)
               !zrk = sqrt ( (zgamma1 - zgamma2) * (zgamma1 + zgamma2) )
               zrp = zrk * prmuz(icol)               
               zrp1 = 1._r8 + zrp
               zrm1 = 1._r8 - zrp
               zrk2 = 2._r8 * zrk
               zrpp = 1._r8 - zrp*zrp
               zrkg = zrk + zgamma1
               zr1  = zrm1 * (za2 + zrk * zgamma3)
               zr2  = zrp1 * (za2 - zrk * zgamma3)
               zr3  = zrk2 * (zgamma3 - za2 * prmuz(icol) )
               zr4  = zrpp * zrkg
               zr5  = zrpp * (zrk - zgamma1)
               zt1  = zrp1 * (za1 + zrk * zgamma4)
               zt2  = zrm1 * (za1 - zrk * zgamma4)
               zt3  = zrk2 * (zgamma4 + za1 * prmuz(icol) )
               zt4  = zr4
               zt5  = zr5
               zbeta = (zgamma1 - zrk) / zrkg !- zr5 / zr4 !- zr5 / zr4 !- zr5 / zr4 !- zr5 / zr4
                            ! Homogeneous reflectance and transmittance
               ze1 = min ( zrk * zto1, 500._r8)
               ze2 = min ( zto1 / prmuz(icol) , 500._r8)
                            !
                            ! Original
                            !              zep1 = exp( ze1 )
                            !              zem1 = exp(-ze1 )
                            !              zep2 = exp( ze2 )
                            !              zem2 = exp(-ze2 )
                            !
                            ! Revised original, to reduce exponentials
                            !              zep1 = exp( ze1 )
                            !              zem1 = 1._r8 / zep1
                            !              zep2 = exp( ze2 )
                            !              zem2 = 1._r8 / zep2
                            !
                            ! Use exponential lookup table for transmittance, or expansion of
                            ! exponential for low tau
               if (ze1 .le. od_lo) then 
                  zem1 = 1._r8 - ze1 + 0.5_r8 * ze1 * ze1
                  zep1 = 1._r8 / zem1
               else
                  tblind = ze1 / (bpade + ze1)
                  itind = tblint * tblind + 0.5_r8
                  zem1 = exp_tbl(itind)
                  zep1 = 1._r8 / zem1
               endif
               if (ze2 .le. od_lo) then 
                  zem2 = 1._r8 - ze2 + 0.5_r8 * ze2 * ze2
                  zep2 = 1._r8 / zem2
               else
                  tblind = ze2 / (bpade + ze2)
                  itind = tblint * tblind + 0.5_r8
                  zem2 = exp_tbl(itind)
                  zep2 = 1._r8 / zem2
               endif
                            ! collimated beam
               zdenr = zr4*zep1 + zr5*zem1
               temp2 = 1._r8 / zdenr
               !zdent = zt4*zep1 + zt5*zem1
               !temp2 = zem1 / (zr4 + zr5 * zem1**2)
               if (zdenr .ge. -eps .and. zdenr .le. eps) then
                  pref(icol,jk) = eps
                  ptra(icol,jk) = zem2
               else
                  !pref(icol,jk) = zw * (zr1*zep1 - zr2*zem1 - zr3*zem2) / zdenr
                  pref(icol,jk) = zw * (zr1*zep1 - zr2*zem1 - zr3*zem2) * temp2
                  !ptra(icol,jk) = zem2 - zem2 * zw * (zt1*zep1 - zt2*zem1 - zt3*zep2) / zdent
                  ptra(icol,jk) = zem2 - zem2 * zw * (zt1*zep1 - zt2*zem1 - zt3*zep2) * temp2
               endif 
                            ! diffuse beam
               zemm = zem1*zem1
               zdend = 1._r8 / ( (1._r8 - zbeta*zemm ) * zrkg)
               prefd(icol,jk) =  zgamma2 * (1._r8 - zemm) * zdend
               ptrad(icol,jk) =  zrk2*zem1*zdend
            endif
         endif         
      enddo
end do
        END SUBROUTINE reftra_sw
    END MODULE rrtmg_sw_reftra
