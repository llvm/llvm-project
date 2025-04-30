
! KGEN-generated Fortran source file
!
! Filename    : rrtmg_sw_taumol.f90
! Generated at: 2015-07-31 20:45:42
! KGEN version: 0.4.13



    MODULE rrtmg_sw_taumol
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
        !      use parrrsw, only : mg, jpband, nbndsw, ngptsw
        USE rrsw_con, ONLY: oneminus
        USE rrsw_wvn, ONLY: nspa
        USE rrsw_wvn, ONLY: nspb
        USE rrsw_vsn, ONLY: hvrtau
        IMPLICIT NONE
        CONTAINS

        ! write subroutines
        ! No subroutines
        ! No module extern variables
        !----------------------------------------------------------------------------

        SUBROUTINE taumol_sw(ncol, nlayers, colh2o, colco2, colch4, colo2, colo3, colmol, laytrop, jp, jt, jt1, fac00, fac01, &
        fac10, fac11, selffac, selffrac, indself, forfac, forfrac, indfor, sfluxzen, taug, taur)
            !----------------------------------------------------------------------------
            ! ******************************************************************************
            ! *                                                                            *
            ! *                 Optical depths developed for the                           *
            ! *                                                                            *
            ! *               RAPID RADIATIVE TRANSFER MODEL (RRTM)                        *
            ! *                                                                            *
            ! *                                                                            *
            ! *           ATMOSPHERIC AND ENVIRONMENTAL RESEARCH, INC.                     *
            ! *                       131 HARTWELL AVENUE                                  *
            ! *                       LEXINGTON, MA 02421                                  *
            ! *                                                                            *
            ! *                                                                            *
            ! *                          ELI J. MLAWER                                     *
            ! *                        JENNIFER DELAMERE                                   *
            ! *                        STEVEN J. TAUBMAN                                   *
            ! *                        SHEPARD A. CLOUGH                                   *
            ! *                                                                            *
            ! *                                                                            *
            ! *                                                                            *
            ! *                                                                            *
            ! *                      email:  mlawer@aer.com                                *
            ! *                      email:  jdelamer@aer.com                              *
            ! *                                                                            *
            ! *       The authors wish to acknowledge the contributions of the             *
            ! *       following people:  Patrick D. Brown, Michael J. Iacono,              *
            ! *       Ronald E. Farren, Luke Chen, Robert Bergstrom.                       *
            ! *                                                                            *
            ! ******************************************************************************
            ! *    TAUMOL                                                                  *
            ! *                                                                            *
            ! *    This file contains the subroutines TAUGBn (where n goes from            *
            ! *    1 to 28).  TAUGBn calculates the optical depths and Planck fractions    *
            ! *    per g-value and layer for band n.                                       *
            ! *                                                                            *
            ! * Output:  optical depths (unitless)                                         *
            ! *          fractions needed to compute Planck functions at every layer       *
            ! *              and g-value                                                   *
            ! *                                                                            *
            ! *    COMMON /TAUGCOM/  TAUG(MXLAY,MG)                                        *
            ! *    COMMON /PLANKG/   FRACS(MXLAY,MG)                                       *
            ! *                                                                            *
            ! * Input                                                                      *
            ! *                                                                            *
            ! *    PARAMETER (MG=16, MXLAY=203, NBANDS=14)                                 *
            ! *                                                                            *
            ! *    COMMON /FEATURES/ NG(NBANDS),NSPA(NBANDS),NSPB(NBANDS)                  *
            ! *    COMMON /PRECISE/  ONEMINUS                                              *
            ! *    COMMON /PROFILE/  NLAYERS,PAVEL(MXLAY),TAVEL(MXLAY),                    *
            ! *   &                  PZ(0:MXLAY),TZ(0:MXLAY),TBOUND                        *
            ! *    COMMON /PROFDATA/ LAYTROP,LAYSWTCH,LAYLOW,                              *
            ! *   &                  COLH2O(MXLAY),COLCO2(MXLAY),                          *
            ! *   &                  COLO3(MXLAY),COLN2O(MXLAY),COLCH4(MXLAY),             *
            ! *   &                  COLO2(MXLAY),CO2MULT(MXLAY)                           *
            ! *    COMMON /INTFAC/   FAC00(MXLAY),FAC01(MXLAY),                            *
            ! *   &                  FAC10(MXLAY),FAC11(MXLAY)                             *
            ! *    COMMON /INTIND/   JP(MXLAY),JT(MXLAY),JT1(MXLAY)                        *
            ! *    COMMON /SELF/     SELFFAC(MXLAY), SELFFRAC(MXLAY), INDSELF(MXLAY)       *
            ! *                                                                            *
            ! *    Description:                                                            *
            ! *    NG(IBAND) - number of g-values in band IBAND                            *
            ! *    NSPA(IBAND) - for the lower atmosphere, the number of reference         *
            ! *                  atmospheres that are stored for band IBAND per            *
            ! *                  pressure level and temperature.  Each of these            *
            ! *                  atmospheres has different relative amounts of the         *
            ! *                  key species for the band (i.e. different binary           *
            ! *                  species parameters).                                      *
            ! *    NSPB(IBAND) - same for upper atmosphere                                 *
            ! *    ONEMINUS - since problems are caused in some cases by interpolation     *
            ! *               parameters equal to or greater than 1, for these cases       *
            ! *               these parameters are set to this value, slightly < 1.        *
            ! *    PAVEL - layer pressures (mb)                                            *
            ! *    TAVEL - layer temperatures (degrees K)                                  *
            ! *    PZ - level pressures (mb)                                               *
            ! *    TZ - level temperatures (degrees K)                                     *
            ! *    LAYTROP - layer at which switch is made from one combination of         *
            ! *              key species to another                                        *
            ! *    COLH2O, COLCO2, COLO3, COLN2O, COLCH4 - column amounts of water         *
            ! *              vapor,carbon dioxide, ozone, nitrous ozide, methane,          *
            ! *              respectively (molecules/cm**2)                                *
            ! *    CO2MULT - for bands in which carbon dioxide is implemented as a         *
            ! *              trace species, this is the factor used to multiply the        *
            ! *              band's average CO2 absorption coefficient to get the added    *
            ! *              contribution to the optical depth relative to 355 ppm.        *
            ! *    FACij(LAY) - for layer LAY, these are factors that are needed to        *
            ! *                 compute the interpolation factors that multiply the        *
            ! *                 appropriate reference k-values.  A value of 0 (1) for      *
            ! *                 i,j indicates that the corresponding factor multiplies     *
            ! *                 reference k-value for the lower (higher) of the two        *
            ! *                 appropriate temperatures, and altitudes, respectively.     *
            ! *    JP - the index of the lower (in altitude) of the two appropriate        *
            ! *         reference pressure levels needed for interpolation                 *
            ! *    JT, JT1 - the indices of the lower of the two appropriate reference     *
            ! *              temperatures needed for interpolation (for pressure           *
            ! *              levels JP and JP+1, respectively)                             *
            ! *    SELFFAC - scale factor needed to water vapor self-continuum, equals     *
            ! *              (water vapor density)/(atmospheric density at 296K and        *
            ! *              1013 mb)                                                      *
            ! *    SELFFRAC - factor needed for temperature interpolation of reference     *
            ! *               water vapor self-continuum data                              *
            ! *    INDSELF - index of the lower of the two appropriate reference           *
            ! *              temperatures needed for the self-continuum interpolation      *
            ! *                                                                            *
            ! * Data input                                                                 *
            ! *    COMMON /Kn/ KA(NSPA(n),5,13,MG), KB(NSPB(n),5,13:59,MG), SELFREF(10,MG) *
            ! *       (note:  n is the band number)                                        *
            ! *                                                                            *
            ! *    Description:                                                            *
            ! *    KA - k-values for low reference atmospheres (no water vapor             *
            ! *         self-continuum) (units: cm**2/molecule)                            *
            ! *    KB - k-values for high reference atmospheres (all sources)              *
            ! *         (units: cm**2/molecule)                                            *
            ! *    SELFREF - k-values for water vapor self-continuum for reference         *
            ! *              atmospheres (used below LAYTROP)                              *
            ! *              (units: cm**2/molecule)                                       *
            ! *                                                                            *
            ! *    DIMENSION ABSA(65*NSPA(n),MG), ABSB(235*NSPB(n),MG)                     *
            ! *    EQUIVALENCE (KA,ABSA),(KB,ABSB)                                         *
            ! *                                                                            *
            ! *****************************************************************************
            !
            ! Modifications
            !
            ! Revised: Adapted to F90 coding, J.-J.Morcrette, ECMWF, Feb 2003
            ! Revised: Modified for g-point reduction, MJIacono, AER, Dec 2003
            ! Revised: Reformatted for consistency with rrtmg_lw, MJIacono, AER, Jul 2006
            !
            ! ------- Declarations -------
            ! ----- Input -----
            INTEGER, intent(in) :: nlayers ! total number of layers
            INTEGER, intent(in) :: ncol ! total number of layers
            INTEGER, intent(in) :: laytrop(ncol) ! tropopause layer index
            INTEGER, intent(in) :: jp(ncol,nlayers) !
            !INTEGER, intent(in) :: nlayers ! total number of layers
            !   Dimensions: (nlayers)
            INTEGER, intent(in) :: jt(ncol,nlayers) !
            !   Dimensions: (nlayers)
            INTEGER, intent(in) :: jt1(ncol,nlayers) !
            !   Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: colh2o(ncol,nlayers) ! column amount (h2o)
            !   Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: colco2(ncol,nlayers) ! column amount (co2)
            !   Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: colo3(ncol,nlayers) ! column amount (o3)
            !   Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: colch4(ncol,nlayers) ! column amount (ch4)
            !   Dimensions: (nlayers)
            !   Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: colo2(ncol,nlayers) ! column amount (o2)
            !   Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: colmol(ncol,nlayers) !
            !   Dimensions: (nlayers)
            INTEGER, intent(in) :: indself(ncol,nlayers)
            !   Dimensions: (nlayers)
            INTEGER, intent(in) :: indfor(ncol,nlayers)
            !   Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: selffac(ncol,nlayers)
            !   Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: selffrac(ncol,nlayers)
            !   Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: forfac(ncol,nlayers)
            !   Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: forfrac(ncol,nlayers)
            !   Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: fac01(ncol,nlayers)
            REAL(KIND=r8), intent(in) :: fac10(ncol,nlayers)
            REAL(KIND=r8), intent(in) :: fac11(ncol,nlayers)
            REAL(KIND=r8), intent(in) :: fac00(ncol,nlayers) !
            !   Dimensions: (nlayers)
            ! ----- Output -----
            REAL(KIND=r8), intent(out) :: sfluxzen(:,:) ! solar source function
            !   Dimensions: (ngptsw)
            REAL(KIND=r8), intent(out) :: taug(:,:,:) ! gaseous optical depth
            !   Dimensions: (nlayers,ngptsw)
            REAL(KIND=r8), intent(out) :: taur(:,:,:) ! Rayleigh
            INTEGER :: icol
            !   Dimensions: (nlayers,ngptsw)
            !      real(kind=r8), intent(out) :: ssa(:,:)             ! single scattering albedo (inactive)
            !   Dimensions: (nlayers,ngptsw)
      hvrtau = '$Revision: 1.2 $'
      call taumol16()
      call taumol17
      call taumol18
      call taumol19
      call taumol20
      call taumol21
      call taumol22
      call taumol23
      call taumol24
      call taumol25
      call taumol26
      call taumol27
      call taumol28
      call taumol29
            !-------------
            CONTAINS
            !-------------
            !----------------------------------------------------------------------------

            SUBROUTINE taumol16()
                !----------------------------------------------------------------------------
                !
                !     band 16:  2600-3250 cm-1 (low - h2o,ch4; high - ch4)
                !
                !----------------------------------------------------------------------------
                ! ------- Modules -------
                USE parrrsw, ONLY: ng16
                USE rrsw_kg16, ONLY: strrat1
                USE rrsw_kg16, ONLY: rayl
                USE rrsw_kg16, ONLY: forref
                USE rrsw_kg16, ONLY: absa
                USE rrsw_kg16, ONLY: selfref
                USE rrsw_kg16, ONLY: layreffr
                USE rrsw_kg16, ONLY: absb
                USE rrsw_kg16, ONLY: sfluxref
                ! ------- Declarations -------
                !INTEGER, intent(in) ::ncol ! total number of layers
                ! Local
                INTEGER :: lay
                INTEGER :: js
                INTEGER :: ind0
                INTEGER :: ind1
                INTEGER :: inds
                INTEGER :: indf
                INTEGER :: ig
                INTEGER :: laysolfr
                REAL(KIND=r8) :: speccomb
                REAL(KIND=r8) :: specparm
                REAL(KIND=r8) :: specmult
                REAL(KIND=r8) :: fs
                REAL(KIND=r8) :: fac000
                REAL(KIND=r8) :: fac010
                REAL(KIND=r8) :: fac100
                REAL(KIND=r8) :: fac110
                REAL(KIND=r8) :: fac001
                REAL(KIND=r8) :: fac011
                REAL(KIND=r8) :: fac101
                REAL(KIND=r8) :: fac111
                REAL(KIND=r8) :: tauray
                ! Compute the optical depth by interpolating in ln(pressure),
                ! temperature, and appropriate species.  Below LAYTROP, the water
                ! vapor self-continuum is interpolated (in temperature) separately.
                ! Lower atmosphere loop
                !print*,"taumol 16 :: before lay loop"
     do icol=1,ncol
      do lay = 1, laytrop(icol)
                    !print*,'inside lay loop'
         speccomb = colh2o(icol,lay) + strrat1*colch4(icol,lay)
         specparm = colh2o(icol,lay)/speccomb 
         if (specparm .ge. oneminus) specparm = oneminus
         specmult = 8._r8*(specparm)
         js = 1 + int(specmult)
         fs = mod(specmult, 1._r8 )
         fac000 = (1._r8 - fs) * fac00(icol,lay)
         fac010 = (1._r8 - fs) * fac10(icol,lay)
         fac100 = fs * fac00(icol,lay)
         fac110 = fs * fac10(icol,lay)
         fac001 = (1._r8 - fs) * fac01(icol,lay)
         fac011 = (1._r8 - fs) * fac11(icol,lay)
         fac101 = fs * fac01(icol,lay)
         fac111 = fs * fac11(icol,lay)
         ind0 = ((jp(icol,lay)-1)*5+(jt(icol,lay)-1))*nspa(16) + js
         ind1 = (jp(icol,lay)*5+(jt1(icol,lay)-1))*nspa(16) + js
         inds = indself(icol,lay)
         indf = indfor(icol,lay)
         tauray = colmol(icol,lay) * rayl
         do ig = 1, ng16
            taug(icol,lay,ig) = speccomb * &
                (fac000 * absa(ind0   ,ig) + &
                 fac100 * absa(ind0 +1,ig) + &
                 fac010 * absa(ind0 +9,ig) + &
                 fac110 * absa(ind0+10,ig) + &
                 fac001 * absa(ind1   ,ig) + &
                 fac101 * absa(ind1 +1,ig) + &
                 fac011 * absa(ind1 +9,ig) + &
                 fac111 * absa(ind1+10,ig)) + &
                 colh2o(icol,lay) * &
                 (selffac(icol,lay) * (selfref(inds,ig) + &
                 selffrac(icol,lay) * &
                 (selfref(inds+1,ig) - selfref(inds,ig))) + &
                 forfac(icol,lay) * (forref(indf,ig) + &
                 forfrac(icol,lay) * &
                 (forref(indf+1,ig) - forref(indf,ig)))) 
                        !            ssa(lay,ig) = tauray/taug(lay,ig)
            taur(icol,lay,ig) = tauray
         enddo
      enddo
      laysolfr = nlayers
                ! Upper atmosphere loop
      do lay = laytrop(icol)+1, nlayers
         if (jp(icol,(lay-1)) .lt. layreffr .and. jp(icol,lay) .ge. layreffr) &
            laysolfr = lay
         ind0 = ((jp(icol,lay)-13)*5+(jt(icol,lay)-1))*nspb(16) + 1
         ind1 = ((jp(icol,lay)-12)*5+(jt1(icol,lay)-1))*nspb(16) + 1
         tauray = colmol(icol,lay) * rayl
         do ig = 1, ng16
            taug(icol,lay,ig) = colch4(icol,lay) * &
                (fac00(icol,lay) * absb(ind0  ,ig) + &
                 fac10(icol,lay) * absb(ind0+1,ig) + &
                 fac01(icol,lay) * absb(ind1  ,ig) + &
                 fac11(icol,lay) * absb(ind1+1,ig)) 
                        !            ssa(lay,ig) = tauray/taug(lay,ig)
            if (lay .eq. laysolfr) sfluxzen(icol,ig) = sfluxref(ig) 
            taur(icol,lay,ig) = tauray  
         enddo
      enddo
     end do
            END SUBROUTINE taumol16
            !----------------------------------------------------------------------------

            SUBROUTINE taumol17()
                !----------------------------------------------------------------------------
                !
                !     band 17:  3250-4000 cm-1 (low - h2o,co2; high - h2o,co2)
                !
                !----------------------------------------------------------------------------
                ! ------- Modules -------
                USE parrrsw, ONLY: ng17
                USE parrrsw, ONLY: ngs16
                USE rrsw_kg17, ONLY: strrat
                USE rrsw_kg17, ONLY: rayl
                USE rrsw_kg17, ONLY: absa
                USE rrsw_kg17, ONLY: selfref
                USE rrsw_kg17, ONLY: forref
                USE rrsw_kg17, ONLY: layreffr
                USE rrsw_kg17, ONLY: absb
                USE rrsw_kg17, ONLY: sfluxref
                ! ------- Declarations -------
                ! Local
                INTEGER :: lay
                INTEGER :: js
                INTEGER :: ind0
                INTEGER :: ind1
                INTEGER :: inds
                INTEGER :: indf
                INTEGER :: ig
                INTEGER :: laysolfr
                REAL(KIND=r8) :: speccomb
                REAL(KIND=r8) :: specparm
                REAL(KIND=r8) :: specmult
                REAL(KIND=r8) :: fs
                REAL(KIND=r8) :: fac000
                REAL(KIND=r8) :: fac010
                REAL(KIND=r8) :: fac100
                REAL(KIND=r8) :: fac110
                REAL(KIND=r8) :: fac001
                REAL(KIND=r8) :: fac011
                REAL(KIND=r8) :: fac101
                REAL(KIND=r8) :: fac111
                REAL(KIND=r8) :: tauray
                ! Compute the optical depth by interpolating in ln(pressure),
                ! temperature, and appropriate species.  Below LAYTROP, the water
                ! vapor self-continuum is interpolated (in temperature) separately.
                ! Lower atmosphere loop
     do icol=1,ncol 
      do lay = 1, laytrop(icol)
         speccomb = colh2o(icol,lay) + strrat*colco2(icol,lay)
         specparm = colh2o(icol,lay)/speccomb 
         if (specparm .ge. oneminus) specparm = oneminus
         specmult = 8._r8*(specparm)
         js = 1 + int(specmult)
         fs = mod(specmult, 1._r8 )
         fac000 = (1._r8 - fs) * fac00(icol,lay)
         fac010 = (1._r8 - fs) * fac10(icol,lay)
         fac100 = fs * fac00(icol,lay)
         fac110 = fs * fac10(icol,lay)
         fac001 = (1._r8 - fs) * fac01(icol,lay)
         fac011 = (1._r8 - fs) * fac11(icol,lay)
         fac101 = fs * fac01(icol,lay)
         fac111 = fs * fac11(icol,lay)
         ind0 = ((jp(icol,lay)-1)*5+(jt(icol,lay)-1))*nspa(17) + js
         ind1 = (jp(icol,lay)*5+(jt1(icol,lay)-1))*nspa(17) + js
         inds = indself(icol,lay)
         indf = indfor(icol,lay)
         tauray = colmol(icol,lay) * rayl
         do ig = 1, ng17
            taug(icol,lay,ngs16+ig) = speccomb * &
                (fac000 * absa(ind0,ig) + &
                 fac100 * absa(ind0+1,ig) + &
                 fac010 * absa(ind0+9,ig) + &
                 fac110 * absa(ind0+10,ig) + &
                 fac001 * absa(ind1,ig) + &
                 fac101 * absa(ind1+1,ig) + &
                 fac011 * absa(ind1+9,ig) + &
                 fac111 * absa(ind1+10,ig)) + &
                 colh2o(icol,lay) * &
                 (selffac(icol,lay) * (selfref(inds,ig) + &
                 selffrac(icol,lay) * &
                 (selfref(inds+1,ig) - selfref(inds,ig))) + &
                 forfac(icol,lay) * (forref(indf,ig) + &
                 forfrac(icol,lay) * &
                 (forref(indf+1,ig) - forref(indf,ig)))) 
                        !             ssa(lay,ngs16+ig) = tauray/taug(lay,ngs16+ig)
            taur(icol,lay,ngs16+ig) = tauray
         enddo
      enddo
      laysolfr = nlayers
                ! Upper atmosphere loop
      do lay = laytrop(icol)+1, nlayers
         if (jp(icol,lay-1) .lt. layreffr .and. jp(icol,lay) .ge. layreffr) &
            laysolfr = lay
         speccomb = colh2o(icol,lay) + strrat*colco2(icol,lay)
         specparm = colh2o(icol,lay)/speccomb 
         if (specparm .ge. oneminus) specparm = oneminus
         specmult = 4._r8*(specparm)
         js = 1 + int(specmult)
         fs = mod(specmult, 1._r8 )
         fac000 = (1._r8 - fs) * fac00(icol,lay)
         fac010 = (1._r8 - fs) * fac10(icol,lay)
         fac100 = fs * fac00(icol,lay)
         fac110 = fs * fac10(icol,lay)
         fac001 = (1._r8 - fs) * fac01(icol,lay)
         fac011 = (1._r8 - fs) * fac11(icol,lay)
         fac101 = fs * fac01(icol,lay)
         fac111 = fs * fac11(icol,lay)
         ind0 = ((jp(icol,lay)-13)*5+(jt(icol,lay)-1))*nspb(17) + js
         ind1 = ((jp(icol,lay)-12)*5+(jt1(icol,lay)-1))*nspb(17) + js
         indf = indfor(icol,lay)
         tauray = colmol(icol,lay) * rayl
         do ig = 1, ng17
            taug(icol,lay,ngs16+ig) = speccomb * &
                (fac000 * absb(ind0,ig) + &
                 fac100 * absb(ind0+1,ig) + &
                 fac010 * absb(ind0+5,ig) + &
                 fac110 * absb(ind0+6,ig) + &
                 fac001 * absb(ind1,ig) + &
                 fac101 * absb(ind1+1,ig) + &
                 fac011 * absb(ind1+5,ig) + &
                 fac111 * absb(ind1+6,ig)) + &
                 colh2o(icol,lay) * &
                 forfac(icol,lay) * (forref(indf,ig) + &
                 forfrac(icol,lay) * &
                 (forref(indf+1,ig) - forref(indf,ig))) 
                        !            ssa(lay,ngs16+ig) = tauray/taug(lay,ngs16+ig)
            if (lay .eq. laysolfr) sfluxzen(icol,ngs16+ig) = sfluxref(ig,js) &
               + fs * (sfluxref(ig,js+1) - sfluxref(ig,js))
            taur(icol,lay,ngs16+ig) = tauray
         enddo
      enddo
     enddo
            END SUBROUTINE taumol17
            !----------------------------------------------------------------------------

            SUBROUTINE taumol18()
                !----------------------------------------------------------------------------
                !
                !     band 18:  4000-4650 cm-1 (low - h2o,ch4; high - ch4)
                !
                !----------------------------------------------------------------------------
                ! ------- Modules -------
                USE parrrsw, ONLY: ng18
                USE parrrsw, ONLY: ngs17
                USE rrsw_kg18, ONLY: layreffr
                USE rrsw_kg18, ONLY: strrat
                USE rrsw_kg18, ONLY: rayl
                USE rrsw_kg18, ONLY: forref
                USE rrsw_kg18, ONLY: absa
                USE rrsw_kg18, ONLY: selfref
                USE rrsw_kg18, ONLY: sfluxref
                USE rrsw_kg18, ONLY: absb
                ! ------- Declarations -------
                ! Local
                INTEGER :: laysolfr
                INTEGER :: lay
                INTEGER :: js
                INTEGER :: ind0
                INTEGER :: ind1
                INTEGER :: inds
                INTEGER :: indf
                INTEGER :: ig
                REAL(KIND=r8) :: speccomb
                REAL(KIND=r8) :: specparm
                REAL(KIND=r8) :: specmult
                REAL(KIND=r8) :: fs
                REAL(KIND=r8) :: fac000
                REAL(KIND=r8) :: fac010
                REAL(KIND=r8) :: fac100
                REAL(KIND=r8) :: fac110
                REAL(KIND=r8) :: fac001
                REAL(KIND=r8) :: fac011
                REAL(KIND=r8) :: fac101
                REAL(KIND=r8) :: fac111
                REAL(KIND=r8) :: tauray
                ! Compute the optical depth by interpolating in ln(pressure),
                ! temperature, and appropriate species.  Below LAYTROP, the water
                ! vapor self-continuum is interpolated (in temperature) separately.
     do icol=1,ncol 
      laysolfr = laytrop(icol)
                ! Lower atmosphere loop
      do lay = 1, laytrop(icol)
         if (jp(icol,lay) .lt. layreffr .and. jp(icol,lay+1) .ge. layreffr) &
            laysolfr = min(lay+1,laytrop(icol))
         speccomb = colh2o(icol,lay) + strrat*colch4(icol,lay)
         specparm = colh2o(icol,lay)/speccomb 
         if (specparm .ge. oneminus) specparm = oneminus
         specmult = 8._r8*(specparm)
         js = 1 + int(specmult)
         fs = mod(specmult, 1._r8 )
         fac000 = (1._r8 - fs) * fac00(icol,lay)
         fac010 = (1._r8 - fs) * fac10(icol,lay)
         fac100 = fs * fac00(icol,lay)
         fac110 = fs * fac10(icol,lay)
         fac001 = (1._r8 - fs) * fac01(icol,lay)
         fac011 = (1._r8 - fs) * fac11(icol,lay)
         fac101 = fs * fac01(icol,lay)
         fac111 = fs * fac11(icol,lay)
         ind0 = ((jp(icol,lay)-1)*5+(jt(icol,lay)-1))*nspa(18) + js
         ind1 = (jp(icol,lay)*5+(jt1(icol,lay)-1))*nspa(18) + js
         inds = indself(icol,lay)
         indf = indfor(icol,lay)
         tauray = colmol(icol,lay) * rayl
         do ig = 1, ng18
            taug(icol,lay,ngs17+ig) = speccomb * &
                (fac000 * absa(ind0,ig) + &
                 fac100 * absa(ind0+1,ig) + &
                 fac010 * absa(ind0+9,ig) + &
                 fac110 * absa(ind0+10,ig) + &
                 fac001 * absa(ind1,ig) + &
                 fac101 * absa(ind1+1,ig) + &
                 fac011 * absa(ind1+9,ig) + &
                 fac111 * absa(ind1+10,ig)) + &
                 colh2o(icol,lay) * &
                 (selffac(icol,lay) * (selfref(inds,ig) + &
                 selffrac(icol,lay) * &
                 (selfref(inds+1,ig) - selfref(inds,ig))) + &
                 forfac(icol,lay) * (forref(indf,ig) + &
                 forfrac(icol,lay) * &
                 (forref(indf+1,ig) - forref(indf,ig)))) 
                        !            ssa(lay,ngs17+ig) = tauray/taug(lay,ngs17+ig)
            if (lay .eq. laysolfr) sfluxzen(icol,ngs17+ig) = sfluxref(ig,js) &
               + fs * (sfluxref(ig,js+1) - sfluxref(ig,js))
            taur(icol,lay,ngs17+ig) = tauray
         enddo
      enddo
                ! Upper atmosphere loop
      do lay = laytrop(icol)+1, nlayers
         ind0 = ((jp(icol,lay)-13)*5+(jt(icol,lay)-1))*nspb(18) + 1
         ind1 = ((jp(icol,lay)-12)*5+(jt1(icol,lay)-1))*nspb(18) + 1
         tauray = colmol(icol,lay) * rayl
         do ig = 1, ng18
            taug(icol,lay,ngs17+ig) = colch4(icol,lay) * &
                (fac00(icol,lay) * absb(ind0,ig) + &
                 fac10(icol,lay) * absb(ind0+1,ig) + &
                 fac01(icol,lay) * absb(ind1,ig) + &	  
                 fac11(icol,lay) * absb(ind1+1,ig)) 
                        !           ssa(lay,ngs17+ig) = tauray/taug(lay,ngs17+ig)
           taur(icol,lay,ngs17+ig) = tauray
         enddo
       enddo
      enddo
            END SUBROUTINE taumol18
            !----------------------------------------------------------------------------

            SUBROUTINE taumol19()
                !----------------------------------------------------------------------------
                !
                !     band 19:  4650-5150 cm-1 (low - h2o,co2; high - co2)
                !
                !----------------------------------------------------------------------------
                ! ------- Modules -------
                USE parrrsw, ONLY: ng19
                USE parrrsw, ONLY: ngs18
                USE rrsw_kg19, ONLY: layreffr
                USE rrsw_kg19, ONLY: strrat
                USE rrsw_kg19, ONLY: rayl
                USE rrsw_kg19, ONLY: selfref
                USE rrsw_kg19, ONLY: absa
                USE rrsw_kg19, ONLY: forref
                USE rrsw_kg19, ONLY: sfluxref
                USE rrsw_kg19, ONLY: absb
                ! ------- Declarations -------
                ! Local
                INTEGER :: laysolfr
                INTEGER :: lay
                INTEGER :: js
                INTEGER :: ind0
                INTEGER :: ind1
                INTEGER :: inds
                INTEGER :: indf
                INTEGER :: ig
                REAL(KIND=r8) :: speccomb
                REAL(KIND=r8) :: specparm
                REAL(KIND=r8) :: specmult
                REAL(KIND=r8) :: fs
                REAL(KIND=r8) :: fac000
                REAL(KIND=r8) :: fac010
                REAL(KIND=r8) :: fac100
                REAL(KIND=r8) :: fac110
                REAL(KIND=r8) :: fac001
                REAL(KIND=r8) :: fac011
                REAL(KIND=r8) :: fac101
                REAL(KIND=r8) :: fac111
                REAL(KIND=r8) :: tauray
                ! Compute the optical depth by interpolating in ln(pressure),
                ! temperature, and appropriate species.  Below LAYTROP, the water
                ! vapor self-continuum is interpolated (in temperature) separately.
     do icol=1,ncol 
     laysolfr = laytrop(icol)
                ! Lower atmosphere loop
      do lay = 1, laytrop(icol)
         if (jp(icol,lay) .lt. layreffr .and. jp(icol,lay+1) .ge. layreffr) &
            laysolfr = min(lay+1,laytrop(icol))
         speccomb = colh2o(icol,lay) + strrat*colco2(icol,lay)
         specparm = colh2o(icol,lay)/speccomb 
         if (specparm .ge. oneminus) specparm = oneminus
         specmult = 8._r8*(specparm)
         js = 1 + int(specmult)
         fs = mod(specmult, 1._r8 )
         fac000 = (1._r8 - fs) * fac00(icol,lay)
         fac010 = (1._r8 - fs) * fac10(icol,lay)
         fac100 = fs * fac00(icol,lay)
         fac110 = fs * fac10(icol,lay)
         fac001 = (1._r8 - fs) * fac01(icol,lay)
         fac011 = (1._r8 - fs) * fac11(icol,lay)
         fac101 = fs * fac01(icol,lay)
         fac111 = fs * fac11(icol,lay)
         ind0 = ((jp(icol,lay)-1)*5+(jt(icol,lay)-1))*nspa(19) + js
         ind1 = (jp(icol,lay)*5+(jt1(icol,lay)-1))*nspa(19) + js
         inds = indself(icol,lay)
         indf = indfor(icol,lay)
         tauray = colmol(icol,lay) * rayl
         do ig = 1 , ng19
            taug(icol,lay,ngs18+ig) = speccomb * &
                (fac000 * absa(ind0,ig) + &
                 fac100 * absa(ind0+1,ig) + &
                 fac010 * absa(ind0+9,ig) + &
                 fac110 * absa(ind0+10,ig) + &
                 fac001 * absa(ind1,ig) + &
                 fac101 * absa(ind1+1,ig) + &
                 fac011 * absa(ind1+9,ig) + &
                 fac111 * absa(ind1+10,ig)) + &
                 colh2o(icol,lay) * &
                 (selffac(icol,lay) * (selfref(inds,ig) + &
                 selffrac(icol,lay) * &
                 (selfref(inds+1,ig) - selfref(inds,ig))) + & 
                 forfac(icol,lay) * (forref(indf,ig) + &
                 forfrac(icol,lay) * &
                 (forref(indf+1,ig) - forref(indf,ig)))) 
                        !            ssa(lay,ngs18+ig) = tauray/taug(lay,ngs18+ig)
            if (lay .eq. laysolfr) sfluxzen(icol,ngs18+ig) = sfluxref(ig,js) &
               + fs * (sfluxref(ig,js+1) - sfluxref(ig,js))
            taur(icol,lay,ngs18+ig) = tauray   
         enddo
      enddo
                ! Upper atmosphere loop
      do lay = laytrop(icol)+1, nlayers
         ind0 = ((jp(icol,lay)-13)*5+(jt(icol,lay)-1))*nspb(19) + 1
         ind1 = ((jp(icol,lay)-12)*5+(jt1(icol,lay)-1))*nspb(19) + 1
         tauray = colmol(icol,lay) * rayl
         do ig = 1 , ng19
            taug(icol,lay,ngs18+ig) = colco2(icol,lay) * &
                (fac00(icol,lay) * absb(ind0,ig) + &
                 fac10(icol,lay) * absb(ind0+1,ig) + &
                 fac01(icol,lay) * absb(ind1,ig) + &
                 fac11(icol,lay) * absb(ind1+1,ig)) 
                        !            ssa(lay,ngs18+ig) = tauray/taug(lay,ngs18+ig)
            taur(icol,lay,ngs18+ig) = tauray   
         enddo
      enddo
    enddo
            END SUBROUTINE taumol19
            !----------------------------------------------------------------------------

            SUBROUTINE taumol20()
                !----------------------------------------------------------------------------
                !
                !     band 20:  5150-6150 cm-1 (low - h2o; high - h2o)
                !
                !----------------------------------------------------------------------------
                ! ------- Modules -------
                USE parrrsw, ONLY: ng20
                USE parrrsw, ONLY: ngs19
                USE rrsw_kg20, ONLY: layreffr
                USE rrsw_kg20, ONLY: rayl
                USE rrsw_kg20, ONLY: absch4
                USE rrsw_kg20, ONLY: forref
                USE rrsw_kg20, ONLY: absa
                USE rrsw_kg20, ONLY: selfref
                USE rrsw_kg20, ONLY: sfluxref
                USE rrsw_kg20, ONLY: absb
                IMPLICIT NONE
                ! ------- Declarations -------
                ! Local
                INTEGER :: laysolfr
                INTEGER :: lay
                INTEGER :: ind0
                INTEGER :: ind1
                INTEGER :: inds
                INTEGER :: indf
                INTEGER :: ig
                REAL(KIND=r8) :: tauray
                ! Compute the optical depth by interpolating in ln(pressure),
                ! temperature, and appropriate species.  Below LAYTROP, the water
                ! vapor self-continuum is interpolated (in temperature) separately.
     do icol=1,ncol
      laysolfr = laytrop(icol)
                ! Lower atmosphere loop
      do lay = 1, laytrop(icol)
         if (jp(icol,lay) .lt. layreffr .and. jp(icol,lay+1) .ge. layreffr) &
            laysolfr = min(lay+1,laytrop(icol))
         ind0 = ((jp(icol,lay)-1)*5+(jt(icol,lay)-1))*nspa(20) + 1
         ind1 = (jp(icol,lay)*5+(jt1(icol,lay)-1))*nspa(20) + 1
         inds = indself(icol,lay)
         indf = indfor(icol,lay)
         tauray = colmol(icol,lay) * rayl
         do ig = 1, ng20
            taug(icol,lay,ngs19+ig) = colh2o(icol,lay) * &
               ((fac00(icol,lay) * absa(ind0,ig) + &
                 fac10(icol,lay) * absa(ind0+1,ig) + &
                 fac01(icol,lay) * absa(ind1,ig) + &
                 fac11(icol,lay) * absa(ind1+1,ig)) + &
                 selffac(icol,lay) * (selfref(inds,ig) + & 
                 selffrac(icol,lay) * &
                 (selfref(inds+1,ig) - selfref(inds,ig))) + &
                 forfac(icol,lay) * (forref(indf,ig) + &
                 forfrac(icol,lay) * &
                 (forref(indf+1,ig) - forref(indf,ig)))) &
                 + colch4(icol,lay) * absch4(ig)
                        !            ssa(lay,ngs19+ig) = tauray/taug(lay,ngs19+ig)
            taur(icol,lay,ngs19+ig) = tauray 
            if (lay .eq. laysolfr) sfluxzen(icol,ngs19+ig) = sfluxref(ig) 
         enddo
      enddo
                ! Upper atmosphere loop
      do lay = laytrop(icol)+1, nlayers
         ind0 = ((jp(icol,lay)-13)*5+(jt(icol,lay)-1))*nspb(20) + 1
         ind1 = ((jp(icol,lay)-12)*5+(jt1(icol,lay)-1))*nspb(20) + 1
         indf = indfor(icol,lay)
         tauray = colmol(icol,lay) * rayl
         do ig = 1, ng20
            taug(icol,lay,ngs19+ig) = colh2o(icol,lay) * &
                (fac00(icol,lay) * absb(ind0,ig) + &
                 fac10(icol,lay) * absb(ind0+1,ig) + &
                 fac01(icol,lay) * absb(ind1,ig) + &
                 fac11(icol,lay) * absb(ind1+1,ig) + &
                 forfac(icol,lay) * (forref(indf,ig) + &
                 forfrac(icol,lay) * &
                 (forref(indf+1,ig) - forref(indf,ig)))) + &
                 colch4(icol,lay) * absch4(ig)
                        !            ssa(lay,ngs19+ig) = tauray/taug(lay,ngs19+ig)
            taur(icol,lay,ngs19+ig) = tauray 
         enddo
      enddo
     enddo
            END SUBROUTINE taumol20
            !----------------------------------------------------------------------------

            SUBROUTINE taumol21()
                !----------------------------------------------------------------------------
                !
                !     band 21:  6150-7700 cm-1 (low - h2o,co2; high - h2o,co2)
                !
                !----------------------------------------------------------------------------
                ! ------- Modules -------
                USE parrrsw, ONLY: ng21
                USE parrrsw, ONLY: ngs20
                USE rrsw_kg21, ONLY: layreffr
                USE rrsw_kg21, ONLY: strrat
                USE rrsw_kg21, ONLY: rayl
                USE rrsw_kg21, ONLY: forref
                USE rrsw_kg21, ONLY: absa
                USE rrsw_kg21, ONLY: selfref
                USE rrsw_kg21, ONLY: sfluxref
                USE rrsw_kg21, ONLY: absb
                ! ------- Declarations -------
                ! Local
                INTEGER :: laysolfr
                INTEGER :: lay
                INTEGER :: js
                INTEGER :: ind0
                INTEGER :: ind1
                INTEGER :: inds
                INTEGER :: indf
                INTEGER :: ig
                REAL(KIND=r8) :: speccomb
                REAL(KIND=r8) :: specparm
                REAL(KIND=r8) :: specmult
                REAL(KIND=r8) :: fs
                REAL(KIND=r8) :: fac000
                REAL(KIND=r8) :: fac010
                REAL(KIND=r8) :: fac100
                REAL(KIND=r8) :: fac110
                REAL(KIND=r8) :: fac001
                REAL(KIND=r8) :: fac011
                REAL(KIND=r8) :: fac101
                REAL(KIND=r8) :: fac111
                REAL(KIND=r8) :: tauray
                ! Compute the optical depth by interpolating in ln(pressure),
                ! temperature, and appropriate species.  Below LAYTROP, the water
                ! vapor self-continuum is interpolated (in temperature) separately.
     do icol=1,ncol
      laysolfr = laytrop(icol)
                ! Lower atmosphere loop
      do lay = 1, laytrop(icol)
         if (jp(icol,lay) .lt. layreffr .and. jp(icol,lay+1) .ge. layreffr) &
            laysolfr = min(lay+1,laytrop(icol))
         speccomb = colh2o(icol,lay) + strrat*colco2(icol,lay)
         specparm = colh2o(icol,lay)/speccomb 
         if (specparm .ge. oneminus) specparm = oneminus
         specmult = 8._r8*(specparm)
         js = 1 + int(specmult)
         fs = mod(specmult, 1._r8 )
         fac000 = (1._r8 - fs) * fac00(icol,lay)
         fac010 = (1._r8 - fs) * fac10(icol,lay)
         fac100 = fs * fac00(icol,lay)
         fac110 = fs * fac10(icol,lay)
         fac001 = (1._r8 - fs) * fac01(icol,lay)
         fac011 = (1._r8 - fs) * fac11(icol,lay)
         fac101 = fs * fac01(icol,lay)
         fac111 = fs * fac11(icol,lay)
         ind0 = ((jp(icol,lay)-1)*5+(jt(icol,lay)-1))*nspa(21) + js
         ind1 = (jp(icol,lay)*5+(jt1(icol,lay)-1))*nspa(21) + js
         inds = indself(icol,lay)
         indf = indfor(icol,lay)
         tauray = colmol(icol,lay) * rayl
         do ig = 1, ng21
            taug(icol,lay,ngs20+ig) = speccomb * &
                (fac000 * absa(ind0,ig) + &
                 fac100 * absa(ind0+1,ig) + &
                 fac010 * absa(ind0+9,ig) + &
                 fac110 * absa(ind0+10,ig) + &
                 fac001 * absa(ind1,ig) + &
                 fac101 * absa(ind1+1,ig) + &
                 fac011 * absa(ind1+9,ig) + &
                 fac111 * absa(ind1+10,ig)) + &
                 colh2o(icol,lay) * &
                 (selffac(icol,lay) * (selfref(inds,ig) + &
                 selffrac(icol,lay) * &
                 (selfref(inds+1,ig) - selfref(inds,ig))) + &
                 forfac(icol,lay) * (forref(indf,ig) + &
                 forfrac(icol,lay) * &
                 (forref(indf+1,ig) - forref(indf,ig))))
                        !            ssa(lay,ngs20+ig) = tauray/taug(lay,ngs20+ig)
            if (lay .eq. laysolfr) sfluxzen(icol,ngs20+ig) = sfluxref(ig,js) &
               + fs * (sfluxref(ig,js+1) - sfluxref(ig,js))
            taur(icol,lay,ngs20+ig) = tauray
         enddo
      enddo
                ! Upper atmosphere loop
      do lay = laytrop(icol)+1, nlayers
         speccomb = colh2o(icol,lay) + strrat*colco2(icol,lay)
         specparm = colh2o(icol,lay)/speccomb 
         if (specparm .ge. oneminus) specparm = oneminus
         specmult = 4._r8*(specparm)
         js = 1 + int(specmult)
         fs = mod(specmult, 1._r8 )
         fac000 = (1._r8 - fs) * fac00(icol,lay)
         fac010 = (1._r8 - fs) * fac10(icol,lay)
         fac100 = fs * fac00(icol,lay)
         fac110 = fs * fac10(icol,lay)
         fac001 = (1._r8 - fs) * fac01(icol,lay)
         fac011 = (1._r8 - fs) * fac11(icol,lay)
         fac101 = fs * fac01(icol,lay)
         fac111 = fs * fac11(icol,lay)
         ind0 = ((jp(icol,lay)-13)*5+(jt(icol,lay)-1))*nspb(21) + js
         ind1 = ((jp(icol,lay)-12)*5+(jt1(icol,lay)-1))*nspb(21) + js
         indf = indfor(icol,lay)
         tauray = colmol(icol,lay) * rayl
         do ig = 1, ng21
            taug(icol,lay,ngs20+ig) = speccomb * &
                (fac000 * absb(ind0,ig) + &
                 fac100 * absb(ind0+1,ig) + &
                 fac010 * absb(ind0+5,ig) + &
                 fac110 * absb(ind0+6,ig) + &
                 fac001 * absb(ind1,ig) + &
                 fac101 * absb(ind1+1,ig) + &
                 fac011 * absb(ind1+5,ig) + &
                 fac111 * absb(ind1+6,ig)) + &
                 colh2o(icol,lay) * &
                 forfac(icol,lay) * (forref(indf,ig) + &
                 forfrac(icol,lay) * &
                 (forref(indf+1,ig) - forref(indf,ig)))
                        !            ssa(lay,ngs20+ig) = tauray/taug(lay,ngs20+ig)
            taur(icol,lay,ngs20+ig) = tauray
         enddo
      enddo
    enddo
            END SUBROUTINE taumol21
            !----------------------------------------------------------------------------

            SUBROUTINE taumol22()
                !----------------------------------------------------------------------------
                !
                !     band 22:  7700-8050 cm-1 (low - h2o,o2; high - o2)
                !
                !----------------------------------------------------------------------------
                ! ------- Modules -------
                USE parrrsw, ONLY: ng22
                USE parrrsw, ONLY: ngs21
                USE rrsw_kg22, ONLY: layreffr
                USE rrsw_kg22, ONLY: strrat
                USE rrsw_kg22, ONLY: rayl
                USE rrsw_kg22, ONLY: forref
                USE rrsw_kg22, ONLY: absa
                USE rrsw_kg22, ONLY: selfref
                USE rrsw_kg22, ONLY: sfluxref
                USE rrsw_kg22, ONLY: absb
                ! ------- Declarations -------
                ! Local
                INTEGER :: laysolfr
                INTEGER :: lay
                INTEGER :: js
                INTEGER :: ind0
                INTEGER :: ind1
                INTEGER :: inds
                INTEGER :: indf
                INTEGER :: ig
                REAL(KIND=r8) :: o2adj
                REAL(KIND=r8) :: o2cont
                REAL(KIND=r8) :: speccomb
                REAL(KIND=r8) :: specparm
                REAL(KIND=r8) :: specmult
                REAL(KIND=r8) :: fs
                REAL(KIND=r8) :: fac000
                REAL(KIND=r8) :: fac010
                REAL(KIND=r8) :: fac100
                REAL(KIND=r8) :: fac110
                REAL(KIND=r8) :: fac001
                REAL(KIND=r8) :: fac011
                REAL(KIND=r8) :: fac101
                REAL(KIND=r8) :: fac111
                REAL(KIND=r8) :: tauray
                ! The following factor is the ratio of total O2 band intensity (lines
                ! and Mate continuum) to O2 band intensity (line only).  It is needed
                ! to adjust the optical depths since the k's include only lines.
      o2adj = 1.6_r8
                ! Compute the optical depth by interpolating in ln(pressure),
                ! temperature, and appropriate species.  Below LAYTROP, the water
                ! vapor self-continuum is interpolated (in temperature) separately.
     do icol=1,ncol
      laysolfr = laytrop(icol)
                ! Lower atmosphere loop
      do lay = 1, laytrop(icol)
         if (jp(icol,lay) .lt. layreffr .and. jp(icol,lay+1) .ge. layreffr) &
            laysolfr = min(lay+1,laytrop(icol))
         o2cont = 4.35e-4_r8*colo2(icol,lay)/(350.0_r8*2.0_r8)
         speccomb = colh2o(icol,lay) + o2adj*strrat*colo2(icol,lay)
         specparm = colh2o(icol,lay)/speccomb 
         if (specparm .ge. oneminus) specparm = oneminus
         specmult = 8._r8*(specparm)
                    !         odadj = specparm + o2adj * (1._r8 - specparm)
         js = 1 + int(specmult)
         fs = mod(specmult, 1._r8 )
         fac000 = (1._r8 - fs) * fac00(icol,lay)
         fac010 = (1._r8 - fs) * fac10(icol,lay)
         fac100 = fs * fac00(icol,lay)
         fac110 = fs * fac10(icol,lay)
         fac001 = (1._r8 - fs) * fac01(icol,lay)
         fac011 = (1._r8 - fs) * fac11(icol,lay)
         fac101 = fs * fac01(icol,lay)
         fac111 = fs * fac11(icol,lay)
         ind0 = ((jp(icol,lay)-1)*5+(jt(icol,lay)-1))*nspa(22) + js
         ind1 = (jp(icol,lay)*5+(jt1(icol,lay)-1))*nspa(22) + js
         inds = indself(icol,lay)
         indf = indfor(icol,lay)
         tauray = colmol(icol,lay) * rayl
         do ig = 1, ng22
            taug(icol,lay,ngs21+ig) = speccomb * &
                (fac000 * absa(ind0,ig) + &
                 fac100 * absa(ind0+1,ig) + &
                 fac010 * absa(ind0+9,ig) + &
                 fac110 * absa(ind0+10,ig) + &
                 fac001 * absa(ind1,ig) + &
                 fac101 * absa(ind1+1,ig) + &
                 fac011 * absa(ind1+9,ig) + &
                 fac111 * absa(ind1+10,ig)) + &
                 colh2o(icol,lay) * &
                 (selffac(icol,lay) * (selfref(inds,ig) + &
                 selffrac(icol,lay) * &
                  (selfref(inds+1,ig) - selfref(inds,ig))) + &
                 forfac(icol,lay) * (forref(indf,ig) + &
                 forfrac(icol,lay) * &
                 (forref(indf+1,ig) - forref(indf,ig)))) &
                 + o2cont
                        !            ssa(lay,ngs21+ig) = tauray/taug(lay,ngs21+ig)
            if (lay .eq. laysolfr) sfluxzen(icol,ngs21+ig) = sfluxref(ig,js) &
                + fs * (sfluxref(ig,js+1) - sfluxref(ig,js))
            taur(icol,lay,ngs21+ig) = tauray
         enddo
      enddo
                ! Upper atmosphere loop
      do lay = laytrop(icol)+1, nlayers
         o2cont = 4.35e-4_r8*colo2(icol,lay)/(350.0_r8*2.0_r8)
         ind0 = ((jp(icol,lay)-13)*5+(jt(icol,lay)-1))*nspb(22) + 1
         ind1 = ((jp(icol,lay)-12)*5+(jt1(icol,lay)-1))*nspb(22) + 1
         tauray = colmol(icol,lay) * rayl
         do ig = 1, ng22
            taug(icol,lay,ngs21+ig) = colo2(icol,lay) * o2adj * &
                (fac00(icol,lay) * absb(ind0,ig) + &
                 fac10(icol,lay) * absb(ind0+1,ig) + &
                 fac01(icol,lay) * absb(ind1,ig) + &
                 fac11(icol,lay) * absb(ind1+1,ig)) + &
                 o2cont
                        !            ssa(lay,ngs21+ig) = tauray/taug(lay,ngs21+ig)
            taur(icol,lay,ngs21+ig) = tauray
         enddo
      enddo
     enddo
            END SUBROUTINE taumol22
            !----------------------------------------------------------------------------

            SUBROUTINE taumol23()
                !----------------------------------------------------------------------------
                !
                !     band 23:  8050-12850 cm-1 (low - h2o; high - nothing)
                !
                !----------------------------------------------------------------------------
                ! ------- Modules -------
                USE parrrsw, ONLY: ng23
                USE parrrsw, ONLY: ngs22
                USE rrsw_kg23, ONLY: layreffr
                USE rrsw_kg23, ONLY: rayl
                USE rrsw_kg23, ONLY: absa
                USE rrsw_kg23, ONLY: givfac
                USE rrsw_kg23, ONLY: forref
                USE rrsw_kg23, ONLY: selfref
                USE rrsw_kg23, ONLY: sfluxref
                ! ------- Declarations -------
                ! Local
                INTEGER :: laysolfr
                INTEGER :: lay
                INTEGER :: ind0
                INTEGER :: ind1
                INTEGER :: inds
                INTEGER :: indf
                INTEGER :: ig
                REAL(KIND=r8) :: tauray
                ! Compute the optical depth by interpolating in ln(pressure),
                ! temperature, and appropriate species.  Below LAYTROP, the water
                ! vapor self-continuum is interpolated (in temperature) separately.
    do icol=1,ncol
      laysolfr = laytrop(icol)
                ! Lower atmosphere loop
      do lay = 1, laytrop(icol)
         if (jp(icol,lay) .lt. layreffr .and. jp(icol,lay+1) .ge. layreffr) &
            laysolfr = min(lay+1,laytrop(icol))
         ind0 = ((jp(icol,lay)-1)*5+(jt(icol,lay)-1))*nspa(23) + 1
         ind1 = (jp(icol,lay)*5+(jt1(icol,lay)-1))*nspa(23) + 1
         inds = indself(icol,lay)
         indf = indfor(icol,lay)
         do ig = 1, ng23
            tauray = colmol(icol,lay) * rayl(ig)
            taug(icol,lay,ngs22+ig) = colh2o(icol,lay) * &
                (givfac * (fac00(icol,lay) * absa(ind0,ig) + &
                 fac10(icol,lay) * absa(ind0+1,ig) + &
                 fac01(icol,lay) * absa(ind1,ig) + &
                 fac11(icol,lay) * absa(ind1+1,ig)) + &
                 selffac(icol,lay) * (selfref(inds,ig) + &
                 selffrac(icol,lay) * &
                 (selfref(inds+1,ig) - selfref(inds,ig))) + &
                 forfac(icol,lay) * (forref(indf,ig) + &
                 forfrac(icol,lay) * &
                 (forref(indf+1,ig) - forref(indf,ig)))) 
                        !            ssa(lay,ngs22+ig) = tauray/taug(lay,ngs22+ig)
            if (lay .eq. laysolfr) sfluxzen(icol,ngs22+ig) = sfluxref(ig) 
            taur(icol,lay,ngs22+ig) = tauray
         enddo
      enddo
                ! Upper atmosphere loop
      do lay = laytrop(icol)+1, nlayers
         do ig = 1, ng23
                        !            taug(lay,ngs22+ig) = colmol(icol,lay) * rayl(ig)
                        !            ssa(lay,ngs22+ig) = 1.0_r8
            taug(icol,lay,ngs22+ig) = 0._r8
            taur(icol,lay,ngs22+ig) = colmol(icol,lay) * rayl(ig) 
         enddo
      enddo
     enddo
            END SUBROUTINE taumol23
            !----------------------------------------------------------------------------

            SUBROUTINE taumol24()
                !----------------------------------------------------------------------------
                !
                !     band 24:  12850-16000 cm-1 (low - h2o,o2; high - o2)
                !
                !----------------------------------------------------------------------------
                ! ------- Modules -------
                USE parrrsw, ONLY: ng24
                USE parrrsw, ONLY: ngs23
                USE rrsw_kg24, ONLY: layreffr
                USE rrsw_kg24, ONLY: strrat
                USE rrsw_kg24, ONLY: rayla
                USE rrsw_kg24, ONLY: absa
                USE rrsw_kg24, ONLY: forref
                USE rrsw_kg24, ONLY: selfref
                USE rrsw_kg24, ONLY: abso3a
                USE rrsw_kg24, ONLY: sfluxref
                USE rrsw_kg24, ONLY: raylb
                USE rrsw_kg24, ONLY: absb
                USE rrsw_kg24, ONLY: abso3b
                ! ------- Declarations -------
                ! Local
                INTEGER :: laysolfr
                INTEGER :: lay
                INTEGER :: js
                INTEGER :: ind0
                INTEGER :: ind1
                INTEGER :: inds
                INTEGER :: indf
                INTEGER :: ig
                REAL(KIND=r8) :: speccomb
                REAL(KIND=r8) :: specparm
                REAL(KIND=r8) :: specmult
                REAL(KIND=r8) :: fs
                REAL(KIND=r8) :: fac000
                REAL(KIND=r8) :: fac010
                REAL(KIND=r8) :: fac100
                REAL(KIND=r8) :: fac110
                REAL(KIND=r8) :: fac001
                REAL(KIND=r8) :: fac011
                REAL(KIND=r8) :: fac101
                REAL(KIND=r8) :: fac111
                REAL(KIND=r8) :: tauray
                ! Compute the optical depth by interpolating in ln(pressure),
                ! temperature, and appropriate species.  Below LAYTROP, the water
                ! vapor self-continuum is interpolated (in temperature) separately.
    do icol=1,ncol
      laysolfr = laytrop(icol)
                ! Lower atmosphere loop
      do lay = 1, laytrop(icol)
         if (jp(icol,lay) .lt. layreffr .and. jp(icol,lay+1) .ge. layreffr) &
            laysolfr = min(lay+1,laytrop(icol))
         speccomb = colh2o(icol,lay) + strrat*colo2(icol,lay)
         specparm = colh2o(icol,lay)/speccomb 
         if (specparm .ge. oneminus) specparm = oneminus
         specmult = 8._r8*(specparm)
         js = 1 + int(specmult)
         fs = mod(specmult, 1._r8 )
         fac000 = (1._r8 - fs) * fac00(icol,lay)
         fac010 = (1._r8 - fs) * fac10(icol,lay)
         fac100 = fs * fac00(icol,lay)
         fac110 = fs * fac10(icol,lay)
         fac001 = (1._r8 - fs) * fac01(icol,lay)
         fac011 = (1._r8 - fs) * fac11(icol,lay)
         fac101 = fs * fac01(icol,lay)
         fac111 = fs * fac11(icol,lay)
         ind0 = ((jp(icol,lay)-1)*5+(jt(icol,lay)-1))*nspa(24) + js
         ind1 = (jp(icol,lay)*5+(jt1(icol,lay)-1))*nspa(24) + js
         inds = indself(icol,lay)
         indf = indfor(icol,lay)
         do ig = 1, ng24
            tauray = colmol(icol,lay) * (rayla(ig,js) + &
               fs * (rayla(ig,js+1) - rayla(ig,js)))
            taug(icol,lay,ngs23+ig) = speccomb * &
                (fac000 * absa(ind0,ig) + &
                 fac100 * absa(ind0+1,ig) + &
                 fac010 * absa(ind0+9,ig) + &
                 fac110 * absa(ind0+10,ig) + &
                 fac001 * absa(ind1,ig) + &
                 fac101 * absa(ind1+1,ig) + &
                 fac011 * absa(ind1+9,ig) + &
                 fac111 * absa(ind1+10,ig)) + &
                 colo3(icol,lay) * abso3a(ig) + &
                 colh2o(icol,lay) * & 
                 (selffac(icol,lay) * (selfref(inds,ig) + &
                 selffrac(icol,lay) * &
                 (selfref(inds+1,ig) - selfref(inds,ig))) + &
                 forfac(icol,lay) * (forref(indf,ig) + & 
                 forfrac(icol,lay) * &
                 (forref(indf+1,ig) - forref(indf,ig))))
                        !            ssa(lay,ngs23+ig) = tauray/taug(lay,ngs23+ig)
            if (lay .eq. laysolfr) sfluxzen(icol,ngs23+ig) = sfluxref(ig,js) &
               + fs * (sfluxref(ig,js+1) - sfluxref(ig,js))
            taur(icol,lay,ngs23+ig) = tauray
         enddo
      enddo
                ! Upper atmosphere loop
      do lay = laytrop(icol)+1, nlayers
         ind0 = ((jp(icol,lay)-13)*5+(jt(icol,lay)-1))*nspb(24) + 1
         ind1 = ((jp(icol,lay)-12)*5+(jt1(icol,lay)-1))*nspb(24) + 1
         do ig = 1, ng24
            tauray = colmol(icol,lay) * raylb(ig)
            taug(icol,lay,ngs23+ig) = colo2(icol,lay) * &
                (fac00(icol,lay) * absb(ind0,ig) + &
                 fac10(icol,lay) * absb(ind0+1,ig) + &
                 fac01(icol,lay) * absb(ind1,ig) + &
                 fac11(icol,lay) * absb(ind1+1,ig)) + &
                 colo3(icol,lay) * abso3b(ig)
                        !            ssa(lay,ngs23+ig) = tauray/taug(lay,ngs23+ig)
            taur(icol,lay,ngs23+ig) = tauray
         enddo
      enddo
     enddo
            END SUBROUTINE taumol24
            !----------------------------------------------------------------------------

            SUBROUTINE taumol25()
                !----------------------------------------------------------------------------
                !
                !     band 25:  16000-22650 cm-1 (low - h2o; high - nothing)
                !
                !----------------------------------------------------------------------------
                ! ------- Modules -------
                USE parrrsw, ONLY: ng25
                USE parrrsw, ONLY: ngs24
                USE rrsw_kg25, ONLY: layreffr
                USE rrsw_kg25, ONLY: rayl
                USE rrsw_kg25, ONLY: abso3a
                USE rrsw_kg25, ONLY: absa
                USE rrsw_kg25, ONLY: sfluxref
                USE rrsw_kg25, ONLY: abso3b
                ! ------- Declarations -------
                ! Local
                INTEGER :: laysolfr
                INTEGER :: lay
                INTEGER :: ind0
                INTEGER :: ind1
                INTEGER :: ig
                REAL(KIND=r8) :: tauray
                ! Compute the optical depth by interpolating in ln(pressure),
                ! temperature, and appropriate species.  Below LAYTROP, the water
                ! vapor self-continuum is interpolated (in temperature) separately.
    do icol=1,ncol
      laysolfr = laytrop(icol)
                ! Lower atmosphere loop
      do lay = 1, laytrop(icol)
         if (jp(icol,lay) .lt. layreffr .and. jp(icol,lay+1) .ge. layreffr) &
            laysolfr = min(lay+1,laytrop(icol))
         ind0 = ((jp(icol,lay)-1)*5+(jt(icol,lay)-1))*nspa(25) + 1
         ind1 = (jp(icol,lay)*5+(jt1(icol,lay)-1))*nspa(25) + 1
         do ig = 1, ng25
            tauray = colmol(icol,lay) * rayl(ig)
            taug(icol,lay,ngs24+ig) = colh2o(icol,lay) * &
                (fac00(icol,lay) * absa(ind0,ig) + &
                 fac10(icol,lay) * absa(ind0+1,ig) + &
                 fac01(icol,lay) * absa(ind1,ig) + &
                 fac11(icol,lay) * absa(ind1+1,ig)) + &
                 colo3(icol,lay) * abso3a(ig) 
                        !            ssa(lay,ngs24+ig) = tauray/taug(lay,ngs24+ig)
            if (lay .eq. laysolfr) sfluxzen(icol,ngs24+ig) = sfluxref(ig) 
            taur(icol,lay,ngs24+ig) = tauray
         enddo
      enddo
                ! Upper atmosphere loop
      do lay = laytrop(icol)+1, nlayers
         do ig = 1, ng25
            tauray = colmol(icol,lay) * rayl(ig)
            taug(icol,lay,ngs24+ig) = colo3(icol,lay) * abso3b(ig) 
                        !            ssa(lay,ngs24+ig) = tauray/taug(lay,ngs24+ig)
            taur(icol,lay,ngs24+ig) = tauray
         enddo
      enddo
     enddo
            END SUBROUTINE taumol25
            !----------------------------------------------------------------------------

            SUBROUTINE taumol26()
                !----------------------------------------------------------------------------
                !
                !     band 26:  22650-29000 cm-1 (low - nothing; high - nothing)
                !
                !----------------------------------------------------------------------------
                ! ------- Modules -------
                USE parrrsw, ONLY: ng26
                USE parrrsw, ONLY: ngs25
                USE rrsw_kg26, ONLY: sfluxref
                USE rrsw_kg26, ONLY: rayl
                ! ------- Declarations -------
                ! Local
                INTEGER :: laysolfr
                INTEGER :: lay
                INTEGER :: ig
                ! Compute the optical depth by interpolating in ln(pressure),
                ! temperature, and appropriate species.  Below LAYTROP, the water
                ! vapor self-continuum is interpolated (in temperature) separately.
     do icol=1,ncol
      laysolfr = laytrop(icol)
                ! Lower atmosphere loop
      do lay = 1, laytrop(icol)
         do ig = 1, ng26 
                        !            taug(lay,ngs25+ig) = colmol(icol,lay) * rayl(ig)
                        !            ssa(lay,ngs25+ig) = 1.0_r8
            if (lay .eq. laysolfr) sfluxzen(icol,ngs25+ig) = sfluxref(ig) 
            taug(icol,lay,ngs25+ig) = 0._r8
            taur(icol,lay,ngs25+ig) = colmol(icol,lay) * rayl(ig) 
         enddo
      enddo
                ! Upper atmosphere loop
      do lay = laytrop(icol)+1, nlayers
         do ig = 1, ng26
                        !            taug(lay,ngs25+ig) = colmol(icol,lay) * rayl(ig)
                        !            ssa(lay,ngs25+ig) = 1.0_r8
            taug(icol,lay,ngs25+ig) = 0._r8
            taur(icol,lay,ngs25+ig) = colmol(icol,lay) * rayl(ig) 
         enddo
      enddo
     enddo
            END SUBROUTINE taumol26
            !----------------------------------------------------------------------------

            SUBROUTINE taumol27()
                !----------------------------------------------------------------------------
                !
                !     band 27:  29000-38000 cm-1 (low - o3; high - o3)
                !
                !----------------------------------------------------------------------------
                ! ------- Modules -------
                USE parrrsw, ONLY: ng27
                USE parrrsw, ONLY: ngs26
                USE rrsw_kg27, ONLY: rayl
                USE rrsw_kg27, ONLY: absa
                USE rrsw_kg27, ONLY: layreffr
                USE rrsw_kg27, ONLY: absb
                USE rrsw_kg27, ONLY: scalekur
                USE rrsw_kg27, ONLY: sfluxref
                ! ------- Declarations -------
                ! Local
                INTEGER :: lay
                INTEGER :: ind0
                INTEGER :: ind1
                INTEGER :: ig
                INTEGER :: laysolfr
                REAL(KIND=r8) :: tauray
                ! Compute the optical depth by interpolating in ln(pressure),
                ! temperature, and appropriate species.  Below LAYTROP, the water
                ! vapor self-continuum is interpolated (in temperature) separately.
                ! Lower atmosphere loop
      do icol=1,ncol
       do lay = 1, laytrop(icol)
         ind0 = ((jp(icol,lay)-1)*5+(jt(icol,lay)-1))*nspa(27) + 1
         ind1 = (jp(icol,lay)*5+(jt1(icol,lay)-1))*nspa(27) + 1
         do ig = 1, ng27
            tauray = colmol(icol,lay) * rayl(ig)
            taug(icol,lay,ngs26+ig) = colo3(icol,lay) * &
                (fac00(icol,lay) * absa(ind0,ig) + &
                 fac10(icol,lay) * absa(ind0+1,ig) + &
                 fac01(icol,lay) * absa(ind1,ig) + &
                 fac11(icol,lay) * absa(ind1+1,ig))
                        !            ssa(lay,ngs26+ig) = tauray/taug(lay,ngs26+ig)
            taur(icol,lay,ngs26+ig) = tauray
         enddo
      enddo
      laysolfr = nlayers
                ! Upper atmosphere loop
      do lay = laytrop(icol)+1, nlayers
         if (jp(icol,lay-1) .lt. layreffr .and. jp(icol,lay) .ge. layreffr) &
            laysolfr = lay
         ind0 = ((jp(icol,lay)-13)*5+(jt(icol,lay)-1))*nspb(27) + 1
         ind1 = ((jp(icol,lay)-12)*5+(jt1(icol,lay)-1))*nspb(27) + 1
         do ig = 1, ng27
            tauray = colmol(icol,lay) * rayl(ig)
            taug(icol,lay,ngs26+ig) = colo3(icol,lay) * &
                (fac00(icol,lay) * absb(ind0,ig) + &
                 fac10(icol,lay) * absb(ind0+1,ig) + &
                 fac01(icol,lay) * absb(ind1,ig) + & 
                 fac11(icol,lay) * absb(ind1+1,ig))
                        !            ssa(lay,ngs26+ig) = tauray/taug(lay,ngs26+ig)
            if (lay.eq.laysolfr) sfluxzen(icol,ngs26+ig) = scalekur * sfluxref(ig) 
            taur(icol,lay,ngs26+ig) = tauray
         enddo
      enddo
     enddo
            END SUBROUTINE taumol27
            !----------------------------------------------------------------------------

            SUBROUTINE taumol28()
                !----------------------------------------------------------------------------
                !
                !     band 28:  38000-50000 cm-1 (low - o3,o2; high - o3,o2)
                !
                !----------------------------------------------------------------------------
                ! ------- Modules -------
                USE parrrsw, ONLY: ng28
                USE parrrsw, ONLY: ngs27
                USE rrsw_kg28, ONLY: strrat
                USE rrsw_kg28, ONLY: rayl
                USE rrsw_kg28, ONLY: absa
                USE rrsw_kg28, ONLY: layreffr
                USE rrsw_kg28, ONLY: absb
                USE rrsw_kg28, ONLY: sfluxref
                ! ------- Declarations -------
                ! Local
                INTEGER :: lay
                INTEGER :: js
                INTEGER :: ind0
                INTEGER :: ind1
                INTEGER :: ig
                INTEGER :: laysolfr
                REAL(KIND=r8) :: speccomb
                REAL(KIND=r8) :: specparm
                REAL(KIND=r8) :: specmult
                REAL(KIND=r8) :: fs
                REAL(KIND=r8) :: fac000
                REAL(KIND=r8) :: fac010
                REAL(KIND=r8) :: fac100
                REAL(KIND=r8) :: fac110
                REAL(KIND=r8) :: fac001
                REAL(KIND=r8) :: fac011
                REAL(KIND=r8) :: fac101
                REAL(KIND=r8) :: fac111
                REAL(KIND=r8) :: tauray
                ! Compute the optical depth by interpolating in ln(pressure),
                ! temperature, and appropriate species.  Below LAYTROP, the water
                ! vapor self-continuum is interpolated (in temperature) separately.
                ! Lower atmosphere loop
      do icol=1,ncol
        do lay = 1, laytrop(icol)
         speccomb = colo3(icol,lay) + strrat*colo2(icol,lay)
         specparm = colo3(icol,lay)/speccomb 
         if (specparm .ge. oneminus) specparm = oneminus
         specmult = 8._r8*(specparm)
         js = 1 + int(specmult)
         fs = mod(specmult, 1._r8 )
         fac000 = (1._r8 - fs) * fac00(icol,lay)
         fac010 = (1._r8 - fs) * fac10(icol,lay)
         fac100 = fs * fac00(icol,lay)
         fac110 = fs * fac10(icol,lay)
         fac001 = (1._r8 - fs) * fac01(icol,lay)
         fac011 = (1._r8 - fs) * fac11(icol,lay)
         fac101 = fs * fac01(icol,lay)
         fac111 = fs * fac11(icol,lay)
         ind0 = ((jp(icol,lay)-1)*5+(jt(icol,lay)-1))*nspa(28) + js
         ind1 = (jp(icol,lay)*5+(jt1(icol,lay)-1))*nspa(28) + js
         tauray = colmol(icol,lay) * rayl
         do ig = 1, ng28
            taug(icol,lay,ngs27+ig) = speccomb * &
                (fac000 * absa(ind0,ig) + &
                 fac100 * absa(ind0+1,ig) + &
                 fac010 * absa(ind0+9,ig) + &
                 fac110 * absa(ind0+10,ig) + &
                 fac001 * absa(ind1,ig) + &
                 fac101 * absa(ind1+1,ig) + &
                 fac011 * absa(ind1+9,ig) + &
                 fac111 * absa(ind1+10,ig)) 
                        !            ssa(lay,ngs27+ig) = tauray/taug(lay,ngs27+ig)
            taur(icol,lay,ngs27+ig) = tauray
         enddo
      enddo
      laysolfr = nlayers
                ! Upper atmosphere loop
      do lay = laytrop(icol)+1, nlayers
         if (jp(icol,lay-1) .lt. layreffr .and. jp(icol,lay) .ge. layreffr) &
            laysolfr = lay
         speccomb = colo3(icol,lay) + strrat*colo2(icol,lay)
         specparm = colo3(icol,lay)/speccomb 
         if (specparm .ge. oneminus) specparm = oneminus
         specmult = 4._r8*(specparm)
         js = 1 + int(specmult)
         fs = mod(specmult, 1._r8 )
         fac000 = (1._r8 - fs) * fac00(icol,lay)
         fac010 = (1._r8 - fs) * fac10(icol,lay)
         fac100 = fs * fac00(icol,lay)
         fac110 = fs * fac10(icol,lay)
         fac001 = (1._r8 - fs) * fac01(icol,lay)
         fac011 = (1._r8 - fs) * fac11(icol,lay)
         fac101 = fs * fac01(icol,lay)
         fac111 = fs * fac11(icol,lay)
         ind0 = ((jp(icol,lay)-13)*5+(jt(icol,lay)-1))*nspb(28) + js
         ind1 = ((jp(icol,lay)-12)*5+(jt1(icol,lay)-1))*nspb(28) + js
         tauray = colmol(icol,lay) * rayl
         do ig = 1, ng28
            taug(icol,lay,ngs27+ig) = speccomb * &
                (fac000 * absb(ind0,ig) + &
                 fac100 * absb(ind0+1,ig) + &
                 fac010 * absb(ind0+5,ig) + &
                 fac110 * absb(ind0+6,ig) + &
                 fac001 * absb(ind1,ig) + &
                 fac101 * absb(ind1+1,ig) + &
                 fac011 * absb(ind1+5,ig) + &
                 fac111 * absb(ind1+6,ig)) 
                        !            ssa(lay,ngs27+ig) = tauray/taug(lay,ngs27+ig)
            if (lay .eq. laysolfr) sfluxzen(icol,ngs27+ig) = sfluxref(ig,js) &
               + fs * (sfluxref(ig,js+1) - sfluxref(ig,js))
            taur(icol,lay,ngs27+ig) = tauray
         enddo
      enddo
    enddo
            END SUBROUTINE taumol28
            !----------------------------------------------------------------------------

            SUBROUTINE taumol29()
                !----------------------------------------------------------------------------
                !
                !     band 29:  820-2600 cm-1 (low - h2o; high - co2)
                !
                !----------------------------------------------------------------------------
                ! ------- Modules -------
                USE parrrsw, ONLY: ng29
                USE parrrsw, ONLY: ngs28
                USE rrsw_kg29, ONLY: rayl
                USE rrsw_kg29, ONLY: forref
                USE rrsw_kg29, ONLY: absa
                USE rrsw_kg29, ONLY: absco2
                USE rrsw_kg29, ONLY: selfref
                USE rrsw_kg29, ONLY: layreffr
                USE rrsw_kg29, ONLY: absh2o
                USE rrsw_kg29, ONLY: absb
                USE rrsw_kg29, ONLY: sfluxref
                ! ------- Declarations -------
                ! Local
                INTEGER :: lay
                INTEGER :: ind0
                INTEGER :: ind1
                INTEGER :: inds
                INTEGER :: indf
                INTEGER :: ig
                INTEGER :: laysolfr
                REAL(KIND=r8) :: tauray
                ! Compute the optical depth by interpolating in ln(pressure),
                ! temperature, and appropriate species.  Below LAYTROP, the water
                ! vapor self-continuum is interpolated (in temperature) separately.
                ! Lower atmosphere loop
      do icol=1,ncol
       do lay = 1, laytrop(icol)
         ind0 = ((jp(icol,lay)-1)*5+(jt(icol,lay)-1))*nspa(29) + 1
         ind1 = (jp(icol,lay)*5+(jt1(icol,lay)-1))*nspa(29) + 1
         inds = indself(icol,lay)
         indf = indfor(icol,lay)
         tauray = colmol(icol,lay) * rayl
         do ig = 1, ng29
            taug(icol,lay,ngs28+ig) = colh2o(icol,lay) * &
               ((fac00(icol,lay) * absa(ind0,ig) + &
                 fac10(icol,lay) * absa(ind0+1,ig) + &
                 fac01(icol,lay) * absa(ind1,ig) + &
                 fac11(icol,lay) * absa(ind1+1,ig)) + &
                 selffac(icol,lay) * (selfref(inds,ig) + &
                 selffrac(icol,lay) * &
                 (selfref(inds+1,ig) - selfref(inds,ig))) + &
                 forfac(icol,lay) * (forref(indf,ig) + & 
                 forfrac(icol,lay) * &
                 (forref(indf+1,ig) - forref(indf,ig)))) &
                 + colco2(icol,lay) * absco2(ig) 
                        !            ssa(lay,ngs28+ig) = tauray/taug(lay,ngs28+ig)
            taur(icol,lay,ngs28+ig) = tauray
         enddo
      enddo
      laysolfr = nlayers
                ! Upper atmosphere loop
      do lay = laytrop(icol)+1, nlayers
         if (jp(icol,lay-1) .lt. layreffr .and. jp(icol,lay) .ge. layreffr) &
            laysolfr = lay
         ind0 = ((jp(icol,lay)-13)*5+(jt(icol,lay)-1))*nspb(29) + 1
         ind1 = ((jp(icol,lay)-12)*5+(jt1(icol,lay)-1))*nspb(29) + 1
         tauray = colmol(icol,lay) * rayl
         do ig = 1, ng29
            taug(icol,lay,ngs28+ig) = colco2(icol,lay) * &
                (fac00(icol,lay) * absb(ind0,ig) + &
                 fac10(icol,lay) * absb(ind0+1,ig) + &
                 fac01(icol,lay) * absb(ind1,ig) + &
                 fac11(icol,lay) * absb(ind1+1,ig)) &  
                 + colh2o(icol,lay) * absh2o(ig) 
                        !            ssa(lay,ngs28+ig) = tauray/taug(lay,ngs28+ig)
            if (lay .eq. laysolfr) sfluxzen(icol,ngs28+ig) = sfluxref(ig) 
            taur(icol,lay,ngs28+ig) = tauray
         enddo
      enddo
     enddo
            END SUBROUTINE taumol29
        END SUBROUTINE taumol_sw
    END MODULE rrtmg_sw_taumol
