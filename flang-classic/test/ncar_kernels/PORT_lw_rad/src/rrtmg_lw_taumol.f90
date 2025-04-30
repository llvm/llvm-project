
! KGEN-generated Fortran source file
!
! Filename    : rrtmg_lw_taumol.f90
! Generated at: 2015-07-06 23:28:44
! KGEN version: 0.4.13



    MODULE rrtmg_lw_taumol
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        !  --------------------------------------------------------------------------
        ! |                                                                          |
        ! |  Copyright 2002-2009, Atmospheric & Environmental Research, Inc. (AER).  |
        ! |  This software may be used, copied, or redistributed as long as it is    |
        ! |  not sold and this copyright notice is reproduced on each copy made.     |
        ! |  This model is provided as is without any express or implied warranties. |
        ! |                       (http://www.rtweb.aer.com/)                        |
        ! |                                                                          |
        !  --------------------------------------------------------------------------
        ! ------- Modules -------
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !      use parkind, only : im => kind_im, rb => kind_r8
        USE rrlw_con, ONLY: oneminus
        USE rrlw_wvn, ONLY: nspa
        USE rrlw_wvn, ONLY: nspb
        USE rrlw_vsn, ONLY: hvrtau
        IMPLICIT NONE
        CONTAINS

        ! write subroutines
        ! No subroutines
        ! No module extern variables
        !----------------------------------------------------------------------------

        SUBROUTINE taumol(nlayers, pavel, wx, coldry, laytrop, jp, jt, jt1, planklay, planklev, plankbnd, colh2o, colco2, colo3, &
        coln2o, colco, colch4, colo2, colbrd, fac00, fac01, fac10, fac11, rat_h2oco2, rat_h2oco2_1, rat_h2oo3, rat_h2oo3_1, &
        rat_h2on2o, rat_h2on2o_1, rat_h2och4, rat_h2och4_1, rat_n2oco2, rat_n2oco2_1, rat_o3co2, rat_o3co2_1, selffac, selffrac, &
        indself, forfac, forfrac, indfor, minorfrac, scaleminor, scaleminorn2, indminor, fracs, taug)
            !----------------------------------------------------------------------------
            ! *******************************************************************************
            ! *                                                                             *
            ! *                  Optical depths developed for the                           *
            ! *                                                                             *
            ! *                RAPID RADIATIVE TRANSFER MODEL (RRTM)                        *
            ! *                                                                             *
            ! *                                                                             *
            ! *            ATMOSPHERIC AND ENVIRONMENTAL RESEARCH, INC.                     *
            ! *                        131 HARTWELL AVENUE                                  *
            ! *                        LEXINGTON, MA 02421                                  *
            ! *                                                                             *
            ! *                                                                             *
            ! *                           ELI J. MLAWER                                     *
            ! *                         JENNIFER DELAMERE                                   *
            ! *                         STEVEN J. TAUBMAN                                   *
            ! *                         SHEPARD A. CLOUGH                                   *
            ! *                                                                             *
            ! *                                                                             *
            ! *                                                                             *
            ! *                                                                             *
            ! *                       email:  mlawer@aer.com                                *
            ! *                       email:  jdelamer@aer.com                              *
            ! *                                                                             *
            ! *        The authors wish to acknowledge the contributions of the             *
            ! *        following people:  Karen Cady-Pereira, Patrick D. Brown,             *
            ! *        Michael J. Iacono, Ronald E. Farren, Luke Chen, Robert Bergstrom.    *
            ! *                                                                             *
            ! *******************************************************************************
            ! *                                                                             *
            ! *  Revision for g-point reduction: Michael J. Iacono, AER, Inc.               *
            ! *                                                                             *
            ! *******************************************************************************
            ! *     TAUMOL                                                                  *
            ! *                                                                             *
            ! *     This file contains the subroutines TAUGBn (where n goes from            *
            ! *     1 to 16).  TAUGBn calculates the optical depths and Planck fractions    *
            ! *     per g-value and layer for band n.                                       *
            ! *                                                                             *
            ! *  Output:  optical depths (unitless)                                         *
            ! *           fractions needed to compute Planck functions at every layer       *
            ! *               and g-value                                                   *
            ! *                                                                             *
            ! *     COMMON /TAUGCOM/  TAUG(MXLAY,MG)                                        *
            ! *     COMMON /PLANKG/   FRACS(MXLAY,MG)                                       *
            ! *                                                                             *
            ! *  Input                                                                      *
            ! *                                                                             *
            ! *     COMMON /FEATURES/ NG(NBANDS),NSPA(NBANDS),NSPB(NBANDS)                  *
            ! *     COMMON /PRECISE/  ONEMINUS                                              *
            ! *     COMMON /PROFILE/  NLAYERS,PAVEL(MXLAY),TAVEL(MXLAY),                    *
            ! *     &                 PZ(0:MXLAY),TZ(0:MXLAY)                               *
            ! *     COMMON /PROFDATA/ LAYTROP,                                              *
            ! *    &                  COLH2O(MXLAY),COLCO2(MXLAY),COLO3(MXLAY),             *
            ! *    &                  COLN2O(MXLAY),COLCO(MXLAY),COLCH4(MXLAY),             *
            ! *    &                  COLO2(MXLAY)
            ! *     COMMON /INTFAC/   FAC00(MXLAY),FAC01(MXLAY),                            *
            ! *    &                  FAC10(MXLAY),FAC11(MXLAY)                             *
            ! *     COMMON /INTIND/   JP(MXLAY),JT(MXLAY),JT1(MXLAY)                        *
            ! *     COMMON /SELF/     SELFFAC(MXLAY), SELFFRAC(MXLAY), INDSELF(MXLAY)       *
            ! *                                                                             *
            ! *     Description:                                                            *
            ! *     NG(IBAND) - number of g-values in band IBAND                            *
            ! *     NSPA(IBAND) - for the lower atmosphere, the number of reference         *
            ! *                   atmospheres that are stored for band IBAND per            *
            ! *                   pressure level and temperature.  Each of these            *
            ! *                   atmospheres has different relative amounts of the         *
            ! *                   key species for the band (i.e. different binary           *
            ! *                   species parameters).                                      *
            ! *     NSPB(IBAND) - same for upper atmosphere                                 *
            ! *     ONEMINUS - since problems are caused in some cases by interpolation     *
            ! *                parameters equal to or greater than 1, for these cases       *
            ! *                these parameters are set to this value, slightly < 1.        *
            ! *     PAVEL - layer pressures (mb)                                            *
            ! *     TAVEL - layer temperatures (degrees K)                                  *
            ! *     PZ - level pressures (mb)                                               *
            ! *     TZ - level temperatures (degrees K)                                     *
            ! *     LAYTROP - layer at which switch is made from one combination of         *
            ! *               key species to another                                        *
            ! *     COLH2O, COLCO2, COLO3, COLN2O, COLCH4 - column amounts of water         *
            ! *               vapor,carbon dioxide, ozone, nitrous ozide, methane,          *
            ! *               respectively (molecules/cm**2)                                *
            ! *     FACij(LAY) - for layer LAY, these are factors that are needed to        *
            ! *                  compute the interpolation factors that multiply the        *
            ! *                  appropriate reference k-values.  A value of 0 (1) for      *
            ! *                  i,j indicates that the corresponding factor multiplies     *
            ! *                  reference k-value for the lower (higher) of the two        *
            ! *                  appropriate temperatures, and altitudes, respectively.     *
            ! *     JP - the index of the lower (in altitude) of the two appropriate        *
            ! *          reference pressure levels needed for interpolation                 *
            ! *     JT, JT1 - the indices of the lower of the two appropriate reference     *
            ! *               temperatures needed for interpolation (for pressure           *
            ! *               levels JP and JP+1, respectively)                             *
            ! *     SELFFAC - scale factor needed for water vapor self-continuum, equals    *
            ! *               (water vapor density)/(atmospheric density at 296K and        *
            ! *               1013 mb)                                                      *
            ! *     SELFFRAC - factor needed for temperature interpolation of reference     *
            ! *                water vapor self-continuum data                              *
            ! *     INDSELF - index of the lower of the two appropriate reference           *
            ! *               temperatures needed for the self-continuum interpolation      *
            ! *     FORFAC  - scale factor needed for water vapor foreign-continuum.        *
            ! *     FORFRAC - factor needed for temperature interpolation of reference      *
            ! *                water vapor foreign-continuum data                           *
            ! *     INDFOR  - index of the lower of the two appropriate reference           *
            ! *               temperatures needed for the foreign-continuum interpolation   *
            ! *                                                                             *
            ! *  Data input                                                                 *
            ! *     COMMON /Kn/ KA(NSPA(n),5,13,MG), KB(NSPB(n),5,13:59,MG), SELFREF(10,MG),*
            ! *                 FORREF(4,MG), KA_M'MGAS', KB_M'MGAS'                        *
            ! *        (note:  n is the band number,'MGAS' is the species name of the minor *
            ! *         gas)                                                                *
            ! *                                                                             *
            ! *     Description:                                                            *
            ! *     KA - k-values for low reference atmospheres (key-species only)          *
            ! *          (units: cm**2/molecule)                                            *
            ! *     KB - k-values for high reference atmospheres (key-species only)         *
            ! *          (units: cm**2/molecule)                                            *
            ! *     KA_M'MGAS' - k-values for low reference atmosphere minor species        *
            ! *          (units: cm**2/molecule)                                            *
            ! *     KB_M'MGAS' - k-values for high reference atmosphere minor species       *
            ! *          (units: cm**2/molecule)                                            *
            ! *     SELFREF - k-values for water vapor self-continuum for reference         *
            ! *               atmospheres (used below LAYTROP)                              *
            ! *               (units: cm**2/molecule)                                       *
            ! *     FORREF  - k-values for water vapor foreign-continuum for reference      *
            ! *               atmospheres (used below/above LAYTROP)                        *
            ! *               (units: cm**2/molecule)                                       *
            ! *                                                                             *
            ! *     DIMENSION ABSA(65*NSPA(n),MG), ABSB(235*NSPB(n),MG)                     *
            ! *     EQUIVALENCE (KA,ABSA),(KB,ABSB)                                         *
            ! *                                                                             *
            !*******************************************************************************
            ! ------- Declarations -------
            ! ----- Input -----
            INTEGER, intent(in) :: nlayers ! total number of layers
            REAL(KIND=r8), intent(in) :: pavel(:) ! layer pressures (mb)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: wx(:,:) ! cross-section amounts (mol/cm2)
            !    Dimensions: (maxxsec,nlayers)
            REAL(KIND=r8), intent(in) :: coldry(:) ! column amount (dry air)
            !    Dimensions: (nlayers)
            INTEGER, intent(in) :: laytrop ! tropopause layer index
            INTEGER, intent(in) :: jp(:) !
            !    Dimensions: (nlayers)
            INTEGER, intent(in) :: jt(:) !
            !    Dimensions: (nlayers)
            INTEGER, intent(in) :: jt1(:) !
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: planklay(:,:) !
            !    Dimensions: (nlayers,nbndlw)
            REAL(KIND=r8), intent(in) :: planklev(0:,:) !
            !    Dimensions: (nlayers,nbndlw)
            REAL(KIND=r8), intent(in) :: plankbnd(:) !
            !    Dimensions: (nbndlw)
            REAL(KIND=r8), intent(in) :: colh2o(:) ! column amount (h2o)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: colco2(:) ! column amount (co2)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: colo3(:) ! column amount (o3)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: coln2o(:) ! column amount (n2o)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: colco(:) ! column amount (co)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: colch4(:) ! column amount (ch4)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: colo2(:) ! column amount (o2)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: colbrd(:) ! column amount (broadening gases)
            !    Dimensions: (nlayers)
            INTEGER, intent(in) :: indself(:)
            !    Dimensions: (nlayers)
            INTEGER, intent(in) :: indfor(:)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: selffac(:)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: selffrac(:)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: forfac(:)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: forfrac(:)
            !    Dimensions: (nlayers)
            INTEGER, intent(in) :: indminor(:)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: minorfrac(:)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: scaleminor(:)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: scaleminorn2(:)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: fac11(:)
            REAL(KIND=r8), intent(in) :: fac00(:)
            REAL(KIND=r8), intent(in) :: fac01(:)
            REAL(KIND=r8), intent(in) :: fac10(:) !
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: rat_h2oco2(:)
            REAL(KIND=r8), intent(in) :: rat_h2oco2_1(:)
            REAL(KIND=r8), intent(in) :: rat_h2oo3(:)
            REAL(KIND=r8), intent(in) :: rat_h2oo3_1(:)
            REAL(KIND=r8), intent(in) :: rat_h2on2o(:)
            REAL(KIND=r8), intent(in) :: rat_h2och4(:)
            REAL(KIND=r8), intent(in) :: rat_h2och4_1(:)
            REAL(KIND=r8), intent(in) :: rat_n2oco2(:)
            REAL(KIND=r8), intent(in) :: rat_n2oco2_1(:)
            REAL(KIND=r8), intent(in) :: rat_o3co2(:)
            REAL(KIND=r8), intent(in) :: rat_o3co2_1(:)
            REAL(KIND=r8), intent(in) :: rat_h2on2o_1(:) !
            !    Dimensions: (nlayers)
            ! ----- Output -----
            REAL(KIND=r8), intent(out) :: fracs(:,:) ! planck fractions
            !    Dimensions: (nlayers,ngptlw)
            REAL(KIND=r8), intent(out) :: taug(:,:) ! gaseous optical depth
            !    Dimensions: (nlayers,ngptlw)
      hvrtau = '$Revision: 1.7 $'
            ! Calculate gaseous optical depth and planck fractions for each spectral band.
      call taugb1
      call taugb2
      call taugb3
      call taugb4
      call taugb5
      call taugb6
      call taugb7
      call taugb8
      call taugb9
      call taugb10
      call taugb11
      call taugb12
      call taugb13
      call taugb14
      call taugb15
      call taugb16
            CONTAINS
            !----------------------------------------------------------------------------

            SUBROUTINE taugb1()
                !----------------------------------------------------------------------------
                ! ------- Modifications -------
                !  Written by Eli J. Mlawer, Atmospheric & Environmental Research.
                !  Revised by Michael J. Iacono, Atmospheric & Environmental Research.
                !
                !     band 1:  10-350 cm-1 (low key - h2o; low minor - n2)
                !                          (high key - h2o; high minor - n2)
                !
                !     note: previous versions of rrtm band 1:
                !           10-250 cm-1 (low - h2o; high - h2o)
                !----------------------------------------------------------------------------
                ! ------- Modules -------
                USE parrrtm, ONLY: ng1
                USE rrlw_kg01, ONLY: selfref
                USE rrlw_kg01, ONLY: forref
                USE rrlw_kg01, ONLY: ka_mn2
                USE rrlw_kg01, ONLY: absa
                USE rrlw_kg01, ONLY: fracrefa
                USE rrlw_kg01, ONLY: kb_mn2
                USE rrlw_kg01, ONLY: absb
                USE rrlw_kg01, ONLY: fracrefb
                ! ------- Declarations -------
                ! Local
                INTEGER :: lay
                INTEGER :: ind0
                INTEGER :: ind1
                INTEGER :: inds
                INTEGER :: indf
                INTEGER :: indm
                INTEGER :: ig
                REAL(KIND=r8) :: pp
                REAL(KIND=r8) :: corradj
                REAL(KIND=r8) :: scalen2
                REAL(KIND=r8) :: tauself
                REAL(KIND=r8) :: taufor
                REAL(KIND=r8) :: taun2
                ! Minor gas mapping levels:
                !     lower - n2, p = 142.5490 mbar, t = 215.70 k
                !     upper - n2, p = 142.5490 mbar, t = 215.70 k
                ! Compute the optical depth by interpolating in ln(pressure) and
                ! temperature.  Below laytrop, the water vapor self-continuum and
                ! foreign continuum is interpolated (in temperature) separately.
                ! Lower atmosphere loop
      do lay = 1, laytrop
         ind0 = ((jp(lay)-1)*5+(jt(lay)-1))*nspa(1) + 1
         ind1 = (jp(lay)*5+(jt1(lay)-1))*nspa(1) + 1
         inds = indself(lay)
         indf = indfor(lay)
         indm = indminor(lay)
         pp = pavel(lay)
         corradj =  1.
         if (pp .lt. 250._r8) then
            corradj = 1._r8 - 0.15_r8 * (250._r8-pp) / 154.4_r8
         endif
         scalen2 = colbrd(lay) * scaleminorn2(lay)
         do ig = 1, ng1
            tauself = selffac(lay) * (selfref(inds,ig) + selffrac(lay) * &
                 (selfref(inds+1,ig) - selfref(inds,ig)))
            taufor =  forfac(lay) * (forref(indf,ig) + forfrac(lay) * &
                 (forref(indf+1,ig) -  forref(indf,ig))) 
            taun2 = scalen2*(ka_mn2(indm,ig) + & 
                 minorfrac(lay) * (ka_mn2(indm+1,ig) - ka_mn2(indm,ig)))
            taug(lay,ig) = corradj * (colh2o(lay) * &
                (fac00(lay) * absa(ind0,ig) + &
                 fac10(lay) * absa(ind0+1,ig) + &
                 fac01(lay) * absa(ind1,ig) + &
                 fac11(lay) * absa(ind1+1,ig)) & 
                 + tauself + taufor + taun2)
             fracs(lay,ig) = fracrefa(ig)
         enddo
      enddo
                ! Upper atmosphere loop
      do lay = laytrop+1, nlayers
         ind0 = ((jp(lay)-13)*5+(jt(lay)-1))*nspb(1) + 1
         ind1 = ((jp(lay)-12)*5+(jt1(lay)-1))*nspb(1) + 1
         indf = indfor(lay)
         indm = indminor(lay)
         pp = pavel(lay)
         corradj =  1._r8 - 0.15_r8 * (pp / 95.6_r8)
         scalen2 = colbrd(lay) * scaleminorn2(lay)
         do ig = 1, ng1
            taufor = forfac(lay) * (forref(indf,ig) + &
                 forfrac(lay) * (forref(indf+1,ig) - forref(indf,ig))) 
            taun2 = scalen2*(kb_mn2(indm,ig) + & 
                 minorfrac(lay) * (kb_mn2(indm+1,ig) - kb_mn2(indm,ig)))
            taug(lay,ig) = corradj * (colh2o(lay) * &
                (fac00(lay) * absb(ind0,ig) + &
                 fac10(lay) * absb(ind0+1,ig) + &
                 fac01(lay) * absb(ind1,ig) + &
                 fac11(lay) * absb(ind1+1,ig)) &  
                 + taufor + taun2)
            fracs(lay,ig) = fracrefb(ig)
         enddo
      enddo
            END SUBROUTINE taugb1
            !----------------------------------------------------------------------------

            SUBROUTINE taugb2()
                !----------------------------------------------------------------------------
                !
                !     band 2:  350-500 cm-1 (low key - h2o; high key - h2o)
                !
                !     note: previous version of rrtm band 2:
                !           250 - 500 cm-1 (low - h2o; high - h2o)
                !----------------------------------------------------------------------------
                ! ------- Modules -------
                USE parrrtm, ONLY: ng2
                USE parrrtm, ONLY: ngs1
                USE rrlw_kg02, ONLY: selfref
                USE rrlw_kg02, ONLY: forref
                USE rrlw_kg02, ONLY: absa
                USE rrlw_kg02, ONLY: fracrefa
                USE rrlw_kg02, ONLY: absb
                USE rrlw_kg02, ONLY: fracrefb
                ! ------- Declarations -------
                ! Local
                INTEGER :: lay
                INTEGER :: ind0
                INTEGER :: ind1
                INTEGER :: inds
                INTEGER :: indf
                INTEGER :: ig
                REAL(KIND=r8) :: pp
                REAL(KIND=r8) :: corradj
                REAL(KIND=r8) :: tauself
                REAL(KIND=r8) :: taufor
                ! Compute the optical depth by interpolating in ln(pressure) and
                ! temperature.  Below laytrop, the water vapor self-continuum and
                ! foreign continuum is interpolated (in temperature) separately.
                ! Lower atmosphere loop
      do lay = 1, laytrop
         ind0 = ((jp(lay)-1)*5+(jt(lay)-1))*nspa(2) + 1
         ind1 = (jp(lay)*5+(jt1(lay)-1))*nspa(2) + 1
         inds = indself(lay)
         indf = indfor(lay)
         pp = pavel(lay)
         corradj = 1._r8 - .05_r8 * (pp - 100._r8) / 900._r8
         do ig = 1, ng2
            tauself = selffac(lay) * (selfref(inds,ig) + selffrac(lay) * &
                 (selfref(inds+1,ig) - selfref(inds,ig)))
            taufor =  forfac(lay) * (forref(indf,ig) + forfrac(lay) * &
                 (forref(indf+1,ig) - forref(indf,ig))) 
            taug(lay,ngs1+ig) = corradj * (colh2o(lay) * &
                (fac00(lay) * absa(ind0,ig) + &
                 fac10(lay) * absa(ind0+1,ig) + &
                 fac01(lay) * absa(ind1,ig) + &
                 fac11(lay) * absa(ind1+1,ig)) &
                 + tauself + taufor)
            fracs(lay,ngs1+ig) = fracrefa(ig)
         enddo
      enddo
                ! Upper atmosphere loop
      do lay = laytrop+1, nlayers
         ind0 = ((jp(lay)-13)*5+(jt(lay)-1))*nspb(2) + 1
         ind1 = ((jp(lay)-12)*5+(jt1(lay)-1))*nspb(2) + 1
         indf = indfor(lay)
         do ig = 1, ng2
            taufor =  forfac(lay) * (forref(indf,ig) + &
                 forfrac(lay) * (forref(indf+1,ig) - forref(indf,ig))) 
            taug(lay,ngs1+ig) = colh2o(lay) * &
                (fac00(lay) * absb(ind0,ig) + &
                 fac10(lay) * absb(ind0+1,ig) + &
                 fac01(lay) * absb(ind1,ig) + &
                 fac11(lay) * absb(ind1+1,ig)) &
                 + taufor
            fracs(lay,ngs1+ig) = fracrefb(ig)
         enddo
      enddo
            END SUBROUTINE taugb2
            !----------------------------------------------------------------------------

            SUBROUTINE taugb3()
                !----------------------------------------------------------------------------
                !
                !     band 3:  500-630 cm-1 (low key - h2o,co2; low minor - n2o)
                !                           (high key - h2o,co2; high minor - n2o)
                !----------------------------------------------------------------------------
                ! ------- Modules -------
                USE parrrtm, ONLY: ng3
                USE parrrtm, ONLY: ngs2
                USE rrlw_ref, ONLY: chi_mls
                USE rrlw_kg03, ONLY: selfref
                USE rrlw_kg03, ONLY: forref
                USE rrlw_kg03, ONLY: ka_mn2o
                USE rrlw_kg03, ONLY: absa
                USE rrlw_kg03, ONLY: fracrefa
                USE rrlw_kg03, ONLY: kb_mn2o
                USE rrlw_kg03, ONLY: absb
                USE rrlw_kg03, ONLY: fracrefb
                ! ------- Declarations -------
                ! Local
                INTEGER :: lay
                INTEGER :: ind0
                INTEGER :: ind1
                INTEGER :: inds
                INTEGER :: indf
                INTEGER :: indm
                INTEGER :: ig
                INTEGER :: js
                INTEGER :: js1
                INTEGER :: jmn2o
                INTEGER :: jpl
                REAL(KIND=r8) :: speccomb
                REAL(KIND=r8) :: specparm
                REAL(KIND=r8) :: specmult
                REAL(KIND=r8) :: fs
                REAL(KIND=r8) :: speccomb1
                REAL(KIND=r8) :: specparm1
                REAL(KIND=r8) :: specmult1
                REAL(KIND=r8) :: fs1
                REAL(KIND=r8) :: speccomb_mn2o
                REAL(KIND=r8) :: specparm_mn2o
                REAL(KIND=r8) :: specmult_mn2o
                REAL(KIND=r8) :: fmn2o
                REAL(KIND=r8) :: fmn2omf
                REAL(KIND=r8) :: chi_n2o
                REAL(KIND=r8) :: ratn2o
                REAL(KIND=r8) :: adjfac
                REAL(KIND=r8) :: adjcoln2o
                REAL(KIND=r8) :: speccomb_planck
                REAL(KIND=r8) :: specparm_planck
                REAL(KIND=r8) :: specmult_planck
                REAL(KIND=r8) :: fpl
                REAL(KIND=r8) :: p
                REAL(KIND=r8) :: p4
                REAL(KIND=r8) :: fk0
                REAL(KIND=r8) :: fk1
                REAL(KIND=r8) :: fk2
                REAL(KIND=r8) :: fac000
                REAL(KIND=r8) :: fac100
                REAL(KIND=r8) :: fac200
                REAL(KIND=r8) :: fac010
                REAL(KIND=r8) :: fac110
                REAL(KIND=r8) :: fac210
                REAL(KIND=r8) :: fac001
                REAL(KIND=r8) :: fac101
                REAL(KIND=r8) :: fac201
                REAL(KIND=r8) :: fac011
                REAL(KIND=r8) :: fac111
                REAL(KIND=r8) :: fac211
                REAL(KIND=r8) :: tauself
                REAL(KIND=r8) :: taufor
                REAL(KIND=r8) :: n2om1
                REAL(KIND=r8) :: n2om2
                REAL(KIND=r8) :: absn2o
                REAL(KIND=r8) :: refrat_planck_a
                REAL(KIND=r8) :: refrat_planck_b
                REAL(KIND=r8) :: refrat_m_a
                REAL(KIND=r8) :: refrat_m_b
                REAL(KIND=r8) :: tau_major
                REAL(KIND=r8) :: tau_major1
                ! Minor gas mapping levels:
                !     lower - n2o, p = 706.272 mbar, t = 278.94 k
                !     upper - n2o, p = 95.58 mbar, t = 215.7 k
                !  P = 212.725 mb
      refrat_planck_a = chi_mls(1,9)/chi_mls(2,9)
                !  P = 95.58 mb
      refrat_planck_b = chi_mls(1,13)/chi_mls(2,13)
                !  P = 706.270mb
      refrat_m_a = chi_mls(1,3)/chi_mls(2,3)
                !  P = 95.58 mb
      refrat_m_b = chi_mls(1,13)/chi_mls(2,13)
                ! Compute the optical depth by interpolating in ln(pressure) and
                ! temperature, and appropriate species.  Below laytrop, the water vapor
                ! self-continuum and foreign continuum is interpolated (in temperature)
                ! separately.
                ! Lower atmosphere loop
      do lay = 1, laytrop
         speccomb = colh2o(lay) + rat_h2oco2(lay)*colco2(lay)
         specparm = colh2o(lay)/speccomb
         if (specparm .ge. oneminus) specparm = oneminus
         specmult = 8._r8*(specparm)
         js = 1 + int(specmult)
         fs = mod(specmult,1.0_r8)        
         speccomb1 = colh2o(lay) + rat_h2oco2_1(lay)*colco2(lay)
         specparm1 = colh2o(lay)/speccomb1
         if (specparm1 .ge. oneminus) specparm1 = oneminus
         specmult1 = 8._r8*(specparm1)
         js1 = 1 + int(specmult1)
         fs1 = mod(specmult1,1.0_r8)
         speccomb_mn2o = colh2o(lay) + refrat_m_a*colco2(lay)
         specparm_mn2o = colh2o(lay)/speccomb_mn2o
         if (specparm_mn2o .ge. oneminus) specparm_mn2o = oneminus
         specmult_mn2o = 8._r8*specparm_mn2o
         jmn2o = 1 + int(specmult_mn2o)
         fmn2o = mod(specmult_mn2o,1.0_r8)
         fmn2omf = minorfrac(lay)*fmn2o
                    !  In atmospheres where the amount of N2O is too great to be considered
                    !  a minor species, adjust the column amount of N2O by an empirical factor
                    !  to obtain the proper contribution.
         chi_n2o = coln2o(lay)/coldry(lay)
         ratn2o = 1.e20_r8*chi_n2o/chi_mls(4,jp(lay)+1)
         if (ratn2o .gt. 1.5_r8) then
            adjfac = 0.5_r8+(ratn2o-0.5_r8)**0.65_r8
            adjcoln2o = adjfac*chi_mls(4,jp(lay)+1)*coldry(lay)*1.e-20_r8
         else
            adjcoln2o = coln2o(lay)
         endif
         speccomb_planck = colh2o(lay)+refrat_planck_a*colco2(lay)
         specparm_planck = colh2o(lay)/speccomb_planck
         if (specparm_planck .ge. oneminus) specparm_planck=oneminus
         specmult_planck = 8._r8*specparm_planck
         jpl= 1 + int(specmult_planck)
         fpl = mod(specmult_planck,1.0_r8)
         ind0 = ((jp(lay)-1)*5+(jt(lay)-1))*nspa(3) + js
         ind1 = (jp(lay)*5+(jt1(lay)-1))*nspa(3) + js1
         inds = indself(lay)
         indf = indfor(lay)
         indm = indminor(lay)
         if (specparm .lt. 0.125_r8) then
            p = fs - 1
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac000 = fk0*fac00(lay)
            fac100 = fk1*fac00(lay)
            fac200 = fk2*fac00(lay)
            fac010 = fk0*fac10(lay)
            fac110 = fk1*fac10(lay)
            fac210 = fk2*fac10(lay)
         else if (specparm .gt. 0.875_r8) then
            p = -fs 
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac000 = fk0*fac00(lay)
            fac100 = fk1*fac00(lay)
            fac200 = fk2*fac00(lay)
            fac010 = fk0*fac10(lay)
            fac110 = fk1*fac10(lay)
            fac210 = fk2*fac10(lay)
         else
            fac000 = (1._r8 - fs) * fac00(lay)
            fac010 = (1._r8 - fs) * fac10(lay)
            fac100 = fs * fac00(lay)
            fac110 = fs * fac10(lay)
         endif
         if (specparm1 .lt. 0.125_r8) then
            p = fs1 - 1
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac001 = fk0*fac01(lay)
            fac101 = fk1*fac01(lay)
            fac201 = fk2*fac01(lay)
            fac011 = fk0*fac11(lay)
            fac111 = fk1*fac11(lay)
            fac211 = fk2*fac11(lay)
         else if (specparm1 .gt. 0.875_r8) then
            p = -fs1 
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac001 = fk0*fac01(lay)
            fac101 = fk1*fac01(lay)
            fac201 = fk2*fac01(lay)
            fac011 = fk0*fac11(lay)
            fac111 = fk1*fac11(lay)
            fac211 = fk2*fac11(lay)
         else
            fac001 = (1._r8 - fs1) * fac01(lay)
            fac011 = (1._r8 - fs1) * fac11(lay)
            fac101 = fs1 * fac01(lay)
            fac111 = fs1 * fac11(lay)
         endif
         do ig = 1, ng3
            tauself = selffac(lay)* (selfref(inds,ig) + selffrac(lay) * &
                 (selfref(inds+1,ig) - selfref(inds,ig)))
            taufor = forfac(lay) * (forref(indf,ig) + forfrac(lay) * &
                 (forref(indf+1,ig) - forref(indf,ig))) 
            n2om1 = ka_mn2o(jmn2o,indm,ig) + fmn2o * &
                 (ka_mn2o(jmn2o+1,indm,ig) - ka_mn2o(jmn2o,indm,ig))
            n2om2 = ka_mn2o(jmn2o,indm+1,ig) + fmn2o * &
                 (ka_mn2o(jmn2o+1,indm+1,ig) - ka_mn2o(jmn2o,indm+1,ig))
            absn2o = n2om1 + minorfrac(lay) * (n2om2 - n2om1)
            if (specparm .lt. 0.125_r8) then
               tau_major = speccomb * &
                    (fac000 * absa(ind0,ig) + &
                    fac100 * absa(ind0+1,ig) + &
                    fac200 * absa(ind0+2,ig) + &
                    fac010 * absa(ind0+9,ig) + &
                    fac110 * absa(ind0+10,ig) + &
                    fac210 * absa(ind0+11,ig))
            else if (specparm .gt. 0.875_r8) then
               tau_major = speccomb * &
                    (fac200 * absa(ind0-1,ig) + &
                    fac100 * absa(ind0,ig) + &
                    fac000 * absa(ind0+1,ig) + &
                    fac210 * absa(ind0+8,ig) + &
                    fac110 * absa(ind0+9,ig) + &
                    fac010 * absa(ind0+10,ig))
            else
               tau_major = speccomb * &
                    (fac000 * absa(ind0,ig) + &
                    fac100 * absa(ind0+1,ig) + &
                    fac010 * absa(ind0+9,ig) + &
                    fac110 * absa(ind0+10,ig))
            endif
            if (specparm1 .lt. 0.125_r8) then
               tau_major1 = speccomb1 * &
                    (fac001 * absa(ind1,ig) + &
                    fac101 * absa(ind1+1,ig) + &
                    fac201 * absa(ind1+2,ig) + &
                    fac011 * absa(ind1+9,ig) + &
                    fac111 * absa(ind1+10,ig) + &
                    fac211 * absa(ind1+11,ig))
            else if (specparm1 .gt. 0.875_r8) then
               tau_major1 = speccomb1 * &
                    (fac201 * absa(ind1-1,ig) + &
                    fac101 * absa(ind1,ig) + &
                    fac001 * absa(ind1+1,ig) + &
                    fac211 * absa(ind1+8,ig) + &
                    fac111 * absa(ind1+9,ig) + &
                    fac011 * absa(ind1+10,ig))
            else
               tau_major1 = speccomb1 * &
                    (fac001 * absa(ind1,ig) +  &
                    fac101 * absa(ind1+1,ig) + &
                    fac011 * absa(ind1+9,ig) + &
                    fac111 * absa(ind1+10,ig))
            endif
            taug(lay,ngs2+ig) = tau_major + tau_major1 &
                 + tauself + taufor &
                 + adjcoln2o*absn2o
            fracs(lay,ngs2+ig) = fracrefa(ig,jpl) + fpl * &
                 (fracrefa(ig,jpl+1)-fracrefa(ig,jpl))
         enddo
      enddo
                ! Upper atmosphere loop
      do lay = laytrop+1, nlayers
         speccomb = colh2o(lay) + rat_h2oco2(lay)*colco2(lay)
         specparm = colh2o(lay)/speccomb
         if (specparm .ge. oneminus) specparm = oneminus
         specmult = 4._r8*(specparm)
         js = 1 + int(specmult)
         fs = mod(specmult,1.0_r8)
         speccomb1 = colh2o(lay) + rat_h2oco2_1(lay)*colco2(lay)
         specparm1 = colh2o(lay)/speccomb1
         if (specparm1 .ge. oneminus) specparm1 = oneminus
         specmult1 = 4._r8*(specparm1)
         js1 = 1 + int(specmult1)
         fs1 = mod(specmult1,1.0_r8)
         fac000 = (1._r8 - fs) * fac00(lay)
         fac010 = (1._r8 - fs) * fac10(lay)
         fac100 = fs * fac00(lay)
         fac110 = fs * fac10(lay)
         fac001 = (1._r8 - fs1) * fac01(lay)
         fac011 = (1._r8 - fs1) * fac11(lay)
         fac101 = fs1 * fac01(lay)
         fac111 = fs1 * fac11(lay)
         speccomb_mn2o = colh2o(lay) + refrat_m_b*colco2(lay)
         specparm_mn2o = colh2o(lay)/speccomb_mn2o
         if (specparm_mn2o .ge. oneminus) specparm_mn2o = oneminus
         specmult_mn2o = 4._r8*specparm_mn2o
         jmn2o = 1 + int(specmult_mn2o)
         fmn2o = mod(specmult_mn2o,1.0_r8)
         fmn2omf = minorfrac(lay)*fmn2o
                    !  In atmospheres where the amount of N2O is too great to be considered
                    !  a minor species, adjust the column amount of N2O by an empirical factor
                    !  to obtain the proper contribution.
         chi_n2o = coln2o(lay)/coldry(lay)
         ratn2o = 1.e20*chi_n2o/chi_mls(4,jp(lay)+1)
         if (ratn2o .gt. 1.5_r8) then
            adjfac = 0.5_r8+(ratn2o-0.5_r8)**0.65_r8
            adjcoln2o = adjfac*chi_mls(4,jp(lay)+1)*coldry(lay)*1.e-20_r8
         else
            adjcoln2o = coln2o(lay)
         endif
         speccomb_planck = colh2o(lay)+refrat_planck_b*colco2(lay)
         specparm_planck = colh2o(lay)/speccomb_planck
         if (specparm_planck .ge. oneminus) specparm_planck=oneminus
         specmult_planck = 4._r8*specparm_planck
         jpl= 1 + int(specmult_planck)
         fpl = mod(specmult_planck,1.0_r8)
         ind0 = ((jp(lay)-13)*5+(jt(lay)-1))*nspb(3) + js
         ind1 = ((jp(lay)-12)*5+(jt1(lay)-1))*nspb(3) + js1
         indf = indfor(lay)
         indm = indminor(lay)
         do ig = 1, ng3
            taufor = forfac(lay) * (forref(indf,ig) + &
                 forfrac(lay) * (forref(indf+1,ig) - forref(indf,ig))) 
            n2om1 = kb_mn2o(jmn2o,indm,ig) + fmn2o * &
                 (kb_mn2o(jmn2o+1,indm,ig)-kb_mn2o(jmn2o,indm,ig))
            n2om2 = kb_mn2o(jmn2o,indm+1,ig) + fmn2o * &
                 (kb_mn2o(jmn2o+1,indm+1,ig)-kb_mn2o(jmn2o,indm+1,ig))
            absn2o = n2om1 + minorfrac(lay) * (n2om2 - n2om1)
            taug(lay,ngs2+ig) = speccomb * &
                (fac000 * absb(ind0,ig) + &
                fac100 * absb(ind0+1,ig) + &
                fac010 * absb(ind0+5,ig) + &
                fac110 * absb(ind0+6,ig)) &
                + speccomb1 * &
                (fac001 * absb(ind1,ig) +  &
                fac101 * absb(ind1+1,ig) + &
                fac011 * absb(ind1+5,ig) + &
                fac111 * absb(ind1+6,ig))  &
                + taufor &
                + adjcoln2o*absn2o
            fracs(lay,ngs2+ig) = fracrefb(ig,jpl) + fpl * &
                (fracrefb(ig,jpl+1)-fracrefb(ig,jpl))
         enddo
      enddo
            END SUBROUTINE taugb3
            !----------------------------------------------------------------------------

            SUBROUTINE taugb4()
                !----------------------------------------------------------------------------
                !
                !     band 4:  630-700 cm-1 (low key - h2o,co2; high key - o3,co2)
                !----------------------------------------------------------------------------
                ! ------- Modules -------
                USE parrrtm, ONLY: ng4
                USE parrrtm, ONLY: ngs3
                USE rrlw_ref, ONLY: chi_mls
                USE rrlw_kg04, ONLY: selfref
                USE rrlw_kg04, ONLY: forref
                USE rrlw_kg04, ONLY: absa
                USE rrlw_kg04, ONLY: fracrefa
                USE rrlw_kg04, ONLY: absb
                USE rrlw_kg04, ONLY: fracrefb
                ! ------- Declarations -------
                ! Local
                INTEGER :: lay
                INTEGER :: ind0
                INTEGER :: ind1
                INTEGER :: inds
                INTEGER :: indf
                INTEGER :: ig
                INTEGER :: js
                INTEGER :: js1
                INTEGER :: jpl
                REAL(KIND=r8) :: speccomb
                REAL(KIND=r8) :: specparm
                REAL(KIND=r8) :: specmult
                REAL(KIND=r8) :: fs
                REAL(KIND=r8) :: speccomb1
                REAL(KIND=r8) :: specparm1
                REAL(KIND=r8) :: specmult1
                REAL(KIND=r8) :: fs1
                REAL(KIND=r8) :: speccomb_planck
                REAL(KIND=r8) :: specparm_planck
                REAL(KIND=r8) :: specmult_planck
                REAL(KIND=r8) :: fpl
                REAL(KIND=r8) :: p
                REAL(KIND=r8) :: p4
                REAL(KIND=r8) :: fk0
                REAL(KIND=r8) :: fk1
                REAL(KIND=r8) :: fk2
                REAL(KIND=r8) :: fac000
                REAL(KIND=r8) :: fac100
                REAL(KIND=r8) :: fac200
                REAL(KIND=r8) :: fac010
                REAL(KIND=r8) :: fac110
                REAL(KIND=r8) :: fac210
                REAL(KIND=r8) :: fac001
                REAL(KIND=r8) :: fac101
                REAL(KIND=r8) :: fac201
                REAL(KIND=r8) :: fac011
                REAL(KIND=r8) :: fac111
                REAL(KIND=r8) :: fac211
                REAL(KIND=r8) :: tauself
                REAL(KIND=r8) :: taufor
                REAL(KIND=r8) :: refrat_planck_a
                REAL(KIND=r8) :: refrat_planck_b
                REAL(KIND=r8) :: tau_major
                REAL(KIND=r8) :: tau_major1
                ! P =   142.5940 mb
      refrat_planck_a = chi_mls(1,11)/chi_mls(2,11)
                ! P = 95.58350 mb
      refrat_planck_b = chi_mls(3,13)/chi_mls(2,13)
                ! Compute the optical depth by interpolating in ln(pressure) and
                ! temperature, and appropriate species.  Below laytrop, the water
                ! vapor self-continuum and foreign continuum is interpolated (in temperature)
                ! separately.
                ! Lower atmosphere loop
      do lay = 1, laytrop
         speccomb = colh2o(lay) + rat_h2oco2(lay)*colco2(lay)
         specparm = colh2o(lay)/speccomb
         if (specparm .ge. oneminus) specparm = oneminus
         specmult = 8._r8*(specparm)
         js = 1 + int(specmult)
         fs = mod(specmult,1.0_r8)
         speccomb1 = colh2o(lay) + rat_h2oco2_1(lay)*colco2(lay)
         specparm1 = colh2o(lay)/speccomb1
         if (specparm1 .ge. oneminus) specparm1 = oneminus
         specmult1 = 8._r8*(specparm1)
         js1 = 1 + int(specmult1)
         fs1 = mod(specmult1,1.0_r8)
         speccomb_planck = colh2o(lay)+refrat_planck_a*colco2(lay)
         specparm_planck = colh2o(lay)/speccomb_planck
         if (specparm_planck .ge. oneminus) specparm_planck=oneminus
         specmult_planck = 8._r8*specparm_planck
         jpl= 1 + int(specmult_planck)
         fpl = mod(specmult_planck,1.0_r8)
         ind0 = ((jp(lay)-1)*5+(jt(lay)-1))*nspa(4) + js
         ind1 = (jp(lay)*5+(jt1(lay)-1))*nspa(4) + js1
         inds = indself(lay)
         indf = indfor(lay)
         if (specparm .lt. 0.125_r8) then
            p = fs - 1
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac000 = fk0*fac00(lay)
            fac100 = fk1*fac00(lay)
            fac200 = fk2*fac00(lay)
            fac010 = fk0*fac10(lay)
            fac110 = fk1*fac10(lay)
            fac210 = fk2*fac10(lay)
         else if (specparm .gt. 0.875_r8) then
            p = -fs 
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac000 = fk0*fac00(lay)
            fac100 = fk1*fac00(lay)
            fac200 = fk2*fac00(lay)
            fac010 = fk0*fac10(lay)
            fac110 = fk1*fac10(lay)
            fac210 = fk2*fac10(lay)
         else
            fac000 = (1._r8 - fs) * fac00(lay)
            fac010 = (1._r8 - fs) * fac10(lay)
            fac100 = fs * fac00(lay)
            fac110 = fs * fac10(lay)
         endif
         if (specparm1 .lt. 0.125_r8) then
            p = fs1 - 1
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac001 = fk0*fac01(lay)
            fac101 = fk1*fac01(lay)
            fac201 = fk2*fac01(lay)
            fac011 = fk0*fac11(lay)
            fac111 = fk1*fac11(lay)
            fac211 = fk2*fac11(lay)
         else if (specparm1 .gt. 0.875_r8) then
            p = -fs1 
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac001 = fk0*fac01(lay)
            fac101 = fk1*fac01(lay)
            fac201 = fk2*fac01(lay)
            fac011 = fk0*fac11(lay)
            fac111 = fk1*fac11(lay)
            fac211 = fk2*fac11(lay)
         else
            fac001 = (1._r8 - fs1) * fac01(lay)
            fac011 = (1._r8 - fs1) * fac11(lay)
            fac101 = fs1 * fac01(lay)
            fac111 = fs1 * fac11(lay)
         endif
         do ig = 1, ng4
            tauself = selffac(lay)* (selfref(inds,ig) + selffrac(lay) * &
                 (selfref(inds+1,ig) - selfref(inds,ig)))
            taufor =  forfac(lay) * (forref(indf,ig) + forfrac(lay) * &
                 (forref(indf+1,ig) - forref(indf,ig))) 
            if (specparm .lt. 0.125_r8) then
               tau_major = speccomb * &
                    (fac000 * absa(ind0,ig) + &
                    fac100 * absa(ind0+1,ig) + &
                    fac200 * absa(ind0+2,ig) + &
                    fac010 * absa(ind0+9,ig) + &
                    fac110 * absa(ind0+10,ig) + &
                    fac210 * absa(ind0+11,ig))
            else if (specparm .gt. 0.875_r8) then
               tau_major = speccomb * &
                    (fac200 * absa(ind0-1,ig) + &
                    fac100 * absa(ind0,ig) + &
                    fac000 * absa(ind0+1,ig) + &
                    fac210 * absa(ind0+8,ig) + &
                    fac110 * absa(ind0+9,ig) + &
                    fac010 * absa(ind0+10,ig))
            else
               tau_major = speccomb * &
                    (fac000 * absa(ind0,ig) + &
                    fac100 * absa(ind0+1,ig) + &
                    fac010 * absa(ind0+9,ig) + &
                    fac110 * absa(ind0+10,ig))
            endif
            if (specparm1 .lt. 0.125_r8) then
               tau_major1 = speccomb1 * &
                    (fac001 * absa(ind1,ig) +  &
                    fac101 * absa(ind1+1,ig) + &
                    fac201 * absa(ind1+2,ig) + &
                    fac011 * absa(ind1+9,ig) + &
                    fac111 * absa(ind1+10,ig) + &
                    fac211 * absa(ind1+11,ig))
            else if (specparm1 .gt. 0.875_r8) then
               tau_major1 = speccomb1 * &
                    (fac201 * absa(ind1-1,ig) + &
                    fac101 * absa(ind1,ig) + &
                    fac001 * absa(ind1+1,ig) + &
                    fac211 * absa(ind1+8,ig) + &
                    fac111 * absa(ind1+9,ig) + &
                    fac011 * absa(ind1+10,ig))
            else
               tau_major1 = speccomb1 * &
                    (fac001 * absa(ind1,ig) + &
                    fac101 * absa(ind1+1,ig) + &
                    fac011 * absa(ind1+9,ig) + &
                    fac111 * absa(ind1+10,ig))
            endif
            taug(lay,ngs3+ig) = tau_major + tau_major1 &
                 + tauself + taufor
            fracs(lay,ngs3+ig) = fracrefa(ig,jpl) + fpl * &
                 (fracrefa(ig,jpl+1)-fracrefa(ig,jpl))
         enddo
      enddo
                ! Upper atmosphere loop
      do lay = laytrop+1, nlayers
         speccomb = colo3(lay) + rat_o3co2(lay)*colco2(lay)
         specparm = colo3(lay)/speccomb
         if (specparm .ge. oneminus) specparm = oneminus
         specmult = 4._r8*(specparm)
         js = 1 + int(specmult)
         fs = mod(specmult,1.0_r8)
         speccomb1 = colo3(lay) + rat_o3co2_1(lay)*colco2(lay)
         specparm1 = colo3(lay)/speccomb1
         if (specparm1 .ge. oneminus) specparm1 = oneminus
         specmult1 = 4._r8*(specparm1)
         js1 = 1 + int(specmult1)
         fs1 = mod(specmult1,1.0_r8)
         fac000 = (1._r8 - fs) * fac00(lay)
         fac010 = (1._r8 - fs) * fac10(lay)
         fac100 = fs * fac00(lay)
         fac110 = fs * fac10(lay)
         fac001 = (1._r8 - fs1) * fac01(lay)
         fac011 = (1._r8 - fs1) * fac11(lay)
         fac101 = fs1 * fac01(lay)
         fac111 = fs1 * fac11(lay)
         speccomb_planck = colo3(lay)+refrat_planck_b*colco2(lay)
         specparm_planck = colo3(lay)/speccomb_planck
         if (specparm_planck .ge. oneminus) specparm_planck=oneminus
         specmult_planck = 4._r8*specparm_planck
         jpl= 1 + int(specmult_planck)
         fpl = mod(specmult_planck,1.0_r8)
         ind0 = ((jp(lay)-13)*5+(jt(lay)-1))*nspb(4) + js
         ind1 = ((jp(lay)-12)*5+(jt1(lay)-1))*nspb(4) + js1
         do ig = 1, ng4
            taug(lay,ngs3+ig) =  speccomb * &
                (fac000 * absb(ind0,ig) + &
                fac100 * absb(ind0+1,ig) + &
                fac010 * absb(ind0+5,ig) + &
                fac110 * absb(ind0+6,ig)) &
                + speccomb1 * &
                (fac001 * absb(ind1,ig) +  &
                fac101 * absb(ind1+1,ig) + &
                fac011 * absb(ind1+5,ig) + &
                fac111 * absb(ind1+6,ig))
            fracs(lay,ngs3+ig) = fracrefb(ig,jpl) + fpl * &
                (fracrefb(ig,jpl+1)-fracrefb(ig,jpl))
         enddo
                    ! Empirical modification to code to improve stratospheric cooling rates
                    ! for co2.  Revised to apply weighting for g-point reduction in this band.
         taug(lay,ngs3+8)=taug(lay,ngs3+8)*0.92
         taug(lay,ngs3+9)=taug(lay,ngs3+9)*0.88
         taug(lay,ngs3+10)=taug(lay,ngs3+10)*1.07
         taug(lay,ngs3+11)=taug(lay,ngs3+11)*1.1
         taug(lay,ngs3+12)=taug(lay,ngs3+12)*0.99
         taug(lay,ngs3+13)=taug(lay,ngs3+13)*0.88
         taug(lay,ngs3+14)=taug(lay,ngs3+14)*0.943
      enddo
            END SUBROUTINE taugb4
            !----------------------------------------------------------------------------

            SUBROUTINE taugb5()
                !----------------------------------------------------------------------------
                !
                !     band 5:  700-820 cm-1 (low key - h2o,co2; low minor - o3, ccl4)
                !                           (high key - o3,co2)
                !----------------------------------------------------------------------------
                ! ------- Modules -------
                USE parrrtm, ONLY: ng5
                USE parrrtm, ONLY: ngs4
                USE rrlw_ref, ONLY: chi_mls
                USE rrlw_kg05, ONLY: selfref
                USE rrlw_kg05, ONLY: forref
                USE rrlw_kg05, ONLY: ka_mo3
                USE rrlw_kg05, ONLY: absa
                USE rrlw_kg05, ONLY: ccl4
                USE rrlw_kg05, ONLY: fracrefa
                USE rrlw_kg05, ONLY: absb
                USE rrlw_kg05, ONLY: fracrefb
                ! ------- Declarations -------
                ! Local
                INTEGER :: lay
                INTEGER :: ind0
                INTEGER :: ind1
                INTEGER :: inds
                INTEGER :: indf
                INTEGER :: indm
                INTEGER :: ig
                INTEGER :: js
                INTEGER :: js1
                INTEGER :: jmo3
                INTEGER :: jpl
                REAL(KIND=r8) :: speccomb
                REAL(KIND=r8) :: specparm
                REAL(KIND=r8) :: specmult
                REAL(KIND=r8) :: fs
                REAL(KIND=r8) :: speccomb1
                REAL(KIND=r8) :: specparm1
                REAL(KIND=r8) :: specmult1
                REAL(KIND=r8) :: fs1
                REAL(KIND=r8) :: speccomb_mo3
                REAL(KIND=r8) :: specparm_mo3
                REAL(KIND=r8) :: specmult_mo3
                REAL(KIND=r8) :: fmo3
                REAL(KIND=r8) :: speccomb_planck
                REAL(KIND=r8) :: specparm_planck
                REAL(KIND=r8) :: specmult_planck
                REAL(KIND=r8) :: fpl
                REAL(KIND=r8) :: p
                REAL(KIND=r8) :: p4
                REAL(KIND=r8) :: fk0
                REAL(KIND=r8) :: fk1
                REAL(KIND=r8) :: fk2
                REAL(KIND=r8) :: fac000
                REAL(KIND=r8) :: fac100
                REAL(KIND=r8) :: fac200
                REAL(KIND=r8) :: fac010
                REAL(KIND=r8) :: fac110
                REAL(KIND=r8) :: fac210
                REAL(KIND=r8) :: fac001
                REAL(KIND=r8) :: fac101
                REAL(KIND=r8) :: fac201
                REAL(KIND=r8) :: fac011
                REAL(KIND=r8) :: fac111
                REAL(KIND=r8) :: fac211
                REAL(KIND=r8) :: tauself
                REAL(KIND=r8) :: taufor
                REAL(KIND=r8) :: o3m1
                REAL(KIND=r8) :: o3m2
                REAL(KIND=r8) :: abso3
                REAL(KIND=r8) :: refrat_planck_a
                REAL(KIND=r8) :: refrat_planck_b
                REAL(KIND=r8) :: refrat_m_a
                REAL(KIND=r8) :: tau_major
                REAL(KIND=r8) :: tau_major1
                ! Minor gas mapping level :
                !     lower - o3, p = 317.34 mbar, t = 240.77 k
                !     lower - ccl4
                ! Calculate reference ratio to be used in calculation of Planck
                ! fraction in lower/upper atmosphere.
                ! P = 473.420 mb
      refrat_planck_a = chi_mls(1,5)/chi_mls(2,5)
                ! P = 0.2369 mb
      refrat_planck_b = chi_mls(3,43)/chi_mls(2,43)
                ! P = 317.3480
      refrat_m_a = chi_mls(1,7)/chi_mls(2,7)
                ! Compute the optical depth by interpolating in ln(pressure) and
                ! temperature, and appropriate species.  Below laytrop, the
                ! water vapor self-continuum and foreign continuum is
                ! interpolated (in temperature) separately.
                ! Lower atmosphere loop
      do lay = 1, laytrop
         speccomb = colh2o(lay) + rat_h2oco2(lay)*colco2(lay)
         specparm = colh2o(lay)/speccomb
         if (specparm .ge. oneminus) specparm = oneminus
         specmult = 8._r8*(specparm)
         js = 1 + int(specmult)
         fs = mod(specmult,1.0_r8)
         speccomb1 = colh2o(lay) + rat_h2oco2_1(lay)*colco2(lay)
         specparm1 = colh2o(lay)/speccomb1
         if (specparm1 .ge. oneminus) specparm1 = oneminus
         specmult1 = 8._r8*(specparm1)
         js1 = 1 + int(specmult1)
         fs1 = mod(specmult1,1.0_r8)
         speccomb_mo3 = colh2o(lay) + refrat_m_a*colco2(lay)
         specparm_mo3 = colh2o(lay)/speccomb_mo3
         if (specparm_mo3 .ge. oneminus) specparm_mo3 = oneminus
         specmult_mo3 = 8._r8*specparm_mo3
         jmo3 = 1 + int(specmult_mo3)
         fmo3 = mod(specmult_mo3,1.0_r8)
         speccomb_planck = colh2o(lay)+refrat_planck_a*colco2(lay)
         specparm_planck = colh2o(lay)/speccomb_planck
         if (specparm_planck .ge. oneminus) specparm_planck=oneminus
         specmult_planck = 8._r8*specparm_planck
         jpl= 1 + int(specmult_planck)
         fpl = mod(specmult_planck,1.0_r8)
         ind0 = ((jp(lay)-1)*5+(jt(lay)-1))*nspa(5) + js
         ind1 = (jp(lay)*5+(jt1(lay)-1))*nspa(5) + js1
         inds = indself(lay)
         indf = indfor(lay)
         indm = indminor(lay)
         if (specparm .lt. 0.125_r8) then
            p = fs - 1
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac000 = fk0*fac00(lay)
            fac100 = fk1*fac00(lay)
            fac200 = fk2*fac00(lay)
            fac010 = fk0*fac10(lay)
            fac110 = fk1*fac10(lay)
            fac210 = fk2*fac10(lay)
         else if (specparm .gt. 0.875_r8) then
            p = -fs 
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac000 = fk0*fac00(lay)
            fac100 = fk1*fac00(lay)
            fac200 = fk2*fac00(lay)
            fac010 = fk0*fac10(lay)
            fac110 = fk1*fac10(lay)
            fac210 = fk2*fac10(lay)
         else
            fac000 = (1._r8 - fs) * fac00(lay)
            fac010 = (1._r8 - fs) * fac10(lay)
            fac100 = fs * fac00(lay)
            fac110 = fs * fac10(lay)
         endif
         if (specparm1 .lt. 0.125_r8) then
            p = fs1 - 1
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac001 = fk0*fac01(lay)
            fac101 = fk1*fac01(lay)
            fac201 = fk2*fac01(lay)
            fac011 = fk0*fac11(lay)
            fac111 = fk1*fac11(lay)
            fac211 = fk2*fac11(lay)
         else if (specparm1 .gt. 0.875_r8) then
            p = -fs1 
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac001 = fk0*fac01(lay)
            fac101 = fk1*fac01(lay)
            fac201 = fk2*fac01(lay)
            fac011 = fk0*fac11(lay)
            fac111 = fk1*fac11(lay)
            fac211 = fk2*fac11(lay)
         else
            fac001 = (1._r8 - fs1) * fac01(lay)
            fac011 = (1._r8 - fs1) * fac11(lay)
            fac101 = fs1 * fac01(lay)
            fac111 = fs1 * fac11(lay)
         endif
         do ig = 1, ng5
            tauself = selffac(lay) * (selfref(inds,ig) + selffrac(lay) * &
                 (selfref(inds+1,ig) - selfref(inds,ig)))
            taufor =  forfac(lay) * (forref(indf,ig) + forfrac(lay) * &
                 (forref(indf+1,ig) - forref(indf,ig))) 
            o3m1 = ka_mo3(jmo3,indm,ig) + fmo3 * &
                 (ka_mo3(jmo3+1,indm,ig)-ka_mo3(jmo3,indm,ig))
            o3m2 = ka_mo3(jmo3,indm+1,ig) + fmo3 * &
                 (ka_mo3(jmo3+1,indm+1,ig)-ka_mo3(jmo3,indm+1,ig))
            abso3 = o3m1 + minorfrac(lay)*(o3m2-o3m1)
            if (specparm .lt. 0.125_r8) then
               tau_major = speccomb * &
                    (fac000 * absa(ind0,ig) + &
                    fac100 * absa(ind0+1,ig) + &
                    fac200 * absa(ind0+2,ig) + &
                    fac010 * absa(ind0+9,ig) + &
                    fac110 * absa(ind0+10,ig) + &
                    fac210 * absa(ind0+11,ig))
            else if (specparm .gt. 0.875_r8) then
               tau_major = speccomb * &
                    (fac200 * absa(ind0-1,ig) + &
                    fac100 * absa(ind0,ig) + &
                    fac000 * absa(ind0+1,ig) + &
                    fac210 * absa(ind0+8,ig) + &
                    fac110 * absa(ind0+9,ig) + &
                    fac010 * absa(ind0+10,ig))
            else
               tau_major = speccomb * &
                    (fac000 * absa(ind0,ig) + &
                    fac100 * absa(ind0+1,ig) + &
                    fac010 * absa(ind0+9,ig) + &
                    fac110 * absa(ind0+10,ig))
            endif
            if (specparm1 .lt. 0.125_r8) then
               tau_major1 = speccomb1 * &
                    (fac001 * absa(ind1,ig) + &
                    fac101 * absa(ind1+1,ig) + &
                    fac201 * absa(ind1+2,ig) + &
                    fac011 * absa(ind1+9,ig) + &
                    fac111 * absa(ind1+10,ig) + &
                    fac211 * absa(ind1+11,ig))
            else if (specparm1 .gt. 0.875_r8) then
               tau_major1 = speccomb1 * & 
                    (fac201 * absa(ind1-1,ig) + &
                    fac101 * absa(ind1,ig) + &
                    fac001 * absa(ind1+1,ig) + &
                    fac211 * absa(ind1+8,ig) + &
                    fac111 * absa(ind1+9,ig) + &
                    fac011 * absa(ind1+10,ig))
            else
               tau_major1 = speccomb1 * &
                    (fac001 * absa(ind1,ig) + &
                    fac101 * absa(ind1+1,ig) + &
                    fac011 * absa(ind1+9,ig) + &
                    fac111 * absa(ind1+10,ig))
            endif
            taug(lay,ngs4+ig) = tau_major + tau_major1 &
                 + tauself + taufor &
                 + abso3*colo3(lay) &
                 + wx(1,lay) * ccl4(ig)
            fracs(lay,ngs4+ig) = fracrefa(ig,jpl) + fpl * &
                 (fracrefa(ig,jpl+1)-fracrefa(ig,jpl))
         enddo
      enddo
                ! Upper atmosphere loop
      do lay = laytrop+1, nlayers
         speccomb = colo3(lay) + rat_o3co2(lay)*colco2(lay)
         specparm = colo3(lay)/speccomb
         if (specparm .ge. oneminus) specparm = oneminus
         specmult = 4._r8*(specparm)
         js = 1 + int(specmult)
         fs = mod(specmult,1.0_r8)
         speccomb1 = colo3(lay) + rat_o3co2_1(lay)*colco2(lay)
         specparm1 = colo3(lay)/speccomb1
         if (specparm1 .ge. oneminus) specparm1 = oneminus
         specmult1 = 4._r8*(specparm1)
         js1 = 1 + int(specmult1)
         fs1 = mod(specmult1,1.0_r8)
         fac000 = (1._r8 - fs) * fac00(lay)
         fac010 = (1._r8 - fs) * fac10(lay)
         fac100 = fs * fac00(lay)
         fac110 = fs * fac10(lay)
         fac001 = (1._r8 - fs1) * fac01(lay)
         fac011 = (1._r8 - fs1) * fac11(lay)
         fac101 = fs1 * fac01(lay)
         fac111 = fs1 * fac11(lay)
         speccomb_planck = colo3(lay)+refrat_planck_b*colco2(lay)
         specparm_planck = colo3(lay)/speccomb_planck
         if (specparm_planck .ge. oneminus) specparm_planck=oneminus
         specmult_planck = 4._r8*specparm_planck
         jpl= 1 + int(specmult_planck)
         fpl = mod(specmult_planck,1.0_r8)
         ind0 = ((jp(lay)-13)*5+(jt(lay)-1))*nspb(5) + js
         ind1 = ((jp(lay)-12)*5+(jt1(lay)-1))*nspb(5) + js1
         do ig = 1, ng5
            taug(lay,ngs4+ig) = speccomb * &
                (fac000 * absb(ind0,ig) + &
                fac100 * absb(ind0+1,ig) + &
                fac010 * absb(ind0+5,ig) + &
                fac110 * absb(ind0+6,ig)) &
                + speccomb1 * &
                (fac001 * absb(ind1,ig) + &
                fac101 * absb(ind1+1,ig) + &
                fac011 * absb(ind1+5,ig) + &
                fac111 * absb(ind1+6,ig))  &
                + wx(1,lay) * ccl4(ig)
            fracs(lay,ngs4+ig) = fracrefb(ig,jpl) + fpl * &
                (fracrefb(ig,jpl+1)-fracrefb(ig,jpl))
         enddo
      enddo
            END SUBROUTINE taugb5
            !----------------------------------------------------------------------------

            SUBROUTINE taugb6()
                !----------------------------------------------------------------------------
                !
                !     band 6:  820-980 cm-1 (low key - h2o; low minor - co2)
                !                           (high key - nothing; high minor - cfc11, cfc12)
                !----------------------------------------------------------------------------
                ! ------- Modules -------
                USE parrrtm, ONLY: ng6
                USE parrrtm, ONLY: ngs5
                USE rrlw_ref, ONLY: chi_mls
                USE rrlw_kg06, ONLY: selfref
                USE rrlw_kg06, ONLY: forref
                USE rrlw_kg06, ONLY: ka_mco2
                USE rrlw_kg06, ONLY: cfc11adj
                USE rrlw_kg06, ONLY: absa
                USE rrlw_kg06, ONLY: cfc12
                USE rrlw_kg06, ONLY: fracrefa
                ! ------- Declarations -------
                ! Local
                INTEGER :: lay
                INTEGER :: ind0
                INTEGER :: ind1
                INTEGER :: inds
                INTEGER :: indf
                INTEGER :: indm
                INTEGER :: ig
                REAL(KIND=r8) :: chi_co2
                REAL(KIND=r8) :: ratco2
                REAL(KIND=r8) :: adjfac
                REAL(KIND=r8) :: adjcolco2
                REAL(KIND=r8) :: tauself
                REAL(KIND=r8) :: taufor
                REAL(KIND=r8) :: absco2
                ! Minor gas mapping level:
                !     lower - co2, p = 706.2720 mb, t = 294.2 k
                !     upper - cfc11, cfc12
                ! Compute the optical depth by interpolating in ln(pressure) and
                ! temperature. The water vapor self-continuum and foreign continuum
                ! is interpolated (in temperature) separately.
                ! Lower atmosphere loop
      do lay = 1, laytrop
                    ! In atmospheres where the amount of CO2 is too great to be considered
                    ! a minor species, adjust the column amount of CO2 by an empirical factor
                    ! to obtain the proper contribution.
         chi_co2 = colco2(lay)/(coldry(lay))
         ratco2 = 1.e20_r8*chi_co2/chi_mls(2,jp(lay)+1)
         if (ratco2 .gt. 3.0_r8) then
            adjfac = 2.0_r8+(ratco2-2.0_r8)**0.77_r8
            adjcolco2 = adjfac*chi_mls(2,jp(lay)+1)*coldry(lay)*1.e-20_r8
         else
            adjcolco2 = colco2(lay)
         endif
         ind0 = ((jp(lay)-1)*5+(jt(lay)-1))*nspa(6) + 1
         ind1 = (jp(lay)*5+(jt1(lay)-1))*nspa(6) + 1
         inds = indself(lay)
         indf = indfor(lay)
         indm = indminor(lay)
         do ig = 1, ng6
            tauself = selffac(lay) * (selfref(inds,ig) + selffrac(lay) * &
                 (selfref(inds+1,ig) - selfref(inds,ig)))
            taufor =  forfac(lay) * (forref(indf,ig) + forfrac(lay) * &
                 (forref(indf+1,ig) - forref(indf,ig)))
            absco2 =  (ka_mco2(indm,ig) + minorfrac(lay) * &
                 (ka_mco2(indm+1,ig) - ka_mco2(indm,ig)))
            taug(lay,ngs5+ig) = colh2o(lay) * &
                (fac00(lay) * absa(ind0,ig) + &
                 fac10(lay) * absa(ind0+1,ig) + &
                 fac01(lay) * absa(ind1,ig) +  &
                 fac11(lay) * absa(ind1+1,ig))  &
                 + tauself + taufor &
                 + adjcolco2 * absco2 &
                 + wx(2,lay) * cfc11adj(ig) &
                 + wx(3,lay) * cfc12(ig)
            fracs(lay,ngs5+ig) = fracrefa(ig)
         enddo
      enddo
                ! Upper atmosphere loop
                ! Nothing important goes on above laytrop in this band.
      do lay = laytrop+1, nlayers
         do ig = 1, ng6
            taug(lay,ngs5+ig) = 0.0_r8 &
                 + wx(2,lay) * cfc11adj(ig) &
                 + wx(3,lay) * cfc12(ig)
            fracs(lay,ngs5+ig) = fracrefa(ig)
         enddo
      enddo
            END SUBROUTINE taugb6
            !----------------------------------------------------------------------------

            SUBROUTINE taugb7()
                !----------------------------------------------------------------------------
                !
                !     band 7:  980-1080 cm-1 (low key - h2o,o3; low minor - co2)
                !                            (high key - o3; high minor - co2)
                !----------------------------------------------------------------------------
                ! ------- Modules -------
                USE parrrtm, ONLY: ng7
                USE parrrtm, ONLY: ngs6
                USE rrlw_ref, ONLY: chi_mls
                USE rrlw_kg07, ONLY: selfref
                USE rrlw_kg07, ONLY: forref
                USE rrlw_kg07, ONLY: ka_mco2
                USE rrlw_kg07, ONLY: absa
                USE rrlw_kg07, ONLY: fracrefa
                USE rrlw_kg07, ONLY: kb_mco2
                USE rrlw_kg07, ONLY: absb
                USE rrlw_kg07, ONLY: fracrefb
                ! ------- Declarations -------
                ! Local
                INTEGER :: lay
                INTEGER :: ind0
                INTEGER :: ind1
                INTEGER :: inds
                INTEGER :: indf
                INTEGER :: indm
                INTEGER :: ig
                INTEGER :: js
                INTEGER :: js1
                INTEGER :: jmco2
                INTEGER :: jpl
                REAL(KIND=r8) :: speccomb
                REAL(KIND=r8) :: specparm
                REAL(KIND=r8) :: specmult
                REAL(KIND=r8) :: fs
                REAL(KIND=r8) :: speccomb1
                REAL(KIND=r8) :: specparm1
                REAL(KIND=r8) :: specmult1
                REAL(KIND=r8) :: fs1
                REAL(KIND=r8) :: speccomb_mco2
                REAL(KIND=r8) :: specparm_mco2
                REAL(KIND=r8) :: specmult_mco2
                REAL(KIND=r8) :: fmco2
                REAL(KIND=r8) :: speccomb_planck
                REAL(KIND=r8) :: specparm_planck
                REAL(KIND=r8) :: specmult_planck
                REAL(KIND=r8) :: fpl
                REAL(KIND=r8) :: p
                REAL(KIND=r8) :: p4
                REAL(KIND=r8) :: fk0
                REAL(KIND=r8) :: fk1
                REAL(KIND=r8) :: fk2
                REAL(KIND=r8) :: fac000
                REAL(KIND=r8) :: fac100
                REAL(KIND=r8) :: fac200
                REAL(KIND=r8) :: fac010
                REAL(KIND=r8) :: fac110
                REAL(KIND=r8) :: fac210
                REAL(KIND=r8) :: fac001
                REAL(KIND=r8) :: fac101
                REAL(KIND=r8) :: fac201
                REAL(KIND=r8) :: fac011
                REAL(KIND=r8) :: fac111
                REAL(KIND=r8) :: fac211
                REAL(KIND=r8) :: tauself
                REAL(KIND=r8) :: taufor
                REAL(KIND=r8) :: co2m1
                REAL(KIND=r8) :: co2m2
                REAL(KIND=r8) :: absco2
                REAL(KIND=r8) :: chi_co2
                REAL(KIND=r8) :: ratco2
                REAL(KIND=r8) :: adjfac
                REAL(KIND=r8) :: adjcolco2
                REAL(KIND=r8) :: refrat_planck_a
                REAL(KIND=r8) :: refrat_m_a
                REAL(KIND=r8) :: tau_major
                REAL(KIND=r8) :: tau_major1
                ! Minor gas mapping level :
                !     lower - co2, p = 706.2620 mbar, t= 278.94 k
                !     upper - co2, p = 12.9350 mbar, t = 234.01 k
                ! Calculate reference ratio to be used in calculation of Planck
                ! fraction in lower atmosphere.
                ! P = 706.2620 mb
      refrat_planck_a = chi_mls(1,3)/chi_mls(3,3)
                ! P = 706.2720 mb
      refrat_m_a = chi_mls(1,3)/chi_mls(3,3)
                ! Compute the optical depth by interpolating in ln(pressure),
                ! temperature, and appropriate species.  Below laytrop, the water
                ! vapor self-continuum and foreign continuum is interpolated
                ! (in temperature) separately.
                ! Lower atmosphere loop
      do lay = 1, laytrop
         speccomb = colh2o(lay) + rat_h2oo3(lay)*colo3(lay)
         specparm = colh2o(lay)/speccomb
         if (specparm .ge. oneminus) specparm = oneminus
         specmult = 8._r8*(specparm)
         js = 1 + int(specmult)
         fs = mod(specmult,1.0_r8)
         speccomb1 = colh2o(lay) + rat_h2oo3_1(lay)*colo3(lay)
         specparm1 = colh2o(lay)/speccomb1
         if (specparm1 .ge. oneminus) specparm1 = oneminus
         specmult1 = 8._r8*(specparm1)
         js1 = 1 + int(specmult1)
         fs1 = mod(specmult1,1.0_r8)
         speccomb_mco2 = colh2o(lay) + refrat_m_a*colo3(lay)
         specparm_mco2 = colh2o(lay)/speccomb_mco2
         if (specparm_mco2 .ge. oneminus) specparm_mco2 = oneminus
         specmult_mco2 = 8._r8*specparm_mco2
         jmco2 = 1 + int(specmult_mco2)
         fmco2 = mod(specmult_mco2,1.0_r8)
                    !  In atmospheres where the amount of CO2 is too great to be considered
                    !  a minor species, adjust the column amount of CO2 by an empirical factor
                    !  to obtain the proper contribution.
         chi_co2 = colco2(lay)/(coldry(lay))
         ratco2 = 1.e20*chi_co2/chi_mls(2,jp(lay)+1)
         if (ratco2 .gt. 3.0_r8) then
            adjfac = 3.0_r8+(ratco2-3.0_r8)**0.79_r8
            adjcolco2 = adjfac*chi_mls(2,jp(lay)+1)*coldry(lay)*1.e-20_r8
         else
            adjcolco2 = colco2(lay)
         endif
         speccomb_planck = colh2o(lay)+refrat_planck_a*colo3(lay)
         specparm_planck = colh2o(lay)/speccomb_planck
         if (specparm_planck .ge. oneminus) specparm_planck=oneminus
         specmult_planck = 8._r8*specparm_planck
         jpl= 1 + int(specmult_planck)
         fpl = mod(specmult_planck,1.0_r8)
         ind0 = ((jp(lay)-1)*5+(jt(lay)-1))*nspa(7) + js
         ind1 = (jp(lay)*5+(jt1(lay)-1))*nspa(7) + js1
         inds = indself(lay)
         indf = indfor(lay)
         indm = indminor(lay)
         if (specparm .lt. 0.125_r8) then
            p = fs - 1
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac000 = fk0*fac00(lay)
            fac100 = fk1*fac00(lay)
            fac200 = fk2*fac00(lay)
            fac010 = fk0*fac10(lay)
            fac110 = fk1*fac10(lay)
            fac210 = fk2*fac10(lay)
         else if (specparm .gt. 0.875_r8) then
            p = -fs 
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac000 = fk0*fac00(lay)
            fac100 = fk1*fac00(lay)
            fac200 = fk2*fac00(lay)
            fac010 = fk0*fac10(lay)
            fac110 = fk1*fac10(lay)
            fac210 = fk2*fac10(lay)
         else
            fac000 = (1._r8 - fs) * fac00(lay)
            fac010 = (1._r8 - fs) * fac10(lay)
            fac100 = fs * fac00(lay)
            fac110 = fs * fac10(lay)
         endif
         if (specparm .lt. 0.125_r8) then
            p = fs1 - 1
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac001 = fk0*fac01(lay)
            fac101 = fk1*fac01(lay)
            fac201 = fk2*fac01(lay)
            fac011 = fk0*fac11(lay)
            fac111 = fk1*fac11(lay)
            fac211 = fk2*fac11(lay)
         else if (specparm1 .gt. 0.875_r8) then
            p = -fs1 
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac001 = fk0*fac01(lay)
            fac101 = fk1*fac01(lay)
            fac201 = fk2*fac01(lay)
            fac011 = fk0*fac11(lay)
            fac111 = fk1*fac11(lay)
            fac211 = fk2*fac11(lay)
         else
            fac001 = (1._r8 - fs1) * fac01(lay)
            fac011 = (1._r8 - fs1) * fac11(lay)
            fac101 = fs1 * fac01(lay)
            fac111 = fs1 * fac11(lay)
         endif
         do ig = 1, ng7
            tauself = selffac(lay)* (selfref(inds,ig) + selffrac(lay) * &
                 (selfref(inds+1,ig) - selfref(inds,ig)))
            taufor = forfac(lay) * (forref(indf,ig) + forfrac(lay) * &
                 (forref(indf+1,ig) - forref(indf,ig))) 
            co2m1 = ka_mco2(jmco2,indm,ig) + fmco2 * &
                 (ka_mco2(jmco2+1,indm,ig) - ka_mco2(jmco2,indm,ig))
            co2m2 = ka_mco2(jmco2,indm+1,ig) + fmco2 * &
                 (ka_mco2(jmco2+1,indm+1,ig) - ka_mco2(jmco2,indm+1,ig))
            absco2 = co2m1 + minorfrac(lay) * (co2m2 - co2m1)
            if (specparm .lt. 0.125_r8) then
               tau_major = speccomb * &
                    (fac000 * absa(ind0,ig) + &
                    fac100 * absa(ind0+1,ig) + &
                    fac200 * absa(ind0+2,ig) + &
                    fac010 * absa(ind0+9,ig) + &
                    fac110 * absa(ind0+10,ig) + &
                    fac210 * absa(ind0+11,ig))
            else if (specparm .gt. 0.875_r8) then
               tau_major = speccomb * &
                    (fac200 * absa(ind0-1,ig) + &
                    fac100 * absa(ind0,ig) + &
                    fac000 * absa(ind0+1,ig) + &
                    fac210 * absa(ind0+8,ig) + &
                    fac110 * absa(ind0+9,ig) + &
                    fac010 * absa(ind0+10,ig))
            else
               tau_major = speccomb * &
                    (fac000 * absa(ind0,ig) + &
                    fac100 * absa(ind0+1,ig) + &
                    fac010 * absa(ind0+9,ig) + &
                    fac110 * absa(ind0+10,ig))
            endif
            if (specparm1 .lt. 0.125_r8) then
               tau_major1 = speccomb1 * &
                    (fac001 * absa(ind1,ig) + &
                    fac101 * absa(ind1+1,ig) + &
                    fac201 * absa(ind1+2,ig) + &
                    fac011 * absa(ind1+9,ig) + &
                    fac111 * absa(ind1+10,ig) + &
                    fac211 * absa(ind1+11,ig))
            else if (specparm1 .gt. 0.875_r8) then
               tau_major1 = speccomb1 * &
                    (fac201 * absa(ind1-1,ig) + &
                    fac101 * absa(ind1,ig) + &
                    fac001 * absa(ind1+1,ig) + &
                    fac211 * absa(ind1+8,ig) + &
                    fac111 * absa(ind1+9,ig) + &
                    fac011 * absa(ind1+10,ig))
            else
               tau_major1 = speccomb1 * &
                    (fac001 * absa(ind1,ig) +  &
                    fac101 * absa(ind1+1,ig) + &
                    fac011 * absa(ind1+9,ig) + &
                    fac111 * absa(ind1+10,ig))
            endif
            taug(lay,ngs6+ig) = tau_major + tau_major1 &
                 + tauself + taufor &
                 + adjcolco2*absco2
            fracs(lay,ngs6+ig) = fracrefa(ig,jpl) + fpl * &
                 (fracrefa(ig,jpl+1)-fracrefa(ig,jpl))
         enddo
      enddo
                ! Upper atmosphere loop
      do lay = laytrop+1, nlayers
                    !  In atmospheres where the amount of CO2 is too great to be considered
                    !  a minor species, adjust the column amount of CO2 by an empirical factor
                    !  to obtain the proper contribution.
         chi_co2 = colco2(lay)/(coldry(lay))
         ratco2 = 1.e20*chi_co2/chi_mls(2,jp(lay)+1)
         if (ratco2 .gt. 3.0_r8) then
            adjfac = 2.0_r8+(ratco2-2.0_r8)**0.79_r8
            adjcolco2 = adjfac*chi_mls(2,jp(lay)+1)*coldry(lay)*1.e-20_r8
         else
            adjcolco2 = colco2(lay)
         endif
         ind0 = ((jp(lay)-13)*5+(jt(lay)-1))*nspb(7) + 1
         ind1 = ((jp(lay)-12)*5+(jt1(lay)-1))*nspb(7) + 1
         indm = indminor(lay)
         do ig = 1, ng7
            absco2 = kb_mco2(indm,ig) + minorfrac(lay) * &
                 (kb_mco2(indm+1,ig) - kb_mco2(indm,ig))
            taug(lay,ngs6+ig) = colo3(lay) * &
                 (fac00(lay) * absb(ind0,ig) + &
                 fac10(lay) * absb(ind0+1,ig) + &
                 fac01(lay) * absb(ind1,ig) + &
                 fac11(lay) * absb(ind1+1,ig)) &
                 + adjcolco2 * absco2
            fracs(lay,ngs6+ig) = fracrefb(ig)
         enddo
                    ! Empirical modification to code to improve stratospheric cooling rates
                    ! for o3.  Revised to apply weighting for g-point reduction in this band.
         taug(lay,ngs6+6)=taug(lay,ngs6+6)*0.92_r8
         taug(lay,ngs6+7)=taug(lay,ngs6+7)*0.88_r8
         taug(lay,ngs6+8)=taug(lay,ngs6+8)*1.07_r8
         taug(lay,ngs6+9)=taug(lay,ngs6+9)*1.1_r8
         taug(lay,ngs6+10)=taug(lay,ngs6+10)*0.99_r8
         taug(lay,ngs6+11)=taug(lay,ngs6+11)*0.855_r8
      enddo
            END SUBROUTINE taugb7
            !----------------------------------------------------------------------------

            SUBROUTINE taugb8()
                !----------------------------------------------------------------------------
                !
                !     band 8:  1080-1180 cm-1 (low key - h2o; low minor - co2,o3,n2o)
                !                             (high key - o3; high minor - co2, n2o)
                !----------------------------------------------------------------------------
                ! ------- Modules -------
                USE parrrtm, ONLY: ng8
                USE parrrtm, ONLY: ngs7
                USE rrlw_ref, ONLY: chi_mls
                USE rrlw_kg08, ONLY: selfref
                USE rrlw_kg08, ONLY: forref
                USE rrlw_kg08, ONLY: ka_mco2
                USE rrlw_kg08, ONLY: ka_mo3
                USE rrlw_kg08, ONLY: ka_mn2o
                USE rrlw_kg08, ONLY: cfc12
                USE rrlw_kg08, ONLY: cfc22adj
                USE rrlw_kg08, ONLY: absa
                USE rrlw_kg08, ONLY: fracrefa
                USE rrlw_kg08, ONLY: kb_mco2
                USE rrlw_kg08, ONLY: kb_mn2o
                USE rrlw_kg08, ONLY: absb
                USE rrlw_kg08, ONLY: fracrefb
                ! ------- Declarations -------
                ! Local
                INTEGER :: lay
                INTEGER :: ind0
                INTEGER :: ind1
                INTEGER :: inds
                INTEGER :: indf
                INTEGER :: indm
                INTEGER :: ig
                REAL(KIND=r8) :: tauself
                REAL(KIND=r8) :: taufor
                REAL(KIND=r8) :: absco2
                REAL(KIND=r8) :: abso3
                REAL(KIND=r8) :: absn2o
                REAL(KIND=r8) :: chi_co2
                REAL(KIND=r8) :: ratco2
                REAL(KIND=r8) :: adjfac
                REAL(KIND=r8) :: adjcolco2
                ! Minor gas mapping level:
                !     lower - co2, p = 1053.63 mb, t = 294.2 k
                !     lower - o3,  p = 317.348 mb, t = 240.77 k
                !     lower - n2o, p = 706.2720 mb, t= 278.94 k
                !     lower - cfc12,cfc11
                !     upper - co2, p = 35.1632 mb, t = 223.28 k
                !     upper - n2o, p = 8.716e-2 mb, t = 226.03 k
                ! Compute the optical depth by interpolating in ln(pressure) and
                ! temperature, and appropriate species.  Below laytrop, the water vapor
                ! self-continuum and foreign continuum is interpolated (in temperature)
                ! separately.
                ! Lower atmosphere loop
      do lay = 1, laytrop
                    !  In atmospheres where the amount of CO2 is too great to be considered
                    !  a minor species, adjust the column amount of CO2 by an empirical factor
                    !  to obtain the proper contribution.
         chi_co2 = colco2(lay)/(coldry(lay))
         ratco2 = 1.e20_r8*chi_co2/chi_mls(2,jp(lay)+1)
         if (ratco2 .gt. 3.0_r8) then
            adjfac = 2.0_r8+(ratco2-2.0_r8)**0.65_r8
            adjcolco2 = adjfac*chi_mls(2,jp(lay)+1)*coldry(lay)*1.e-20_r8
         else
            adjcolco2 = colco2(lay)
         endif
         ind0 = ((jp(lay)-1)*5+(jt(lay)-1))*nspa(8) + 1
         ind1 = (jp(lay)*5+(jt1(lay)-1))*nspa(8) + 1
         inds = indself(lay)
         indf = indfor(lay)
         indm = indminor(lay)
         do ig = 1, ng8
            tauself = selffac(lay) * (selfref(inds,ig) + selffrac(lay) * &
                 (selfref(inds+1,ig) - selfref(inds,ig)))
            taufor = forfac(lay) * (forref(indf,ig) + forfrac(lay) * &
                 (forref(indf+1,ig) - forref(indf,ig)))
            absco2 =  (ka_mco2(indm,ig) + minorfrac(lay) * &
                 (ka_mco2(indm+1,ig) - ka_mco2(indm,ig)))
            abso3 =  (ka_mo3(indm,ig) + minorfrac(lay) * &
                 (ka_mo3(indm+1,ig) - ka_mo3(indm,ig)))
            absn2o =  (ka_mn2o(indm,ig) + minorfrac(lay) * &
                 (ka_mn2o(indm+1,ig) - ka_mn2o(indm,ig)))
            taug(lay,ngs7+ig) = colh2o(lay) * &
                 (fac00(lay) * absa(ind0,ig) + &
                 fac10(lay) * absa(ind0+1,ig) + &
                 fac01(lay) * absa(ind1,ig) +  &
                 fac11(lay) * absa(ind1+1,ig)) &
                 + tauself + taufor &
                 + adjcolco2*absco2 &
                 + colo3(lay) * abso3 &
                 + coln2o(lay) * absn2o &
                 + wx(3,lay) * cfc12(ig) &
                 + wx(4,lay) * cfc22adj(ig)
            fracs(lay,ngs7+ig) = fracrefa(ig)
         enddo
      enddo
                ! Upper atmosphere loop
      do lay = laytrop+1, nlayers
                    !  In atmospheres where the amount of CO2 is too great to be considered
                    !  a minor species, adjust the column amount of CO2 by an empirical factor
                    !  to obtain the proper contribution.
         chi_co2 = colco2(lay)/coldry(lay)
         ratco2 = 1.e20_r8*chi_co2/chi_mls(2,jp(lay)+1)
         if (ratco2 .gt. 3.0_r8) then
            adjfac = 2.0_r8+(ratco2-2.0_r8)**0.65_r8
            adjcolco2 = adjfac*chi_mls(2,jp(lay)+1) * coldry(lay)*1.e-20_r8
         else
            adjcolco2 = colco2(lay)
         endif
         ind0 = ((jp(lay)-13)*5+(jt(lay)-1))*nspb(8) + 1
         ind1 = ((jp(lay)-12)*5+(jt1(lay)-1))*nspb(8) + 1
         indm = indminor(lay)
         do ig = 1, ng8
            absco2 =  (kb_mco2(indm,ig) + minorfrac(lay) * &
                 (kb_mco2(indm+1,ig) - kb_mco2(indm,ig)))
            absn2o =  (kb_mn2o(indm,ig) + minorfrac(lay) * &
                 (kb_mn2o(indm+1,ig) - kb_mn2o(indm,ig)))
            taug(lay,ngs7+ig) = colo3(lay) * &
                 (fac00(lay) * absb(ind0,ig) + &
                 fac10(lay) * absb(ind0+1,ig) + &
                 fac01(lay) * absb(ind1,ig) + &
                 fac11(lay) * absb(ind1+1,ig)) &
                 + adjcolco2*absco2 &
                 + coln2o(lay)*absn2o & 
                 + wx(3,lay) * cfc12(ig) &
                 + wx(4,lay) * cfc22adj(ig)
            fracs(lay,ngs7+ig) = fracrefb(ig)
         enddo
      enddo
            END SUBROUTINE taugb8
            !----------------------------------------------------------------------------

            SUBROUTINE taugb9()
                !----------------------------------------------------------------------------
                !
                !     band 9:  1180-1390 cm-1 (low key - h2o,ch4; low minor - n2o)
                !                             (high key - ch4; high minor - n2o)
                !----------------------------------------------------------------------------
                ! ------- Modules -------
                USE parrrtm, ONLY: ng9
                USE parrrtm, ONLY: ngs8
                USE rrlw_ref, ONLY: chi_mls
                USE rrlw_kg09, ONLY: selfref
                USE rrlw_kg09, ONLY: forref
                USE rrlw_kg09, ONLY: ka_mn2o
                USE rrlw_kg09, ONLY: absa
                USE rrlw_kg09, ONLY: fracrefa
                USE rrlw_kg09, ONLY: kb_mn2o
                USE rrlw_kg09, ONLY: absb
                USE rrlw_kg09, ONLY: fracrefb
                ! ------- Declarations -------
                ! Local
                INTEGER :: lay
                INTEGER :: ind0
                INTEGER :: ind1
                INTEGER :: inds
                INTEGER :: indf
                INTEGER :: indm
                INTEGER :: ig
                INTEGER :: js
                INTEGER :: js1
                INTEGER :: jmn2o
                INTEGER :: jpl
                REAL(KIND=r8) :: speccomb
                REAL(KIND=r8) :: specparm
                REAL(KIND=r8) :: specmult
                REAL(KIND=r8) :: fs
                REAL(KIND=r8) :: speccomb1
                REAL(KIND=r8) :: specparm1
                REAL(KIND=r8) :: specmult1
                REAL(KIND=r8) :: fs1
                REAL(KIND=r8) :: speccomb_mn2o
                REAL(KIND=r8) :: specparm_mn2o
                REAL(KIND=r8) :: specmult_mn2o
                REAL(KIND=r8) :: fmn2o
                REAL(KIND=r8) :: speccomb_planck
                REAL(KIND=r8) :: specparm_planck
                REAL(KIND=r8) :: specmult_planck
                REAL(KIND=r8) :: fpl
                REAL(KIND=r8) :: p
                REAL(KIND=r8) :: p4
                REAL(KIND=r8) :: fk0
                REAL(KIND=r8) :: fk1
                REAL(KIND=r8) :: fk2
                REAL(KIND=r8) :: fac000
                REAL(KIND=r8) :: fac100
                REAL(KIND=r8) :: fac200
                REAL(KIND=r8) :: fac010
                REAL(KIND=r8) :: fac110
                REAL(KIND=r8) :: fac210
                REAL(KIND=r8) :: fac001
                REAL(KIND=r8) :: fac101
                REAL(KIND=r8) :: fac201
                REAL(KIND=r8) :: fac011
                REAL(KIND=r8) :: fac111
                REAL(KIND=r8) :: fac211
                REAL(KIND=r8) :: tauself
                REAL(KIND=r8) :: taufor
                REAL(KIND=r8) :: n2om1
                REAL(KIND=r8) :: n2om2
                REAL(KIND=r8) :: absn2o
                REAL(KIND=r8) :: chi_n2o
                REAL(KIND=r8) :: ratn2o
                REAL(KIND=r8) :: adjfac
                REAL(KIND=r8) :: adjcoln2o
                REAL(KIND=r8) :: refrat_planck_a
                REAL(KIND=r8) :: refrat_m_a
                REAL(KIND=r8) :: tau_major
                REAL(KIND=r8) :: tau_major1
                ! Minor gas mapping level :
                !     lower - n2o, p = 706.272 mbar, t = 278.94 k
                !     upper - n2o, p = 95.58 mbar, t = 215.7 k
                ! Calculate reference ratio to be used in calculation of Planck
                ! fraction in lower/upper atmosphere.
                ! P = 212 mb
      refrat_planck_a = chi_mls(1,9)/chi_mls(6,9)
                ! P = 706.272 mb
      refrat_m_a = chi_mls(1,3)/chi_mls(6,3)
                ! Compute the optical depth by interpolating in ln(pressure),
                ! temperature, and appropriate species.  Below laytrop, the water
                ! vapor self-continuum and foreign continuum is interpolated
                ! (in temperature) separately.
                ! Lower atmosphere loop
      do lay = 1, laytrop
         speccomb = colh2o(lay) + rat_h2och4(lay)*colch4(lay)
         specparm = colh2o(lay)/speccomb
         if (specparm .ge. oneminus) specparm = oneminus
         specmult = 8._r8*(specparm)
         js = 1 + int(specmult)
         fs = mod(specmult,1.0_r8)
         speccomb1 = colh2o(lay) + rat_h2och4_1(lay)*colch4(lay)
         specparm1 = colh2o(lay)/speccomb1
         if (specparm1 .ge. oneminus) specparm1 = oneminus
         specmult1 = 8._r8*(specparm1)
         js1 = 1 + int(specmult1)
         fs1 = mod(specmult1,1.0_r8)
         speccomb_mn2o = colh2o(lay) + refrat_m_a*colch4(lay)
         specparm_mn2o = colh2o(lay)/speccomb_mn2o
         if (specparm_mn2o .ge. oneminus) specparm_mn2o = oneminus
         specmult_mn2o = 8._r8*specparm_mn2o
         jmn2o = 1 + int(specmult_mn2o)
         fmn2o = mod(specmult_mn2o,1.0_r8)
                    !  In atmospheres where the amount of N2O is too great to be considered
                    !  a minor species, adjust the column amount of N2O by an empirical factor
                    !  to obtain the proper contribution.
         chi_n2o = coln2o(lay)/(coldry(lay))
         ratn2o = 1.e20_r8*chi_n2o/chi_mls(4,jp(lay)+1)
         if (ratn2o .gt. 1.5_r8) then
            adjfac = 0.5_r8+(ratn2o-0.5_r8)**0.65_r8
            adjcoln2o = adjfac*chi_mls(4,jp(lay)+1)*coldry(lay)*1.e-20_r8
         else
            adjcoln2o = coln2o(lay)
         endif
         speccomb_planck = colh2o(lay)+refrat_planck_a*colch4(lay)
         specparm_planck = colh2o(lay)/speccomb_planck
         if (specparm_planck .ge. oneminus) specparm_planck=oneminus
         specmult_planck = 8._r8*specparm_planck
         jpl= 1 + int(specmult_planck)
         fpl = mod(specmult_planck,1.0_r8)
         ind0 = ((jp(lay)-1)*5+(jt(lay)-1))*nspa(9) + js
         ind1 = (jp(lay)*5+(jt1(lay)-1))*nspa(9) + js1
         inds = indself(lay)
         indf = indfor(lay)
         indm = indminor(lay)
         if (specparm .lt. 0.125_r8) then
            p = fs - 1
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac000 = fk0*fac00(lay)
            fac100 = fk1*fac00(lay)
            fac200 = fk2*fac00(lay)
            fac010 = fk0*fac10(lay)
            fac110 = fk1*fac10(lay)
            fac210 = fk2*fac10(lay)
         else if (specparm .gt. 0.875_r8) then
            p = -fs 
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac000 = fk0*fac00(lay)
            fac100 = fk1*fac00(lay)
            fac200 = fk2*fac00(lay)
            fac010 = fk0*fac10(lay)
            fac110 = fk1*fac10(lay)
            fac210 = fk2*fac10(lay)
         else
            fac000 = (1._r8 - fs) * fac00(lay)
            fac010 = (1._r8 - fs) * fac10(lay)
            fac100 = fs * fac00(lay)
            fac110 = fs * fac10(lay)
         endif
         if (specparm1 .lt. 0.125_r8) then
            p = fs1 - 1
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac001 = fk0*fac01(lay)
            fac101 = fk1*fac01(lay)
            fac201 = fk2*fac01(lay)
            fac011 = fk0*fac11(lay)
            fac111 = fk1*fac11(lay)
            fac211 = fk2*fac11(lay)
         else if (specparm1 .gt. 0.875_r8) then
            p = -fs1 
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac001 = fk0*fac01(lay)
            fac101 = fk1*fac01(lay)
            fac201 = fk2*fac01(lay)
            fac011 = fk0*fac11(lay)
            fac111 = fk1*fac11(lay)
            fac211 = fk2*fac11(lay)
         else
            fac001 = (1._r8 - fs1) * fac01(lay)
            fac011 = (1._r8 - fs1) * fac11(lay)
            fac101 = fs1 * fac01(lay)
            fac111 = fs1 * fac11(lay)
         endif
         do ig = 1, ng9
            tauself = selffac(lay)* (selfref(inds,ig) + selffrac(lay) * &
                 (selfref(inds+1,ig) - selfref(inds,ig)))
            taufor = forfac(lay) * (forref(indf,ig) + forfrac(lay) * &
                 (forref(indf+1,ig) - forref(indf,ig))) 
            n2om1 = ka_mn2o(jmn2o,indm,ig) + fmn2o * &
                 (ka_mn2o(jmn2o+1,indm,ig) - ka_mn2o(jmn2o,indm,ig))
            n2om2 = ka_mn2o(jmn2o,indm+1,ig) + fmn2o * &
                 (ka_mn2o(jmn2o+1,indm+1,ig) - ka_mn2o(jmn2o,indm+1,ig))
            absn2o = n2om1 + minorfrac(lay) * (n2om2 - n2om1)
            if (specparm .lt. 0.125_r8) then
               tau_major = speccomb * &
                    (fac000 * absa(ind0,ig) + &
                    fac100 * absa(ind0+1,ig) + &
                    fac200 * absa(ind0+2,ig) + &
                    fac010 * absa(ind0+9,ig) + &
                    fac110 * absa(ind0+10,ig) + &
                    fac210 * absa(ind0+11,ig))
            else if (specparm .gt. 0.875_r8) then
               tau_major = speccomb * &
                    (fac200 * absa(ind0-1,ig) + &
                    fac100 * absa(ind0,ig) + &
                    fac000 * absa(ind0+1,ig) + &
                    fac210 * absa(ind0+8,ig) + &
                    fac110 * absa(ind0+9,ig) + &
                    fac010 * absa(ind0+10,ig))
            else
               tau_major = speccomb * &
                    (fac000 * absa(ind0,ig) + &
                    fac100 * absa(ind0+1,ig) + &
                    fac010 * absa(ind0+9,ig) + &
                    fac110 * absa(ind0+10,ig))
            endif
            if (specparm1 .lt. 0.125_r8) then
               tau_major1 = speccomb1 * &
                    (fac001 * absa(ind1,ig) + & 
                    fac101 * absa(ind1+1,ig) + &
                    fac201 * absa(ind1+2,ig) + &
                    fac011 * absa(ind1+9,ig) + &
                    fac111 * absa(ind1+10,ig) + &
                    fac211 * absa(ind1+11,ig))
            else if (specparm1 .gt. 0.875_r8) then
               tau_major1 = speccomb1 * &
                    (fac201 * absa(ind1-1,ig) + &
                    fac101 * absa(ind1,ig) + &
                    fac001 * absa(ind1+1,ig) + &
                    fac211 * absa(ind1+8,ig) + &
                    fac111 * absa(ind1+9,ig) + &
                    fac011 * absa(ind1+10,ig))
            else
               tau_major1 = speccomb1 * &
                    (fac001 * absa(ind1,ig) + &
                    fac101 * absa(ind1+1,ig) + &
                    fac011 * absa(ind1+9,ig) + &
                    fac111 * absa(ind1+10,ig))
            endif
            taug(lay,ngs8+ig) = tau_major + tau_major1 &
                 + tauself + taufor &
                 + adjcoln2o*absn2o
            fracs(lay,ngs8+ig) = fracrefa(ig,jpl) + fpl * &
                 (fracrefa(ig,jpl+1)-fracrefa(ig,jpl))
         enddo
      enddo
                ! Upper atmosphere loop
      do lay = laytrop+1, nlayers
                    !  In atmospheres where the amount of N2O is too great to be considered
                    !  a minor species, adjust the column amount of N2O by an empirical factor
                    !  to obtain the proper contribution.
         chi_n2o = coln2o(lay)/(coldry(lay))
         ratn2o = 1.e20_r8*chi_n2o/chi_mls(4,jp(lay)+1)
         if (ratn2o .gt. 1.5_r8) then
            adjfac = 0.5_r8+(ratn2o-0.5_r8)**0.65_r8
            adjcoln2o = adjfac*chi_mls(4,jp(lay)+1)*coldry(lay)*1.e-20_r8
         else
            adjcoln2o = coln2o(lay)
         endif
         ind0 = ((jp(lay)-13)*5+(jt(lay)-1))*nspb(9) + 1
         ind1 = ((jp(lay)-12)*5+(jt1(lay)-1))*nspb(9) + 1
         indm = indminor(lay)
         do ig = 1, ng9
            absn2o = kb_mn2o(indm,ig) + minorfrac(lay) * &
                (kb_mn2o(indm+1,ig) - kb_mn2o(indm,ig))
            taug(lay,ngs8+ig) = colch4(lay) * &
                 (fac00(lay) * absb(ind0,ig) + &
                 fac10(lay) * absb(ind0+1,ig) + &
                 fac01(lay) * absb(ind1,ig) +  &
                 fac11(lay) * absb(ind1+1,ig)) &
                 + adjcoln2o*absn2o
            fracs(lay,ngs8+ig) = fracrefb(ig)
         enddo
      enddo
            END SUBROUTINE taugb9
            !----------------------------------------------------------------------------

            SUBROUTINE taugb10()
                !----------------------------------------------------------------------------
                !
                !     band 10:  1390-1480 cm-1 (low key - h2o; high key - h2o)
                !----------------------------------------------------------------------------
                ! ------- Modules -------
                USE parrrtm, ONLY: ng10
                USE parrrtm, ONLY: ngs9
                USE rrlw_kg10, ONLY: selfref
                USE rrlw_kg10, ONLY: forref
                USE rrlw_kg10, ONLY: absa
                USE rrlw_kg10, ONLY: fracrefa
                USE rrlw_kg10, ONLY: absb
                USE rrlw_kg10, ONLY: fracrefb
                ! ------- Declarations -------
                ! Local
                INTEGER :: lay
                INTEGER :: ind0
                INTEGER :: ind1
                INTEGER :: inds
                INTEGER :: indf
                INTEGER :: ig
                REAL(KIND=r8) :: tauself
                REAL(KIND=r8) :: taufor
                ! Compute the optical depth by interpolating in ln(pressure) and
                ! temperature.  Below laytrop, the water vapor self-continuum and
                ! foreign continuum is interpolated (in temperature) separately.
                ! Lower atmosphere loop
      do lay = 1, laytrop
         ind0 = ((jp(lay)-1)*5+(jt(lay)-1))*nspa(10) + 1
         ind1 = (jp(lay)*5+(jt1(lay)-1))*nspa(10) + 1
         inds = indself(lay)
         indf = indfor(lay)
         do ig = 1, ng10
            tauself = selffac(lay) * (selfref(inds,ig) + selffrac(lay) * &
                 (selfref(inds+1,ig) - selfref(inds,ig)))
            taufor = forfac(lay) * (forref(indf,ig) + forfrac(lay) * &
                 (forref(indf+1,ig) - forref(indf,ig))) 
            taug(lay,ngs9+ig) = colh2o(lay) * &
                 (fac00(lay) * absa(ind0,ig) + &
                 fac10(lay) * absa(ind0+1,ig) + &
                 fac01(lay) * absa(ind1,ig) + &
                 fac11(lay) * absa(ind1+1,ig))  &
                 + tauself + taufor
            fracs(lay,ngs9+ig) = fracrefa(ig)
         enddo
      enddo
                ! Upper atmosphere loop
      do lay = laytrop+1, nlayers
         ind0 = ((jp(lay)-13)*5+(jt(lay)-1))*nspb(10) + 1
         ind1 = ((jp(lay)-12)*5+(jt1(lay)-1))*nspb(10) + 1
         indf = indfor(lay)
         do ig = 1, ng10
            taufor = forfac(lay) * (forref(indf,ig) + forfrac(lay) * &
                 (forref(indf+1,ig) - forref(indf,ig))) 
            taug(lay,ngs9+ig) = colh2o(lay) * &
                 (fac00(lay) * absb(ind0,ig) + &
                 fac10(lay) * absb(ind0+1,ig) + &
                 fac01(lay) * absb(ind1,ig) +  &
                 fac11(lay) * absb(ind1+1,ig)) &
                 + taufor
            fracs(lay,ngs9+ig) = fracrefb(ig)
         enddo
      enddo
            END SUBROUTINE taugb10
            !----------------------------------------------------------------------------

            SUBROUTINE taugb11()
                !----------------------------------------------------------------------------
                !
                !     band 11:  1480-1800 cm-1 (low - h2o; low minor - o2)
                !                              (high key - h2o; high minor - o2)
                !----------------------------------------------------------------------------
                ! ------- Modules -------
                USE parrrtm, ONLY: ng11
                USE parrrtm, ONLY: ngs10
                USE rrlw_kg11, ONLY: selfref
                USE rrlw_kg11, ONLY: forref
                USE rrlw_kg11, ONLY: ka_mo2
                USE rrlw_kg11, ONLY: absa
                USE rrlw_kg11, ONLY: fracrefa
                USE rrlw_kg11, ONLY: kb_mo2
                USE rrlw_kg11, ONLY: absb
                USE rrlw_kg11, ONLY: fracrefb
                ! ------- Declarations -------
                ! Local
                INTEGER :: lay
                INTEGER :: ind0
                INTEGER :: ind1
                INTEGER :: inds
                INTEGER :: indf
                INTEGER :: indm
                INTEGER :: ig
                REAL(KIND=r8) :: scaleo2
                REAL(KIND=r8) :: tauself
                REAL(KIND=r8) :: taufor
                REAL(KIND=r8) :: tauo2
                ! Minor gas mapping level :
                !     lower - o2, p = 706.2720 mbar, t = 278.94 k
                !     upper - o2, p = 4.758820 mbarm t = 250.85 k
                ! Compute the optical depth by interpolating in ln(pressure) and
                ! temperature.  Below laytrop, the water vapor self-continuum and
                ! foreign continuum is interpolated (in temperature) separately.
                ! Lower atmosphere loop
      do lay = 1, laytrop
         ind0 = ((jp(lay)-1)*5+(jt(lay)-1))*nspa(11) + 1
         ind1 = (jp(lay)*5+(jt1(lay)-1))*nspa(11) + 1
         inds = indself(lay)
         indf = indfor(lay)
         indm = indminor(lay)
         scaleo2 = colo2(lay)*scaleminor(lay)
         do ig = 1, ng11
            tauself = selffac(lay) * (selfref(inds,ig) + selffrac(lay) * &
                 (selfref(inds+1,ig) - selfref(inds,ig)))
            taufor = forfac(lay) * (forref(indf,ig) + forfrac(lay) * &
                 (forref(indf+1,ig) - forref(indf,ig)))
            tauo2 =  scaleo2 * (ka_mo2(indm,ig) + minorfrac(lay) * &
                 (ka_mo2(indm+1,ig) - ka_mo2(indm,ig)))
            taug(lay,ngs10+ig) = colh2o(lay) * &
                 (fac00(lay) * absa(ind0,ig) + &
                 fac10(lay) * absa(ind0+1,ig) + &
                 fac01(lay) * absa(ind1,ig) + &
                 fac11(lay) * absa(ind1+1,ig)) &
                 + tauself + taufor &
                 + tauo2
            fracs(lay,ngs10+ig) = fracrefa(ig)
         enddo
      enddo
                ! Upper atmosphere loop
      do lay = laytrop+1, nlayers
         ind0 = ((jp(lay)-13)*5+(jt(lay)-1))*nspb(11) + 1
         ind1 = ((jp(lay)-12)*5+(jt1(lay)-1))*nspb(11) + 1
         indf = indfor(lay)
         indm = indminor(lay)
         scaleo2 = colo2(lay)*scaleminor(lay)
         do ig = 1, ng11
            taufor = forfac(lay) * (forref(indf,ig) + forfrac(lay) * &
                 (forref(indf+1,ig) - forref(indf,ig))) 
            tauo2 =  scaleo2 * (kb_mo2(indm,ig) + minorfrac(lay) * &
                 (kb_mo2(indm+1,ig) - kb_mo2(indm,ig)))
            taug(lay,ngs10+ig) = colh2o(lay) * &
                 (fac00(lay) * absb(ind0,ig) + &
                 fac10(lay) * absb(ind0+1,ig) + &
                 fac01(lay) * absb(ind1,ig) + &
                 fac11(lay) * absb(ind1+1,ig))  &
                 + taufor &
                 + tauo2
            fracs(lay,ngs10+ig) = fracrefb(ig)
         enddo
      enddo
            END SUBROUTINE taugb11
            !----------------------------------------------------------------------------

            SUBROUTINE taugb12()
                !----------------------------------------------------------------------------
                !
                !     band 12:  1800-2080 cm-1 (low - h2o,co2; high - nothing)
                !----------------------------------------------------------------------------
                ! ------- Modules -------
                USE parrrtm, ONLY: ng12
                USE parrrtm, ONLY: ngs11
                USE rrlw_ref, ONLY: chi_mls
                USE rrlw_kg12, ONLY: selfref
                USE rrlw_kg12, ONLY: forref
                USE rrlw_kg12, ONLY: absa
                USE rrlw_kg12, ONLY: fracrefa
                ! ------- Declarations -------
                ! Local
                INTEGER :: lay
                INTEGER :: ind0
                INTEGER :: ind1
                INTEGER :: inds
                INTEGER :: indf
                INTEGER :: ig
                INTEGER :: js
                INTEGER :: js1
                INTEGER :: jpl
                REAL(KIND=r8) :: speccomb
                REAL(KIND=r8) :: specparm
                REAL(KIND=r8) :: specmult
                REAL(KIND=r8) :: fs
                REAL(KIND=r8) :: speccomb1
                REAL(KIND=r8) :: specparm1
                REAL(KIND=r8) :: specmult1
                REAL(KIND=r8) :: fs1
                REAL(KIND=r8) :: speccomb_planck
                REAL(KIND=r8) :: specparm_planck
                REAL(KIND=r8) :: specmult_planck
                REAL(KIND=r8) :: fpl
                REAL(KIND=r8) :: p
                REAL(KIND=r8) :: p4
                REAL(KIND=r8) :: fk0
                REAL(KIND=r8) :: fk1
                REAL(KIND=r8) :: fk2
                REAL(KIND=r8) :: fac000
                REAL(KIND=r8) :: fac100
                REAL(KIND=r8) :: fac200
                REAL(KIND=r8) :: fac010
                REAL(KIND=r8) :: fac110
                REAL(KIND=r8) :: fac210
                REAL(KIND=r8) :: fac001
                REAL(KIND=r8) :: fac101
                REAL(KIND=r8) :: fac201
                REAL(KIND=r8) :: fac011
                REAL(KIND=r8) :: fac111
                REAL(KIND=r8) :: fac211
                REAL(KIND=r8) :: tauself
                REAL(KIND=r8) :: taufor
                REAL(KIND=r8) :: refrat_planck_a
                REAL(KIND=r8) :: tau_major
                REAL(KIND=r8) :: tau_major1
                ! Calculate reference ratio to be used in calculation of Planck
                ! fraction in lower/upper atmosphere.
                ! P =   174.164 mb
      refrat_planck_a = chi_mls(1,10)/chi_mls(2,10)
                ! Compute the optical depth by interpolating in ln(pressure),
                ! temperature, and appropriate species.  Below laytrop, the water
                ! vapor self-continuum adn foreign continuum is interpolated
                ! (in temperature) separately.
                ! Lower atmosphere loop
      do lay = 1, laytrop
         speccomb = colh2o(lay) + rat_h2oco2(lay)*colco2(lay)
         specparm = colh2o(lay)/speccomb
         if (specparm .ge. oneminus) specparm = oneminus
         specmult = 8._r8*(specparm)
         js = 1 + int(specmult)
         fs = mod(specmult,1.0_r8)
         speccomb1 = colh2o(lay) + rat_h2oco2_1(lay)*colco2(lay)
         specparm1 = colh2o(lay)/speccomb1
         if (specparm1 .ge. oneminus) specparm1 = oneminus
         specmult1 = 8._r8*(specparm1)
         js1 = 1 + int(specmult1)
         fs1 = mod(specmult1,1.0_r8)
         speccomb_planck = colh2o(lay)+refrat_planck_a*colco2(lay)
         specparm_planck = colh2o(lay)/speccomb_planck
         if (specparm_planck .ge. oneminus) specparm_planck=oneminus
         specmult_planck = 8._r8*specparm_planck
         jpl= 1 + int(specmult_planck)
         fpl = mod(specmult_planck,1.0_r8)
         ind0 = ((jp(lay)-1)*5+(jt(lay)-1))*nspa(12) + js
         ind1 = (jp(lay)*5+(jt1(lay)-1))*nspa(12) + js1
         inds = indself(lay)
         indf = indfor(lay)
         if (specparm .lt. 0.125_r8) then
            p = fs - 1
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac000 = fk0*fac00(lay)
            fac100 = fk1*fac00(lay)
            fac200 = fk2*fac00(lay)
            fac010 = fk0*fac10(lay)
            fac110 = fk1*fac10(lay)
            fac210 = fk2*fac10(lay)
         else if (specparm .gt. 0.875_r8) then
            p = -fs 
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac000 = fk0*fac00(lay)
            fac100 = fk1*fac00(lay)
            fac200 = fk2*fac00(lay)
            fac010 = fk0*fac10(lay)
            fac110 = fk1*fac10(lay)
            fac210 = fk2*fac10(lay)
         else
            fac000 = (1._r8 - fs) * fac00(lay)
            fac010 = (1._r8 - fs) * fac10(lay)
            fac100 = fs * fac00(lay)
            fac110 = fs * fac10(lay)
         endif
         if (specparm1 .lt. 0.125_r8) then
            p = fs1 - 1
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac001 = fk0*fac01(lay)
            fac101 = fk1*fac01(lay)
            fac201 = fk2*fac01(lay)
            fac011 = fk0*fac11(lay)
            fac111 = fk1*fac11(lay)
            fac211 = fk2*fac11(lay)
         else if (specparm1 .gt. 0.875_r8) then
            p = -fs1 
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac001 = fk0*fac01(lay)
            fac101 = fk1*fac01(lay)
            fac201 = fk2*fac01(lay)
            fac011 = fk0*fac11(lay)
            fac111 = fk1*fac11(lay)
            fac211 = fk2*fac11(lay)
         else
            fac001 = (1._r8 - fs1) * fac01(lay)
            fac011 = (1._r8 - fs1) * fac11(lay)
            fac101 = fs1 * fac01(lay)
            fac111 = fs1 * fac11(lay)
         endif
         do ig = 1, ng12
            tauself = selffac(lay)* (selfref(inds,ig) + selffrac(lay) * &
                 (selfref(inds+1,ig) - selfref(inds,ig)))
            taufor = forfac(lay) * (forref(indf,ig) + forfrac(lay) * &
                 (forref(indf+1,ig) - forref(indf,ig))) 
            if (specparm .lt. 0.125_r8) then
               tau_major = speccomb * &
                    (fac000 * absa(ind0,ig) + &
                    fac100 * absa(ind0+1,ig) + &
                    fac200 * absa(ind0+2,ig) + &
                    fac010 * absa(ind0+9,ig) + &
                    fac110 * absa(ind0+10,ig) + &
                    fac210 * absa(ind0+11,ig))
            else if (specparm .gt. 0.875_r8) then
               tau_major = speccomb * &
                    (fac200 * absa(ind0-1,ig) + &
                    fac100 * absa(ind0,ig) + &
                    fac000 * absa(ind0+1,ig) + &
                    fac210 * absa(ind0+8,ig) + &
                    fac110 * absa(ind0+9,ig) + &
                    fac010 * absa(ind0+10,ig))
            else
               tau_major = speccomb * &
                    (fac000 * absa(ind0,ig) + &
                    fac100 * absa(ind0+1,ig) + &
                    fac010 * absa(ind0+9,ig) + &
                    fac110 * absa(ind0+10,ig))
            endif
            if (specparm1 .lt. 0.125_r8) then
               tau_major1 = speccomb1 * &
                    (fac001 * absa(ind1,ig) + &
                    fac101 * absa(ind1+1,ig) + &
                    fac201 * absa(ind1+2,ig) + &
                    fac011 * absa(ind1+9,ig) + &
                    fac111 * absa(ind1+10,ig) + &
                    fac211 * absa(ind1+11,ig))
            else if (specparm1 .gt. 0.875_r8) then
               tau_major1 = speccomb1 * &
                    (fac201 * absa(ind1-1,ig) + &
                    fac101 * absa(ind1,ig) + &
                    fac001 * absa(ind1+1,ig) + &
                    fac211 * absa(ind1+8,ig) + &
                    fac111 * absa(ind1+9,ig) + &
                    fac011 * absa(ind1+10,ig))
            else
               tau_major1 = speccomb1 * &
                    (fac001 * absa(ind1,ig) + &
                    fac101 * absa(ind1+1,ig) + &
                    fac011 * absa(ind1+9,ig) + &
                    fac111 * absa(ind1+10,ig))
            endif
            taug(lay,ngs11+ig) = tau_major + tau_major1 &
                 + tauself + taufor
            fracs(lay,ngs11+ig) = fracrefa(ig,jpl) + fpl * &
                 (fracrefa(ig,jpl+1)-fracrefa(ig,jpl))
         enddo
      enddo
                ! Upper atmosphere loop
      do lay = laytrop+1, nlayers
         do ig = 1, ng12
            taug(lay,ngs11+ig) = 0.0_r8
            fracs(lay,ngs11+ig) = 0.0_r8
         enddo
      enddo
            END SUBROUTINE taugb12
            !----------------------------------------------------------------------------

            SUBROUTINE taugb13()
                !----------------------------------------------------------------------------
                !
                !     band 13:  2080-2250 cm-1 (low key - h2o,n2o; high minor - o3 minor)
                !----------------------------------------------------------------------------
                ! ------- Modules -------
                USE parrrtm, ONLY: ng13
                USE parrrtm, ONLY: ngs12
                USE rrlw_ref, ONLY: chi_mls
                USE rrlw_kg13, ONLY: selfref
                USE rrlw_kg13, ONLY: forref
                USE rrlw_kg13, ONLY: ka_mco2
                USE rrlw_kg13, ONLY: ka_mco
                USE rrlw_kg13, ONLY: absa
                USE rrlw_kg13, ONLY: fracrefa
                USE rrlw_kg13, ONLY: kb_mo3
                USE rrlw_kg13, ONLY: fracrefb
                ! ------- Declarations -------
                ! Local
                INTEGER :: lay
                INTEGER :: ind0
                INTEGER :: ind1
                INTEGER :: inds
                INTEGER :: indf
                INTEGER :: indm
                INTEGER :: ig
                INTEGER :: js
                INTEGER :: js1
                INTEGER :: jmco2
                INTEGER :: jmco
                INTEGER :: jpl
                REAL(KIND=r8) :: speccomb
                REAL(KIND=r8) :: specparm
                REAL(KIND=r8) :: specmult
                REAL(KIND=r8) :: fs
                REAL(KIND=r8) :: speccomb1
                REAL(KIND=r8) :: specparm1
                REAL(KIND=r8) :: specmult1
                REAL(KIND=r8) :: fs1
                REAL(KIND=r8) :: speccomb_mco2
                REAL(KIND=r8) :: specparm_mco2
                REAL(KIND=r8) :: specmult_mco2
                REAL(KIND=r8) :: fmco2
                REAL(KIND=r8) :: speccomb_mco
                REAL(KIND=r8) :: specparm_mco
                REAL(KIND=r8) :: specmult_mco
                REAL(KIND=r8) :: fmco
                REAL(KIND=r8) :: speccomb_planck
                REAL(KIND=r8) :: specparm_planck
                REAL(KIND=r8) :: specmult_planck
                REAL(KIND=r8) :: fpl
                REAL(KIND=r8) :: p
                REAL(KIND=r8) :: p4
                REAL(KIND=r8) :: fk0
                REAL(KIND=r8) :: fk1
                REAL(KIND=r8) :: fk2
                REAL(KIND=r8) :: fac000
                REAL(KIND=r8) :: fac100
                REAL(KIND=r8) :: fac200
                REAL(KIND=r8) :: fac010
                REAL(KIND=r8) :: fac110
                REAL(KIND=r8) :: fac210
                REAL(KIND=r8) :: fac001
                REAL(KIND=r8) :: fac101
                REAL(KIND=r8) :: fac201
                REAL(KIND=r8) :: fac011
                REAL(KIND=r8) :: fac111
                REAL(KIND=r8) :: fac211
                REAL(KIND=r8) :: tauself
                REAL(KIND=r8) :: taufor
                REAL(KIND=r8) :: co2m1
                REAL(KIND=r8) :: co2m2
                REAL(KIND=r8) :: absco2
                REAL(KIND=r8) :: com1
                REAL(KIND=r8) :: com2
                REAL(KIND=r8) :: absco
                REAL(KIND=r8) :: abso3
                REAL(KIND=r8) :: chi_co2
                REAL(KIND=r8) :: ratco2
                REAL(KIND=r8) :: adjfac
                REAL(KIND=r8) :: adjcolco2
                REAL(KIND=r8) :: refrat_planck_a
                REAL(KIND=r8) :: refrat_m_a
                REAL(KIND=r8) :: refrat_m_a3
                REAL(KIND=r8) :: tau_major
                REAL(KIND=r8) :: tau_major1
                ! Minor gas mapping levels :
                !     lower - co2, p = 1053.63 mb, t = 294.2 k
                !     lower - co, p = 706 mb, t = 278.94 k
                !     upper - o3, p = 95.5835 mb, t = 215.7 k
                ! Calculate reference ratio to be used in calculation of Planck
                ! fraction in lower/upper atmosphere.
                ! P = 473.420 mb (Level 5)
      refrat_planck_a = chi_mls(1,5)/chi_mls(4,5)
                ! P = 1053. (Level 1)
      refrat_m_a = chi_mls(1,1)/chi_mls(4,1)
                ! P = 706. (Level 3)
      refrat_m_a3 = chi_mls(1,3)/chi_mls(4,3)
                ! Compute the optical depth by interpolating in ln(pressure),
                ! temperature, and appropriate species.  Below laytrop, the water
                ! vapor self-continuum and foreign continuum is interpolated
                ! (in temperature) separately.
                ! Lower atmosphere loop
      do lay = 1, laytrop
         speccomb = colh2o(lay) + rat_h2on2o(lay)*coln2o(lay)
         specparm = colh2o(lay)/speccomb
         if (specparm .ge. oneminus) specparm = oneminus
         specmult = 8._r8*(specparm)
         js = 1 + int(specmult)
         fs = mod(specmult,1.0_r8)
         speccomb1 = colh2o(lay) + rat_h2on2o_1(lay)*coln2o(lay)
         specparm1 = colh2o(lay)/speccomb1
         if (specparm1 .ge. oneminus) specparm1 = oneminus
         specmult1 = 8._r8*(specparm1)
         js1 = 1 + int(specmult1)
         fs1 = mod(specmult1,1.0_r8)
         speccomb_mco2 = colh2o(lay) + refrat_m_a*coln2o(lay)
         specparm_mco2 = colh2o(lay)/speccomb_mco2
         if (specparm_mco2 .ge. oneminus) specparm_mco2 = oneminus
         specmult_mco2 = 8._r8*specparm_mco2
         jmco2 = 1 + int(specmult_mco2)
         fmco2 = mod(specmult_mco2,1.0_r8)
                    !  In atmospheres where the amount of CO2 is too great to be considered
                    !  a minor species, adjust the column amount of CO2 by an empirical factor
                    !  to obtain the proper contribution.
         chi_co2 = colco2(lay)/(coldry(lay))
         ratco2 = 1.e20_r8*chi_co2/3.55e-4_r8
         if (ratco2 .gt. 3.0_r8) then
            adjfac = 2.0_r8+(ratco2-2.0_r8)**0.68_r8
            adjcolco2 = adjfac*3.55e-4*coldry(lay)*1.e-20_r8
         else
            adjcolco2 = colco2(lay)
         endif
         speccomb_mco = colh2o(lay) + refrat_m_a3*coln2o(lay)
         specparm_mco = colh2o(lay)/speccomb_mco
         if (specparm_mco .ge. oneminus) specparm_mco = oneminus
         specmult_mco = 8._r8*specparm_mco
         jmco = 1 + int(specmult_mco)
         fmco = mod(specmult_mco,1.0_r8)
         speccomb_planck = colh2o(lay)+refrat_planck_a*coln2o(lay)
         specparm_planck = colh2o(lay)/speccomb_planck
         if (specparm_planck .ge. oneminus) specparm_planck=oneminus
         specmult_planck = 8._r8*specparm_planck
         jpl= 1 + int(specmult_planck)
         fpl = mod(specmult_planck,1.0_r8)
         ind0 = ((jp(lay)-1)*5+(jt(lay)-1))*nspa(13) + js
         ind1 = (jp(lay)*5+(jt1(lay)-1))*nspa(13) + js1
         inds = indself(lay)
         indf = indfor(lay)
         indm = indminor(lay)
         if (specparm .lt. 0.125_r8) then
            p = fs - 1
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac000 = fk0*fac00(lay)
            fac100 = fk1*fac00(lay)
            fac200 = fk2*fac00(lay)
            fac010 = fk0*fac10(lay)
            fac110 = fk1*fac10(lay)
            fac210 = fk2*fac10(lay)
         else if (specparm .gt. 0.875_r8) then
            p = -fs 
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac000 = fk0*fac00(lay)
            fac100 = fk1*fac00(lay)
            fac200 = fk2*fac00(lay)
            fac010 = fk0*fac10(lay)
            fac110 = fk1*fac10(lay)
            fac210 = fk2*fac10(lay)
         else
            fac000 = (1._r8 - fs) * fac00(lay)
            fac010 = (1._r8 - fs) * fac10(lay)
            fac100 = fs * fac00(lay)
            fac110 = fs * fac10(lay)
         endif
         if (specparm1 .lt. 0.125_r8) then
            p = fs1 - 1
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac001 = fk0*fac01(lay)
            fac101 = fk1*fac01(lay)
            fac201 = fk2*fac01(lay)
            fac011 = fk0*fac11(lay)
            fac111 = fk1*fac11(lay)
            fac211 = fk2*fac11(lay)
         else if (specparm1 .gt. 0.875_r8) then
            p = -fs1 
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac001 = fk0*fac01(lay)
            fac101 = fk1*fac01(lay)
            fac201 = fk2*fac01(lay)
            fac011 = fk0*fac11(lay)
            fac111 = fk1*fac11(lay)
            fac211 = fk2*fac11(lay)
         else
            fac001 = (1._r8 - fs1) * fac01(lay)
            fac011 = (1._r8 - fs1) * fac11(lay)
            fac101 = fs1 * fac01(lay)
            fac111 = fs1 * fac11(lay)
         endif
         do ig = 1, ng13
            tauself = selffac(lay)* (selfref(inds,ig) + selffrac(lay) * &
                 (selfref(inds+1,ig) - selfref(inds,ig)))
            taufor = forfac(lay) * (forref(indf,ig) + forfrac(lay) * &
                 (forref(indf+1,ig) - forref(indf,ig))) 
            co2m1 = ka_mco2(jmco2,indm,ig) + fmco2 * &
                 (ka_mco2(jmco2+1,indm,ig) - ka_mco2(jmco2,indm,ig))
            co2m2 = ka_mco2(jmco2,indm+1,ig) + fmco2 * &
                 (ka_mco2(jmco2+1,indm+1,ig) - ka_mco2(jmco2,indm+1,ig))
            absco2 = co2m1 + minorfrac(lay) * (co2m2 - co2m1)
            com1 = ka_mco(jmco,indm,ig) + fmco * &
                 (ka_mco(jmco+1,indm,ig) - ka_mco(jmco,indm,ig))
            com2 = ka_mco(jmco,indm+1,ig) + fmco * &
                 (ka_mco(jmco+1,indm+1,ig) - ka_mco(jmco,indm+1,ig))
            absco = com1 + minorfrac(lay) * (com2 - com1)
            if (specparm .lt. 0.125_r8) then
               tau_major = speccomb * &
                    (fac000 * absa(ind0,ig) + &
                    fac100 * absa(ind0+1,ig) + &
                    fac200 * absa(ind0+2,ig) + &
                    fac010 * absa(ind0+9,ig) + &
                    fac110 * absa(ind0+10,ig) + &
                    fac210 * absa(ind0+11,ig))
            else if (specparm .gt. 0.875_r8) then
               tau_major = speccomb * &
                    (fac200 * absa(ind0-1,ig) + &
                    fac100 * absa(ind0,ig) + &
                    fac000 * absa(ind0+1,ig) + &
                    fac210 * absa(ind0+8,ig) + &
                    fac110 * absa(ind0+9,ig) + &
                    fac010 * absa(ind0+10,ig))
            else
               tau_major = speccomb * &
                    (fac000 * absa(ind0,ig) + &
                    fac100 * absa(ind0+1,ig) + &
                    fac010 * absa(ind0+9,ig) + &
                    fac110 * absa(ind0+10,ig))
            endif
            if (specparm1 .lt. 0.125_r8) then
               tau_major1 = speccomb1 * &
                    (fac001 * absa(ind1,ig) + &
                    fac101 * absa(ind1+1,ig) + &
                    fac201 * absa(ind1+2,ig) + &
                    fac011 * absa(ind1+9,ig) + &
                    fac111 * absa(ind1+10,ig) + &
                    fac211 * absa(ind1+11,ig))
            else if (specparm1 .gt. 0.875_r8) then
               tau_major1 = speccomb1 * &
                    (fac201 * absa(ind1-1,ig) + &
                    fac101 * absa(ind1,ig) + &
                    fac001 * absa(ind1+1,ig) + &
                    fac211 * absa(ind1+8,ig) + &
                    fac111 * absa(ind1+9,ig) + &
                    fac011 * absa(ind1+10,ig))
            else
               tau_major1 = speccomb1 * &
                    (fac001 * absa(ind1,ig) + &
                    fac101 * absa(ind1+1,ig) + &
                    fac011 * absa(ind1+9,ig) + &
                    fac111 * absa(ind1+10,ig))
            endif
            taug(lay,ngs12+ig) = tau_major + tau_major1 &
                 + tauself + taufor &
                 + adjcolco2*absco2 &
                 + colco(lay)*absco
            fracs(lay,ngs12+ig) = fracrefa(ig,jpl) + fpl * &
                 (fracrefa(ig,jpl+1)-fracrefa(ig,jpl))
         enddo
      enddo
                ! Upper atmosphere loop
      do lay = laytrop+1, nlayers
         indm = indminor(lay)
         do ig = 1, ng13
            abso3 = kb_mo3(indm,ig) + minorfrac(lay) * &
                 (kb_mo3(indm+1,ig) - kb_mo3(indm,ig))
            taug(lay,ngs12+ig) = colo3(lay)*abso3
            fracs(lay,ngs12+ig) =  fracrefb(ig)
         enddo
      enddo
            END SUBROUTINE taugb13
            !----------------------------------------------------------------------------

            SUBROUTINE taugb14()
                !----------------------------------------------------------------------------
                !
                !     band 14:  2250-2380 cm-1 (low - co2; high - co2)
                !----------------------------------------------------------------------------
                ! ------- Modules -------
                USE parrrtm, ONLY: ng14
                USE parrrtm, ONLY: ngs13
                USE rrlw_kg14, ONLY: selfref
                USE rrlw_kg14, ONLY: forref
                USE rrlw_kg14, ONLY: absa
                USE rrlw_kg14, ONLY: fracrefa
                USE rrlw_kg14, ONLY: absb
                USE rrlw_kg14, ONLY: fracrefb
                ! ------- Declarations -------
                ! Local
                INTEGER :: lay
                INTEGER :: ind0
                INTEGER :: ind1
                INTEGER :: inds
                INTEGER :: indf
                INTEGER :: ig
                REAL(KIND=r8) :: tauself
                REAL(KIND=r8) :: taufor
                ! Compute the optical depth by interpolating in ln(pressure) and
                ! temperature.  Below laytrop, the water vapor self-continuum
                ! and foreign continuum is interpolated (in temperature) separately.
                ! Lower atmosphere loop
      do lay = 1, laytrop
         ind0 = ((jp(lay)-1)*5+(jt(lay)-1))*nspa(14) + 1
         ind1 = (jp(lay)*5+(jt1(lay)-1))*nspa(14) + 1
         inds = indself(lay)
         indf = indfor(lay)
         do ig = 1, ng14
            tauself = selffac(lay) * (selfref(inds,ig) + selffrac(lay) * &
                 (selfref(inds+1,ig) - selfref(inds,ig)))
            taufor =  forfac(lay) * (forref(indf,ig) + forfrac(lay) * &
                 (forref(indf+1,ig) - forref(indf,ig))) 
            taug(lay,ngs13+ig) = colco2(lay) * &
                 (fac00(lay) * absa(ind0,ig) + &
                 fac10(lay) * absa(ind0+1,ig) + &
                 fac01(lay) * absa(ind1,ig) + &
                 fac11(lay) * absa(ind1+1,ig)) &
                 + tauself + taufor
            fracs(lay,ngs13+ig) = fracrefa(ig)
         enddo
      enddo
                ! Upper atmosphere loop
      do lay = laytrop+1, nlayers
         ind0 = ((jp(lay)-13)*5+(jt(lay)-1))*nspb(14) + 1
         ind1 = ((jp(lay)-12)*5+(jt1(lay)-1))*nspb(14) + 1
         do ig = 1, ng14
            taug(lay,ngs13+ig) = colco2(lay) * &
                 (fac00(lay) * absb(ind0,ig) + &
                 fac10(lay) * absb(ind0+1,ig) + &
                 fac01(lay) * absb(ind1,ig) + &
                 fac11(lay) * absb(ind1+1,ig))
            fracs(lay,ngs13+ig) = fracrefb(ig)
         enddo
      enddo
            END SUBROUTINE taugb14
            !----------------------------------------------------------------------------

            SUBROUTINE taugb15()
                !----------------------------------------------------------------------------
                !
                !     band 15:  2380-2600 cm-1 (low - n2o,co2; low minor - n2)
                !                              (high - nothing)
                !----------------------------------------------------------------------------
                ! ------- Modules -------
                USE parrrtm, ONLY: ng15
                USE parrrtm, ONLY: ngs14
                USE rrlw_ref, ONLY: chi_mls
                USE rrlw_kg15, ONLY: selfref
                USE rrlw_kg15, ONLY: forref
                USE rrlw_kg15, ONLY: ka_mn2
                USE rrlw_kg15, ONLY: absa
                USE rrlw_kg15, ONLY: fracrefa
                ! ------- Declarations -------
                ! Local
                INTEGER :: lay
                INTEGER :: ind0
                INTEGER :: ind1
                INTEGER :: inds
                INTEGER :: indf
                INTEGER :: indm
                INTEGER :: ig
                INTEGER :: js
                INTEGER :: js1
                INTEGER :: jmn2
                INTEGER :: jpl
                REAL(KIND=r8) :: speccomb
                REAL(KIND=r8) :: specparm
                REAL(KIND=r8) :: specmult
                REAL(KIND=r8) :: fs
                REAL(KIND=r8) :: speccomb1
                REAL(KIND=r8) :: specparm1
                REAL(KIND=r8) :: specmult1
                REAL(KIND=r8) :: fs1
                REAL(KIND=r8) :: speccomb_mn2
                REAL(KIND=r8) :: specparm_mn2
                REAL(KIND=r8) :: specmult_mn2
                REAL(KIND=r8) :: fmn2
                REAL(KIND=r8) :: speccomb_planck
                REAL(KIND=r8) :: specparm_planck
                REAL(KIND=r8) :: specmult_planck
                REAL(KIND=r8) :: fpl
                REAL(KIND=r8) :: p
                REAL(KIND=r8) :: p4
                REAL(KIND=r8) :: fk0
                REAL(KIND=r8) :: fk1
                REAL(KIND=r8) :: fk2
                REAL(KIND=r8) :: fac000
                REAL(KIND=r8) :: fac100
                REAL(KIND=r8) :: fac200
                REAL(KIND=r8) :: fac010
                REAL(KIND=r8) :: fac110
                REAL(KIND=r8) :: fac210
                REAL(KIND=r8) :: fac001
                REAL(KIND=r8) :: fac101
                REAL(KIND=r8) :: fac201
                REAL(KIND=r8) :: fac011
                REAL(KIND=r8) :: fac111
                REAL(KIND=r8) :: fac211
                REAL(KIND=r8) :: scalen2
                REAL(KIND=r8) :: tauself
                REAL(KIND=r8) :: taufor
                REAL(KIND=r8) :: n2m1
                REAL(KIND=r8) :: n2m2
                REAL(KIND=r8) :: taun2
                REAL(KIND=r8) :: refrat_planck_a
                REAL(KIND=r8) :: refrat_m_a
                REAL(KIND=r8) :: tau_major
                REAL(KIND=r8) :: tau_major1
                ! Minor gas mapping level :
                !     Lower - Nitrogen Continuum, P = 1053., T = 294.
                ! Calculate reference ratio to be used in calculation of Planck
                ! fraction in lower atmosphere.
                ! P = 1053. mb (Level 1)
      refrat_planck_a = chi_mls(4,1)/chi_mls(2,1)
                ! P = 1053.
      refrat_m_a = chi_mls(4,1)/chi_mls(2,1)
                ! Compute the optical depth by interpolating in ln(pressure),
                ! temperature, and appropriate species.  Below laytrop, the water
                ! vapor self-continuum and foreign continuum is interpolated
                ! (in temperature) separately.
                ! Lower atmosphere loop
      do lay = 1, laytrop
         speccomb = coln2o(lay) + rat_n2oco2(lay)*colco2(lay)
         specparm = coln2o(lay)/speccomb
         if (specparm .ge. oneminus) specparm = oneminus
         specmult = 8._r8*(specparm)
         js = 1 + int(specmult)
         fs = mod(specmult,1.0_r8)
         speccomb1 = coln2o(lay) + rat_n2oco2_1(lay)*colco2(lay)
         specparm1 = coln2o(lay)/speccomb1
         if (specparm1 .ge. oneminus) specparm1 = oneminus
         specmult1 = 8._r8*(specparm1)
         js1 = 1 + int(specmult1)
         fs1 = mod(specmult1,1.0_r8)
         speccomb_mn2 = coln2o(lay) + refrat_m_a*colco2(lay)
         specparm_mn2 = coln2o(lay)/speccomb_mn2
         if (specparm_mn2 .ge. oneminus) specparm_mn2 = oneminus
         specmult_mn2 = 8._r8*specparm_mn2
         jmn2 = 1 + int(specmult_mn2)
         fmn2 = mod(specmult_mn2,1.0_r8)
         speccomb_planck = coln2o(lay)+refrat_planck_a*colco2(lay)
         specparm_planck = coln2o(lay)/speccomb_planck
         if (specparm_planck .ge. oneminus) specparm_planck=oneminus
         specmult_planck = 8._r8*specparm_planck
         jpl= 1 + int(specmult_planck)
         fpl = mod(specmult_planck,1.0_r8)
         ind0 = ((jp(lay)-1)*5+(jt(lay)-1))*nspa(15) + js
         ind1 = (jp(lay)*5+(jt1(lay)-1))*nspa(15) + js1
         inds = indself(lay)
         indf = indfor(lay)
         indm = indminor(lay)
         scalen2 = colbrd(lay)*scaleminor(lay)
         if (specparm .lt. 0.125_r8) then
            p = fs - 1
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac000 = fk0*fac00(lay)
            fac100 = fk1*fac00(lay)
            fac200 = fk2*fac00(lay)
            fac010 = fk0*fac10(lay)
            fac110 = fk1*fac10(lay)
            fac210 = fk2*fac10(lay)
         else if (specparm .gt. 0.875_r8) then
            p = -fs 
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac000 = fk0*fac00(lay)
            fac100 = fk1*fac00(lay)
            fac200 = fk2*fac00(lay)
            fac010 = fk0*fac10(lay)
            fac110 = fk1*fac10(lay)
            fac210 = fk2*fac10(lay)
         else
            fac000 = (1._r8 - fs) * fac00(lay)
            fac010 = (1._r8 - fs) * fac10(lay)
            fac100 = fs * fac00(lay)
            fac110 = fs * fac10(lay)
         endif
         if (specparm1 .lt. 0.125_r8) then
            p = fs1 - 1
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac001 = fk0*fac01(lay)
            fac101 = fk1*fac01(lay)
            fac201 = fk2*fac01(lay)
            fac011 = fk0*fac11(lay)
            fac111 = fk1*fac11(lay)
            fac211 = fk2*fac11(lay)
         else if (specparm1 .gt. 0.875_r8) then
            p = -fs1 
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac001 = fk0*fac01(lay)
            fac101 = fk1*fac01(lay)
            fac201 = fk2*fac01(lay)
            fac011 = fk0*fac11(lay)
            fac111 = fk1*fac11(lay)
            fac211 = fk2*fac11(lay)
         else
            fac001 = (1._r8 - fs1) * fac01(lay)
            fac011 = (1._r8 - fs1) * fac11(lay)
            fac101 = fs1 * fac01(lay)
            fac111 = fs1 * fac11(lay)
         endif
         do ig = 1, ng15
            tauself = selffac(lay)* (selfref(inds,ig) + selffrac(lay) * &
                 (selfref(inds+1,ig) - selfref(inds,ig)))
            taufor =  forfac(lay) * (forref(indf,ig) + forfrac(lay) * &
                 (forref(indf+1,ig) - forref(indf,ig))) 
            n2m1 = ka_mn2(jmn2,indm,ig) + fmn2 * &
                 (ka_mn2(jmn2+1,indm,ig) - ka_mn2(jmn2,indm,ig))
            n2m2 = ka_mn2(jmn2,indm+1,ig) + fmn2 * &
                 (ka_mn2(jmn2+1,indm+1,ig) - ka_mn2(jmn2,indm+1,ig))
            taun2 = scalen2 * (n2m1 + minorfrac(lay) * (n2m2 - n2m1))
            if (specparm .lt. 0.125_r8) then
               tau_major = speccomb * &
                    (fac000 * absa(ind0,ig) + &
                    fac100 * absa(ind0+1,ig) + &
                    fac200 * absa(ind0+2,ig) + &
                    fac010 * absa(ind0+9,ig) + &
                    fac110 * absa(ind0+10,ig) + &
                    fac210 * absa(ind0+11,ig))
            else if (specparm .gt. 0.875_r8) then
               tau_major = speccomb * &
                    (fac200 * absa(ind0-1,ig) + &
                    fac100 * absa(ind0,ig) + &
                    fac000 * absa(ind0+1,ig) + &
                    fac210 * absa(ind0+8,ig) + &
                    fac110 * absa(ind0+9,ig) + &
                    fac010 * absa(ind0+10,ig))
            else
               tau_major = speccomb * &
                    (fac000 * absa(ind0,ig) + &
                    fac100 * absa(ind0+1,ig) + &
                    fac010 * absa(ind0+9,ig) + &
                    fac110 * absa(ind0+10,ig))
            endif 
            if (specparm1 .lt. 0.125_r8) then
               tau_major1 = speccomb1 * &
                    (fac001 * absa(ind1,ig) + &
                    fac101 * absa(ind1+1,ig) + &
                    fac201 * absa(ind1+2,ig) + &
                    fac011 * absa(ind1+9,ig) + &
                    fac111 * absa(ind1+10,ig) + &
                    fac211 * absa(ind1+11,ig))
            else if (specparm1 .gt. 0.875_r8) then
               tau_major1 = speccomb1 * &
                    (fac201 * absa(ind1-1,ig) + &
                    fac101 * absa(ind1,ig) + &
                    fac001 * absa(ind1+1,ig) + &
                    fac211 * absa(ind1+8,ig) + &
                    fac111 * absa(ind1+9,ig) + &
                    fac011 * absa(ind1+10,ig))
            else
               tau_major1 = speccomb1 * &
                    (fac001 * absa(ind1,ig) + &
                    fac101 * absa(ind1+1,ig) + &
                    fac011 * absa(ind1+9,ig) + &
                    fac111 * absa(ind1+10,ig))
            endif
            taug(lay,ngs14+ig) = tau_major + tau_major1 &
                 + tauself + taufor &
                 + taun2
            fracs(lay,ngs14+ig) = fracrefa(ig,jpl) + fpl * &
                 (fracrefa(ig,jpl+1)-fracrefa(ig,jpl))
         enddo
      enddo
                ! Upper atmosphere loop
      do lay = laytrop+1, nlayers
         do ig = 1, ng15
            taug(lay,ngs14+ig) = 0.0_r8
            fracs(lay,ngs14+ig) = 0.0_r8
         enddo
      enddo
            END SUBROUTINE taugb15
            !----------------------------------------------------------------------------

            SUBROUTINE taugb16()
                !----------------------------------------------------------------------------
                !
                !     band 16:  2600-3250 cm-1 (low key- h2o,ch4; high key - ch4)
                !----------------------------------------------------------------------------
                ! ------- Modules -------
                USE parrrtm, ONLY: ng16
                USE parrrtm, ONLY: ngs15
                USE rrlw_ref, ONLY: chi_mls
                USE rrlw_kg16, ONLY: selfref
                USE rrlw_kg16, ONLY: forref
                USE rrlw_kg16, ONLY: absa
                USE rrlw_kg16, ONLY: fracrefa
                USE rrlw_kg16, ONLY: absb
                USE rrlw_kg16, ONLY: fracrefb
                ! ------- Declarations -------
                ! Local
                INTEGER :: lay
                INTEGER :: ind0
                INTEGER :: ind1
                INTEGER :: inds
                INTEGER :: indf
                INTEGER :: ig
                INTEGER :: js
                INTEGER :: js1
                INTEGER :: jpl
                REAL(KIND=r8) :: speccomb
                REAL(KIND=r8) :: specparm
                REAL(KIND=r8) :: specmult
                REAL(KIND=r8) :: fs
                REAL(KIND=r8) :: speccomb1
                REAL(KIND=r8) :: specparm1
                REAL(KIND=r8) :: specmult1
                REAL(KIND=r8) :: fs1
                REAL(KIND=r8) :: speccomb_planck
                REAL(KIND=r8) :: specparm_planck
                REAL(KIND=r8) :: specmult_planck
                REAL(KIND=r8) :: fpl
                REAL(KIND=r8) :: p
                REAL(KIND=r8) :: p4
                REAL(KIND=r8) :: fk0
                REAL(KIND=r8) :: fk1
                REAL(KIND=r8) :: fk2
                REAL(KIND=r8) :: fac000
                REAL(KIND=r8) :: fac100
                REAL(KIND=r8) :: fac200
                REAL(KIND=r8) :: fac010
                REAL(KIND=r8) :: fac110
                REAL(KIND=r8) :: fac210
                REAL(KIND=r8) :: fac001
                REAL(KIND=r8) :: fac101
                REAL(KIND=r8) :: fac201
                REAL(KIND=r8) :: fac011
                REAL(KIND=r8) :: fac111
                REAL(KIND=r8) :: fac211
                REAL(KIND=r8) :: tauself
                REAL(KIND=r8) :: taufor
                REAL(KIND=r8) :: refrat_planck_a
                REAL(KIND=r8) :: tau_major
                REAL(KIND=r8) :: tau_major1
                ! Calculate reference ratio to be used in calculation of Planck
                ! fraction in lower atmosphere.
                ! P = 387. mb (Level 6)
      refrat_planck_a = chi_mls(1,6)/chi_mls(6,6)
                ! Compute the optical depth by interpolating in ln(pressure),
                ! temperature,and appropriate species.  Below laytrop, the water
                ! vapor self-continuum and foreign continuum is interpolated
                ! (in temperature) separately.
                ! Lower atmosphere loop
      do lay = 1, laytrop
         speccomb = colh2o(lay) + rat_h2och4(lay)*colch4(lay)
         specparm = colh2o(lay)/speccomb
         if (specparm .ge. oneminus) specparm = oneminus
         specmult = 8._r8*(specparm)
         js = 1 + int(specmult)
         fs = mod(specmult,1.0_r8)
         speccomb1 = colh2o(lay) + rat_h2och4_1(lay)*colch4(lay)
         specparm1 = colh2o(lay)/speccomb1
         if (specparm1 .ge. oneminus) specparm1 = oneminus
         specmult1 = 8._r8*(specparm1)
         js1 = 1 + int(specmult1)
         fs1 = mod(specmult1,1.0_r8)
         speccomb_planck = colh2o(lay)+refrat_planck_a*colch4(lay)
         specparm_planck = colh2o(lay)/speccomb_planck
         if (specparm_planck .ge. oneminus) specparm_planck=oneminus
         specmult_planck = 8._r8*specparm_planck
         jpl= 1 + int(specmult_planck)
         fpl = mod(specmult_planck,1.0_r8)
         ind0 = ((jp(lay)-1)*5+(jt(lay)-1))*nspa(16) + js
         ind1 = (jp(lay)*5+(jt1(lay)-1))*nspa(16) + js1
         inds = indself(lay)
         indf = indfor(lay)
         if (specparm .lt. 0.125_r8) then
            p = fs - 1
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac000 = fk0*fac00(lay)
            fac100 = fk1*fac00(lay)
            fac200 = fk2*fac00(lay)
            fac010 = fk0*fac10(lay)
            fac110 = fk1*fac10(lay)
            fac210 = fk2*fac10(lay)
         else if (specparm .gt. 0.875_r8) then
            p = -fs 
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac000 = fk0*fac00(lay)
            fac100 = fk1*fac00(lay)
            fac200 = fk2*fac00(lay)
            fac010 = fk0*fac10(lay)
            fac110 = fk1*fac10(lay)
            fac210 = fk2*fac10(lay)
         else
            fac000 = (1._r8 - fs) * fac00(lay)
            fac010 = (1._r8 - fs) * fac10(lay)
            fac100 = fs * fac00(lay)
            fac110 = fs * fac10(lay)
         endif
         if (specparm1 .lt. 0.125_r8) then
            p = fs1 - 1
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac001 = fk0*fac01(lay)
            fac101 = fk1*fac01(lay)
            fac201 = fk2*fac01(lay)
            fac011 = fk0*fac11(lay)
            fac111 = fk1*fac11(lay)
            fac211 = fk2*fac11(lay)
         else if (specparm1 .gt. 0.875_r8) then
            p = -fs1 
            p4 = p**4
            fk0 = p4
            fk1 = 1 - p - 2.0_r8*p4
            fk2 = p + p4
            fac001 = fk0*fac01(lay)
            fac101 = fk1*fac01(lay)
            fac201 = fk2*fac01(lay)
            fac011 = fk0*fac11(lay)
            fac111 = fk1*fac11(lay)
            fac211 = fk2*fac11(lay)
         else
            fac001 = (1._r8 - fs1) * fac01(lay)
            fac011 = (1._r8 - fs1) * fac11(lay)
            fac101 = fs1 * fac01(lay)
            fac111 = fs1 * fac11(lay)
         endif
         do ig = 1, ng16
            tauself = selffac(lay)* (selfref(inds,ig) + selffrac(lay) * &
                 (selfref(inds+1,ig) - selfref(inds,ig)))
            taufor =  forfac(lay) * (forref(indf,ig) + forfrac(lay) * &
                 (forref(indf+1,ig) - forref(indf,ig))) 
            if (specparm .lt. 0.125_r8) then
               tau_major = speccomb * &
                    (fac000 * absa(ind0,ig) + &
                    fac100 * absa(ind0+1,ig) + &
                    fac200 * absa(ind0+2,ig) + &
                    fac010 * absa(ind0+9,ig) + &
                    fac110 * absa(ind0+10,ig) + &
                    fac210 * absa(ind0+11,ig))
            else if (specparm .gt. 0.875_r8) then
               tau_major = speccomb * &
                    (fac200 * absa(ind0-1,ig) + &
                    fac100 * absa(ind0,ig) + &
                    fac000 * absa(ind0+1,ig) + &
                    fac210 * absa(ind0+8,ig) + &
                    fac110 * absa(ind0+9,ig) + &
                    fac010 * absa(ind0+10,ig))
            else
               tau_major = speccomb * &
                    (fac000 * absa(ind0,ig) + &
                    fac100 * absa(ind0+1,ig) + &
                    fac010 * absa(ind0+9,ig) + &
                    fac110 * absa(ind0+10,ig))
            endif
            if (specparm1 .lt. 0.125_r8) then
               tau_major1 = speccomb1 * &
                    (fac001 * absa(ind1,ig) + &
                    fac101 * absa(ind1+1,ig) + &
                    fac201 * absa(ind1+2,ig) + &
                    fac011 * absa(ind1+9,ig) + &
                    fac111 * absa(ind1+10,ig) + &
                    fac211 * absa(ind1+11,ig))
            else if (specparm1 .gt. 0.875_r8) then
               tau_major1 = speccomb1 * &
                    (fac201 * absa(ind1-1,ig) + &
                    fac101 * absa(ind1,ig) + &
                    fac001 * absa(ind1+1,ig) + &
                    fac211 * absa(ind1+8,ig) + &
                    fac111 * absa(ind1+9,ig) + &
                    fac011 * absa(ind1+10,ig))
            else
               tau_major1 = speccomb1 * &
                    (fac001 * absa(ind1,ig) + &
                    fac101 * absa(ind1+1,ig) + &
                    fac011 * absa(ind1+9,ig) + &
                    fac111 * absa(ind1+10,ig))
            endif
            taug(lay,ngs15+ig) = tau_major + tau_major1 &
                 + tauself + taufor
            fracs(lay,ngs15+ig) = fracrefa(ig,jpl) + fpl * &
                 (fracrefa(ig,jpl+1)-fracrefa(ig,jpl))
         enddo
      enddo
                ! Upper atmosphere loop
      do lay = laytrop+1, nlayers
         ind0 = ((jp(lay)-13)*5+(jt(lay)-1))*nspb(16) + 1
         ind1 = ((jp(lay)-12)*5+(jt1(lay)-1))*nspb(16) + 1
         do ig = 1, ng16
            taug(lay,ngs15+ig) = colch4(lay) * &
                 (fac00(lay) * absb(ind0,ig) + &
                 fac10(lay) * absb(ind0+1,ig) + &
                 fac01(lay) * absb(ind1,ig) + &
                 fac11(lay) * absb(ind1+1,ig))
            fracs(lay,ngs15+ig) = fracrefb(ig)
         enddo
      enddo
            END SUBROUTINE taugb16
        END SUBROUTINE taumol
    END MODULE rrtmg_lw_taumol
