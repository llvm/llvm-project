
! KGEN-generated Fortran source file
!
! Filename    : rrtmg_sw_setcoef.f90
! Generated at: 2015-07-27 00:47:03
! KGEN version: 0.4.13



    MODULE rrtmg_sw_setcoef
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
        USE rrsw_ref, ONLY: preflog
        USE rrsw_ref, ONLY: tref
        IMPLICIT NONE
        CONTAINS

        ! write subroutines
        ! No subroutines
        ! No module extern variables
        !----------------------------------------------------------------------------

        SUBROUTINE setcoef_sw(ncol, nlayers, vec_pavel, vec_tavel, vec_pz, vec_tz, vec_tbound, vec_coldry, vec_wkl, vec_laytrop, &
        vec_layswtch, vec_laylow, vec_jp, vec_jt, vec_jt1, vec_co2mult, vec_colch4, vec_colco2, vec_colh2o, vec_colmol, &
        vec_coln2o, vec_colo2, vec_colo3, vec_fac00, vec_fac01, vec_fac10, vec_fac11, vec_selffac, vec_selffrac, vec_indself, &
        vec_forfac, vec_forfrac, vec_indfor)
            !----------------------------------------------------------------------------
            !
            ! Purpose:  For a given atmosphere, calculate the indices and
            ! fractions related to the pressure and temperature interpolations.
            ! Modifications:
            ! Original: J. Delamere, AER, Inc. (version 2.5, 02/04/01)
            ! Revised: Rewritten and adapted to ECMWF F90, JJMorcrette 030224
            ! Revised: For uniform rrtmg formatting, MJIacono, Jul 2006
            ! ------ Declarations -------
            ! ----- Input -----
            INTEGER, intent(in) :: ncol ! total number of columns
            INTEGER, intent(in) :: nlayers ! total number of layers
            REAL(KIND=r8), intent(in) :: vec_pavel(:,:) ! layer pressures (mb)
            !    Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(in) :: vec_tavel(:,:) ! layer temperatures (K)
            !    Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(in) :: vec_pz(:,0:) ! level (interface) pressures (hPa, mb)
            !    Dimensions: (ncol,0:nlayers)
            REAL(KIND=r8), intent(in) :: vec_tz(:,0:) ! level (interface) temperatures (K)
            !    Dimensions: (ncol,0:nlayers)
            REAL(KIND=r8), intent(in) :: vec_tbound(:) ! surface temperature (K)
            REAL(KIND=r8), intent(in) :: vec_coldry(:,:) ! dry air column density (mol/cm2)
            !    Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(in) :: vec_wkl(:,:,:) ! molecular amounts (mol/cm-2)
            !    Dimensions: (mxmol,ncol,nlayers)
            ! ----- Output -----
            INTEGER, intent(out) :: vec_laytrop(:) ! tropopause layer index
            INTEGER, intent(out) :: vec_layswtch(:) !
            INTEGER, intent(out) :: vec_laylow(:) !
            INTEGER, intent(out) :: vec_jp(:,:) !
            !    Dimensions: (ncol,nlayers)
            INTEGER, intent(out) :: vec_jt(:,:) !
            !    Dimensions: (ncol,nlayers)
            INTEGER, intent(out) :: vec_jt1(:,:) !
            !    Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(out) :: vec_colh2o(:,:) ! column amount (h2o)
            !    Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(out) :: vec_colco2(:,:) ! column amount (co2)
            !    Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(out) :: vec_colo3(:,:) ! column amount (o3)
            !    Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(out) :: vec_coln2o(:,:) ! column amount (n2o)
            !    Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(out) :: vec_colch4(:,:) ! column amount (ch4)
            !    Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(out) :: vec_colo2(:,:) ! column amount (o2)
            !    Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(out) :: vec_colmol(:,:) !
            !    Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(out) :: vec_co2mult(:,:) !
            !    Dimensions: (ncol,nlayers)
            INTEGER, intent(out) :: vec_indself(:,:)
            !    Dimensions: (ncol,nlayers)
            INTEGER, intent(out) :: vec_indfor(:,:)
            !    Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(out) :: vec_selffac(:,:)
            !    Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(out) :: vec_selffrac(:,:)
            !    Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(out) :: vec_forfac(:,:)
            !    Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(out) :: vec_forfrac(:,:)
            !    Dimensions: (ncol,nlayers)
            REAL(KIND=r8), intent(out) :: vec_fac11(:,:)
            REAL(KIND=r8), intent(out) :: vec_fac10(:,:)
            REAL(KIND=r8), intent(out) :: vec_fac00(:,:)
            REAL(KIND=r8), intent(out) :: vec_fac01(:,:) !
            !    Dimensions: (ncol,nlayers)
            ! ----- Local -----
            INTEGER :: indbound
            INTEGER :: indlev0
            INTEGER :: lay
            INTEGER :: jp1
            INTEGER :: iplon
            REAL(KIND=r8) :: stpfac
            REAL(KIND=r8) :: tbndfrac
            REAL(KIND=r8) :: t0frac
            REAL(KIND=r8) :: plog
            REAL(KIND=r8) :: fp
            REAL(KIND=r8) :: ft
            REAL(KIND=r8) :: ft1
            REAL(KIND=r8) :: water
            REAL(KIND=r8) :: scalefac
            REAL(KIND=r8) :: factor
            REAL(KIND=r8) :: co2reg
            REAL(KIND=r8) :: compfp
            ! Initializations
      stpfac = 296._r8/1013._r8
            !Begin column loop
    do iplon=1, ncol
      vec_laytrop(iplon)  = 0
      vec_layswtch(iplon) = 0
      vec_laylow(iplon)   = 0
      indbound = vec_tbound(iplon) - 159._r8
      tbndfrac = vec_tbound(iplon) - int(vec_tbound(iplon))
      indlev0  = vec_tz(iplon,0) - 159._r8
      t0frac   = vec_tz(iplon,0) - int(vec_tz(iplon,0))
                ! Begin layer loop
      do lay = 1, nlayers
                    ! Find the two reference pressures on either side of the
                    ! layer pressure.  Store them in JP and JP1.  Store in FP the
                    ! fraction of the difference (in ln(pressure)) between these
                    ! two values that the layer pressure lies.
         plog = log(vec_pavel(iplon,lay))
         vec_jp(iplon,lay) = int(36._r8 - 5*(plog+0.04_r8))
         if (vec_jp(iplon,lay) .lt. 1) then
            vec_jp(iplon,lay) = 1
         elseif (vec_jp(iplon,lay) .gt. 58) then
            vec_jp(iplon,lay) = 58
         endif
         jp1 = vec_jp(iplon,lay) + 1
         fp = 5._r8 * (preflog(vec_jp(iplon,lay)) - plog)
                    ! Determine, for each reference pressure (JP and JP1), which
                    ! reference temperature (these are different for each
                    ! reference pressure) is nearest the layer temperature but does
                    ! not exceed it.  Store these indices in JT and JT1, resp.
                    ! Store in FT (resp. FT1) the fraction of the way between JT
                    ! (JT1) and the next highest reference temperature that the
                    ! layer temperature falls.
         vec_jt(iplon,lay) = int(3._r8 + (vec_tavel(iplon,lay)-tref(vec_jp(iplon,lay)))/15._r8)
         if (vec_jt(iplon,lay) .lt. 1) then
            vec_jt(iplon,lay) = 1
         elseif (vec_jt(iplon,lay) .gt. 4) then
            vec_jt(iplon,lay) = 4
         endif
         ft = ((vec_tavel(iplon,lay)-tref(vec_jp(iplon,lay)))/15._r8) - float(vec_jt(iplon,lay)-3)
         vec_jt1(iplon,lay) = int(3._r8 + (vec_tavel(iplon,lay)-tref(jp1))/15._r8)
         if (vec_jt1(iplon,lay) .lt. 1) then
            vec_jt1(iplon,lay) = 1
         elseif (vec_jt1(iplon,lay) .gt. 4) then
            vec_jt1(iplon,lay) = 4
         endif
         ft1 = ((vec_tavel(iplon,lay)-tref(jp1))/15._r8) - float(vec_jt1(iplon,lay)-3)
         water = vec_wkl(iplon,1,lay)/vec_coldry(iplon,lay)
         scalefac = vec_pavel(iplon,lay) * stpfac / vec_tavel(iplon,lay)
                    ! If the pressure is less than ~100mb, perform a different
                    ! set of species interpolations.
         if (plog .le. 4.56_r8) go to 5300
         vec_laytrop(iplon) =  vec_laytrop(iplon) + 1
         if (plog .ge. 6.62_r8) vec_laylow(iplon) = vec_laylow(iplon) + 1
                    ! Set up factors needed to separately include the water vapor
                    ! foreign-continuum in the calculation of absorption coefficient.
         vec_forfac(iplon,lay) = scalefac / (1.+water)
         factor = (332.0_r8-vec_tavel(iplon,lay))/36.0_r8
         vec_indfor(iplon,lay) = min(2, max(1, int(factor)))
         vec_forfrac(iplon,lay) = factor - float(vec_indfor(iplon,lay))
                    ! Set up factors needed to separately include the water vapor
                    ! self-continuum in the calculation of absorption coefficient.
         vec_selffac(iplon,lay) = water * vec_forfac(iplon,lay)
         factor = (vec_tavel(iplon,lay)-188.0_r8)/7.2_r8
         vec_indself(iplon,lay) = min(9, max(1, int(factor)-7))
         vec_selffrac(iplon,lay) = factor - float(vec_indself(iplon,lay) + 7)
                    ! Calculate needed column amounts.
         vec_colh2o(iplon,lay) = 1.e-20_r8 * vec_wkl(iplon,1,lay)
         vec_colco2(iplon,lay) = 1.e-20_r8 * vec_wkl(iplon,2,lay)
         vec_colo3(iplon,lay) = 1.e-20_r8 * vec_wkl(iplon,3,lay)
                    !           colo3(lay) = 0._r8
                    !           colo3(lay) = colo3(lay)/1.16_r8
         vec_coln2o(iplon,lay) = 1.e-20_r8 * vec_wkl(iplon,4,lay)
         vec_colch4(iplon,lay) = 1.e-20_r8 * vec_wkl(iplon,6,lay)
         vec_colo2(iplon,lay) = 1.e-20_r8 * vec_wkl(iplon,7,lay)
         vec_colmol(iplon,lay) = 1.e-20_r8 * vec_coldry(iplon,lay) + vec_colh2o(iplon,lay)
                    !           vec_colco2(lay) = 0._r8
                    !           colo3(lay) = 0._r8
                    !           coln2o(lay) = 0._r8
                    !           colch4(lay) = 0._r8
                    !           colo2(lay) = 0._r8
                    !           colmol(lay) = 0._r8
         if (vec_colco2(iplon,lay) .eq. 0._r8) vec_colco2(iplon,lay) = 1.e-32_r8 * vec_coldry(iplon,lay)
         if (vec_coln2o(iplon,lay) .eq. 0._r8) vec_coln2o(iplon,lay) = 1.e-32_r8 * vec_coldry(iplon,lay)
         if (vec_colch4(iplon,lay) .eq. 0._r8) vec_colch4(iplon,lay) = 1.e-32_r8 * vec_coldry(iplon,lay)
         if (vec_colo2(iplon,lay) .eq. 0._r8) vec_colo2(iplon,lay) = 1.e-32_r8 * vec_coldry(iplon,lay)
                    ! Using E = 1334.2 cm-1.
         co2reg = 3.55e-24_r8 * vec_coldry(iplon,lay)
         vec_co2mult(iplon,lay)= (vec_colco2(iplon,lay) - co2reg) * &
               272.63_r8*exp(-1919.4_r8/vec_tavel(iplon,lay))/(8.7604e-4_r8*vec_tavel(iplon,lay))
         goto 5400
                    ! Above vec_laytrop.
 5300    continue
                    ! Set up factors needed to separately include the water vapor
                    ! foreign-continuum in the calculation of absorption coefficient.
         vec_forfac(iplon,lay) = scalefac / (1.+water)
         factor = (vec_tavel(iplon,lay)-188.0_r8)/36.0_r8
         vec_indfor(iplon,lay) = 3
         vec_forfrac(iplon,lay) = factor - 1.0_r8
                    ! Calculate needed column amounts.
         vec_colh2o(iplon,lay) = 1.e-20_r8 * vec_wkl(iplon,1,lay)
         vec_colco2(iplon,lay) = 1.e-20_r8 * vec_wkl(iplon,2,lay)
         vec_colo3(iplon,lay)  = 1.e-20_r8 * vec_wkl(iplon,3,lay)
         vec_coln2o(iplon,lay) = 1.e-20_r8 * vec_wkl(iplon,4,lay)
         vec_colch4(iplon,lay) = 1.e-20_r8 * vec_wkl(iplon,6,lay)
         vec_colo2(iplon,lay)  = 1.e-20_r8 * vec_wkl(iplon,7,lay)
         vec_colmol(iplon,lay) = 1.e-20_r8 * vec_coldry(iplon,lay) + vec_colh2o(iplon,lay)
         if (vec_colco2(iplon,lay) .eq. 0._r8) vec_colco2(iplon,lay) = 1.e-32_r8 * vec_coldry(iplon,lay)
         if (vec_coln2o(iplon,lay) .eq. 0._r8) vec_coln2o(iplon,lay) = 1.e-32_r8 * vec_coldry(iplon,lay)
         if (vec_colch4(iplon,lay) .eq. 0._r8) vec_colch4(iplon,lay) = 1.e-32_r8 * vec_coldry(iplon,lay)
         if (vec_colo2(iplon,lay)  .eq. 0._r8) vec_colo2(iplon,lay)  = 1.e-32_r8 * vec_coldry(iplon,lay)
         co2reg = 3.55e-24_r8 * vec_coldry(iplon,lay)
         vec_co2mult(iplon,lay)= (vec_colco2(iplon,lay) - co2reg) * &
               272.63_r8*exp(-1919.4_r8/vec_tavel(iplon,lay))/(8.7604e-4_r8*vec_tavel(iplon,lay))
         vec_selffac(iplon,lay) = 0._r8
         vec_selffrac(iplon,lay)= 0._r8
         vec_indself(iplon,lay) = 0
 5400    continue
                    ! We have now isolated the layer ln pressure and temperature,
                    ! between two reference pressures and two reference temperatures
                    ! (for each reference pressure).  We multiply the pressure
                    ! fraction FP with the appropriate temperature fractions to get
                    ! the factors that will be needed for the interpolation that yields
                    ! the optical depths (performed in routines TAUGBn for band n).
         compfp = 1._r8 - fp
         vec_fac10(iplon,lay) = compfp * ft
         vec_fac00(iplon,lay) = compfp * (1._r8 - ft)
         vec_fac11(iplon,lay) = fp * ft1
         vec_fac01(iplon,lay) = fp * (1._r8 - ft1)
                    ! End layer loop
      enddo
                !End column loop
    enddo
        END SUBROUTINE setcoef_sw
        !***************************************************************************

    END MODULE rrtmg_sw_setcoef
