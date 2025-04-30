
! KGEN-generated Fortran source file
!
! Filename    : mo_rrtm_coeffs.f90
! Generated at: 2015-02-19 15:30:32
! KGEN version: 0.4.4



    MODULE mo_rrtm_coeffs
        USE mo_kind, ONLY: wp
        USE mo_rrtm_params, ONLY: preflog
        USE mo_rrtm_params, ONLY: tref
        USE rrlw_planck, ONLY: chi_mls
        IMPLICIT NONE
        REAL(KIND=wp), parameter :: stpfac  = 296._wp/1013._wp
        CONTAINS

        ! read subroutines
        ! --------------------------------------------------------------------------------------------

        SUBROUTINE lrtm_coeffs(kproma, kbdim, klev, play, tlay, coldry, wkl, wbroad, laytrop, jp, jt, jt1, colh2o, colco2, colo3, &
        coln2o, colco, colch4, colo2, colbrd, fac00, fac01, fac10, fac11, rat_h2oco2, rat_h2oco2_1, rat_h2oo3, rat_h2oo3_1, &
        rat_h2on2o, rat_h2on2o_1, rat_h2och4, rat_h2och4_1, rat_n2oco2, rat_n2oco2_1, rat_o3co2, rat_o3co2_1, selffac, selffrac, &
        indself, forfac, forfrac, indfor, minorfrac, scaleminor, scaleminorn2, indminor)
            INTEGER, intent(in) :: kbdim
            INTEGER, intent(in) :: klev
            INTEGER, intent(in) :: kproma
            ! number of columns
            ! maximum number of column as first dim is declared in calling (sub)prog.
            ! total number of layers
            REAL(KIND=wp), intent(in) :: wkl(:,:,:)
            REAL(KIND=wp), intent(in) :: play(kbdim,klev)
            REAL(KIND=wp), intent(in) :: tlay(kbdim,klev)
            REAL(KIND=wp), intent(in) :: coldry(kbdim,klev)
            REAL(KIND=wp), intent(in) :: wbroad(kbdim,klev)
            ! layer pressures (mb)
            ! layer temperatures (K)
            ! dry air column density (mol/cm2)
            ! broadening gas column density (mol/cm2)
            !< molecular amounts (mol/cm-2) (mxmol,klev)
            !
            ! Output Dimensions kproma, klev unless otherwise specified
            !
            INTEGER, intent(out) :: laytrop(kbdim)
            INTEGER, intent(out) :: jp(kbdim,klev)
            INTEGER, intent(out) :: jt(kbdim,klev)
            INTEGER, intent(out) :: jt1(kbdim,klev)
            INTEGER, intent(out) :: indfor(kbdim,klev)
            INTEGER, intent(out) :: indself(kbdim,klev)
            INTEGER, intent(out) :: indminor(kbdim,klev)
            !< tropopause layer index
            !
            !
            !
            !
            !
            !
            REAL(KIND=wp), intent(out) :: forfac(kbdim,klev)
            REAL(KIND=wp), intent(out) :: colch4(kbdim,klev)
            REAL(KIND=wp), intent(out) :: colco2(kbdim,klev)
            REAL(KIND=wp), intent(out) :: colh2o(kbdim,klev)
            REAL(KIND=wp), intent(out) :: coln2o(kbdim,klev)
            REAL(KIND=wp), intent(out) :: forfrac(kbdim,klev)
            REAL(KIND=wp), intent(out) :: selffrac(kbdim,klev)
            REAL(KIND=wp), intent(out) :: colo3(kbdim,klev)
            REAL(KIND=wp), intent(out) :: fac00(kbdim,klev)
            REAL(KIND=wp), intent(out) :: colo2(kbdim,klev)
            REAL(KIND=wp), intent(out) :: fac01(kbdim,klev)
            REAL(KIND=wp), intent(out) :: fac10(kbdim,klev)
            REAL(KIND=wp), intent(out) :: fac11(kbdim,klev)
            REAL(KIND=wp), intent(out) :: selffac(kbdim,klev)
            REAL(KIND=wp), intent(out) :: colbrd(kbdim,klev)
            REAL(KIND=wp), intent(out) :: colco(kbdim,klev)
            REAL(KIND=wp), intent(out) :: rat_h2oco2(kbdim,klev)
            REAL(KIND=wp), intent(out) :: rat_h2oco2_1(kbdim,klev)
            REAL(KIND=wp), intent(out) :: rat_h2oo3(kbdim,klev)
            REAL(KIND=wp), intent(out) :: rat_h2oo3_1(kbdim,klev)
            REAL(KIND=wp), intent(out) :: rat_h2on2o(kbdim,klev)
            REAL(KIND=wp), intent(out) :: rat_h2on2o_1(kbdim,klev)
            REAL(KIND=wp), intent(out) :: rat_h2och4(kbdim,klev)
            REAL(KIND=wp), intent(out) :: rat_h2och4_1(kbdim,klev)
            REAL(KIND=wp), intent(out) :: rat_n2oco2(kbdim,klev)
            REAL(KIND=wp), intent(out) :: rat_n2oco2_1(kbdim,klev)
            REAL(KIND=wp), intent(out) :: rat_o3co2(kbdim,klev)
            REAL(KIND=wp), intent(out) :: rat_o3co2_1(kbdim,klev)
            REAL(KIND=wp), intent(out) :: scaleminor(kbdim,klev)
            REAL(KIND=wp), intent(out) :: scaleminorn2(kbdim,klev)
            REAL(KIND=wp), intent(out) :: minorfrac(kbdim,klev)
            !< column amount (h2o)
            !< column amount (co2)
            !< column amount (o3)
            !< column amount (n2o)
            !< column amount (co)
            !< column amount (ch4)
            !< column amount (o2)
            !< column amount (broadening gases)
            !<
            !<
            !<
            !<
            !<
            INTEGER :: jk
            REAL(KIND=wp) :: colmol(kbdim,klev)
            REAL(KIND=wp) :: factor(kbdim,klev)
            ! ------------------------------------------------
            CALL srtm_coeffs(kproma, kbdim, klev, play, tlay, coldry, wkl, laytrop, jp, jt, jt1, colch4, colco2, colh2o, colmol, &
            coln2o, colo2, colo3, fac00, fac01, fac10, fac11, selffac, selffrac, indself, forfac, forfrac, indfor)
            colbrd(1:kproma,1:klev) = 1.e-20_wp * wbroad(1:kproma,1:klev)
            colco(1:kproma,1:klev) = merge(1.e-20_wp * wkl(1:kproma,5,1:klev),                                      1.e-32_wp * &
            coldry(1:kproma,1:klev),                                     wkl(1:kproma,5,1:klev) > 0._wp)
            !
            ! Water vapor continuum broadening factors are used differently in LW and SW?
            !
            forfac(1:kproma,1:klev) = forfac(1:kproma,1:klev) * colh2o(1:kproma,1:klev)
            selffac(1:kproma,1:klev) = selffac(1:kproma,1:klev) * colh2o(1:kproma,1:klev)
            !
            !  Setup reference ratio to be used in calculation of binary species parameter.
            !
            DO jk = 1, klev
                rat_h2oco2(1:kproma, jk) = chi_mls(1,jp(1:kproma, jk))/chi_mls(2,jp(1:kproma, jk))
                rat_h2oco2_1(1:kproma, jk) = chi_mls(1,jp(1:kproma, jk)+1)/chi_mls(2,jp(1:kproma, jk)+1)
                !
                ! Needed only in lower atmos (plog > 4.56_wp)
                !
                rat_h2oo3(1:kproma, jk) = chi_mls(1,jp(1:kproma, jk))/chi_mls(3,jp(1:kproma, jk))
                rat_h2oo3_1(1:kproma, jk) = chi_mls(1,jp(1:kproma, jk)+1)/chi_mls(3,jp(1:kproma, jk)+1)
                rat_h2on2o(1:kproma, jk) = chi_mls(1,jp(1:kproma, jk))/chi_mls(4,jp(1:kproma, jk))
                rat_h2on2o_1(1:kproma, jk) = chi_mls(1,jp(1:kproma, jk)+1)/chi_mls(4,jp(1:kproma, jk)+1)
                rat_h2och4(1:kproma, jk) = chi_mls(1,jp(1:kproma, jk))/chi_mls(6,jp(1:kproma, jk))
                rat_h2och4_1(1:kproma, jk) = chi_mls(1,jp(1:kproma, jk)+1)/chi_mls(6,jp(1:kproma, jk)+1)
                rat_n2oco2(1:kproma, jk) = chi_mls(4,jp(1:kproma, jk))/chi_mls(2,jp(1:kproma, jk))
                rat_n2oco2_1(1:kproma, jk) = chi_mls(4,jp(1:kproma, jk)+1)/chi_mls(2,jp(1:kproma, jk)+1)
                !
                ! Needed only in upper atmos (plog <= 4.56_wp)
                !
                rat_o3co2(1:kproma, jk) = chi_mls(3,jp(1:kproma, jk))/chi_mls(2,jp(1:kproma, jk))
                rat_o3co2_1(1:kproma, jk) = chi_mls(3,jp(1:kproma, jk)+1)/chi_mls(2,jp(1:kproma, jk)+1)
            END DO 
            !
            !  Set up factors needed to separately include the minor gases
            !  in the calculation of absorption coefficient
            !
            scaleminor(1:kproma,1:klev) = play(1:kproma,1:klev)/tlay(1:kproma,1:klev)
            scaleminorn2(1:kproma,1:klev) = scaleminor(1:kproma,1:klev) *                              (wbroad(1:kproma,1:klev)/(&
            coldry(1:kproma,1:klev)+wkl(1:kproma,1,1:klev)))
            factor(1:kproma,1:klev) = (tlay(1:kproma,1:klev)-180.8_wp)/7.2_wp
            indminor(1:kproma,1:klev) = min(18, max(1, int(factor(1:kproma,1:klev))))
            minorfrac(1:kproma,1:klev) = (tlay(1:kproma,1:klev)-180.8_wp)/7.2_wp - float(indminor(1:kproma,1:klev))
        END SUBROUTINE lrtm_coeffs
        ! --------------------------------------------------------------------------------------------

        SUBROUTINE srtm_coeffs(kproma, kbdim, klev, play, tlay, coldry, wkl, laytrop, jp, jt, jt1, colch4, colco2, colh2o, colmol,&
         coln2o, colo2, colo3, fac00, fac01, fac10, fac11, selffac, selffrac, indself, forfac, forfrac, indfor)
            INTEGER, intent(in) :: kbdim
            INTEGER, intent(in) :: klev
            INTEGER, intent(in) :: kproma
            ! number of columns
            ! maximum number of col. as declared in calling (sub)programs
            ! total number of layers
            REAL(KIND=wp), intent(in) :: play(kbdim,klev)
            REAL(KIND=wp), intent(in) :: tlay(kbdim,klev)
            REAL(KIND=wp), intent(in) :: wkl(:,:,:)
            REAL(KIND=wp), intent(in) :: coldry(kbdim,klev)
            ! layer pressures (mb)
            ! layer temperatures (K)
            ! dry air column density (mol/cm2)
            !< molecular amounts (mol/cm-2) (mxmol,klev)
            !
            ! Output Dimensions kproma, klev unless otherwise specified
            !
            INTEGER, intent(out) :: jp(kbdim,klev)
            INTEGER, intent(out) :: jt(kbdim,klev)
            INTEGER, intent(out) :: jt1(kbdim,klev)
            INTEGER, intent(out) :: laytrop(kbdim)
            INTEGER, intent(out) :: indfor(kbdim,klev)
            INTEGER, intent(out) :: indself(kbdim,klev)
            !< tropopause layer index
            !
            !
            !
            !
            REAL(KIND=wp), intent(out) :: fac10(kbdim,klev)
            REAL(KIND=wp), intent(out) :: fac00(kbdim,klev)
            REAL(KIND=wp), intent(out) :: fac11(kbdim,klev)
            REAL(KIND=wp), intent(out) :: fac01(kbdim,klev)
            REAL(KIND=wp), intent(out) :: colh2o(kbdim,klev)
            REAL(KIND=wp), intent(out) :: colco2(kbdim,klev)
            REAL(KIND=wp), intent(out) :: colo3(kbdim,klev)
            REAL(KIND=wp), intent(out) :: coln2o(kbdim,klev)
            REAL(KIND=wp), intent(out) :: colch4(kbdim,klev)
            REAL(KIND=wp), intent(out) :: colo2(kbdim,klev)
            REAL(KIND=wp), intent(out) :: colmol(kbdim,klev)
            REAL(KIND=wp), intent(out) :: forfac(kbdim,klev)
            REAL(KIND=wp), intent(out) :: selffac(kbdim,klev)
            REAL(KIND=wp), intent(out) :: forfrac(kbdim,klev)
            REAL(KIND=wp), intent(out) :: selffrac(kbdim,klev)
            !< column amount (h2o)
            !< column amount (co2)
            !< column amount (o3)
            !< column amount (n2o)
            !< column amount (ch4)
            !< column amount (o2)
            !<
            !<
            !<
            !<
            !<
            INTEGER :: jp1(kbdim,klev)
            INTEGER :: jk
            REAL(KIND=wp) :: plog  (kbdim,klev)
            REAL(KIND=wp) :: fp      (kbdim,klev)
            REAL(KIND=wp) :: ft    (kbdim,klev)
            REAL(KIND=wp) :: ft1     (kbdim,klev)
            REAL(KIND=wp) :: water (kbdim,klev)
            REAL(KIND=wp) :: scalefac(kbdim,klev)
            REAL(KIND=wp) :: compfp(kbdim,klev)
            REAL(KIND=wp) :: factor  (kbdim,klev)
            ! -------------------------------------------------------------------------
            !
            !  Find the two reference pressures on either side of the
            !  layer pressure.  Store them in JP and JP1.  Store in FP the
            !  fraction of the difference (in ln(pressure)) between these
            !  two values that the layer pressure lies.
            !
            plog(1:kproma,1:klev) = log(play(1:kproma,1:klev))
            jp(1:kproma,1:klev) = min(58,max(1,int(36._wp - 5*(plog(1:kproma,1:klev)+0.04_wp))))
            jp1(1:kproma,1:klev) = jp(1:kproma,1:klev) + 1
            DO jk = 1, klev
                fp(1:kproma,jk) = 5._wp *(preflog(jp(1:kproma,jk)) - plog(1:kproma,jk))
            END DO 
            !
            !  Determine, for each reference pressure (JP and JP1), which
            !  reference temperature (these are different for each
            !  reference pressure) is nearest the layer temperature but does
            !  not exceed it.  Store these indices in JT and JT1, resp.
            !  Store in FT (resp. FT1) the fraction of the way between JT
            !  (JT1) and the next highest reference temperature that the
            !  layer temperature falls.
            !
            DO jk = 1, klev
                jt(1:kproma,jk) = min(4,max(1,int(3._wp + (tlay(1:kproma,jk)                                               - tref(&
                jp (1:kproma,jk)))/15._wp)))
                jt1(1:kproma,jk) = min(4,max(1,int(3._wp + (tlay(1:kproma,jk)                                               - &
                tref(jp1(1:kproma,jk)))/15._wp)))
            END DO 
            DO jk = 1, klev
                ft(1:kproma,jk) = ((tlay(1:kproma,jk)-tref(jp (1:kproma,jk)))/15._wp)                             - float(jt (&
                1:kproma,jk)-3)
                ft1(1:kproma,jk) = ((tlay(1:kproma,jk)-tref(jp1(1:kproma,jk)))/15._wp)                             - float(jt1(&
                1:kproma,jk)-3)
            END DO 
            water(1:kproma,1:klev) = wkl(1:kproma,1,1:klev)/coldry(1:kproma,1:klev)
            scalefac(1:kproma,1:klev) = play(1:kproma,1:klev) * stpfac / tlay(1:kproma,1:klev)
            !
            !  We have now isolated the layer ln pressure and temperature,
            !  between two reference pressures and two reference temperatures
            !  (for each reference pressure).  We multiply the pressure
            !  fraction FP with the appropriate temperature fractions to get
            !  the factors that will be needed for the interpolation that yields
            !  the optical depths (performed in routines TAUGBn for band n).`
            !
            compfp(1:kproma,1:klev) = 1. - fp(1:kproma,1:klev)
            fac10(1:kproma,1:klev) = compfp(1:kproma,1:klev) * ft(1:kproma,1:klev)
            fac00(1:kproma,1:klev) = compfp(1:kproma,1:klev) * (1._wp - ft(1:kproma,1:klev))
            fac11(1:kproma,1:klev) = fp(1:kproma,1:klev) * ft1(1:kproma,1:klev)
            fac01(1:kproma,1:klev) = fp(1:kproma,1:klev) * (1._wp - ft1(1:kproma,1:klev))
            ! Tropopause defined in terms of pressure (~100 hPa)
            !   We're looking for the first layer (counted from the bottom) at which the pressure reaches
            !   or falls below this value
            !
            laytrop(1:kproma) = count(plog(1:kproma,1:klev) > 4.56_wp, dim = 2)
            !
            !  Calculate needed column amounts.
            !    Only a few ratios are used in the upper atmosphere but masking may be less efficient
            !
            colh2o(1:kproma,1:klev) = merge(1.e-20_wp * wkl(1:kproma,1,1:klev),                                      1.e-32_wp * &
            coldry(1:kproma,1:klev),                                     wkl(1:kproma,1,1:klev) > 0._wp)
            colco2(1:kproma,1:klev) = merge(1.e-20_wp * wkl(1:kproma,2,1:klev),                                      1.e-32_wp * &
            coldry(1:kproma,1:klev),                                     wkl(1:kproma,2,1:klev) > 0._wp)
            colo3(1:kproma,1:klev) = merge(1.e-20_wp * wkl(1:kproma,3,1:klev),                                      1.e-32_wp * &
            coldry(1:kproma,1:klev),                                     wkl(1:kproma,3,1:klev) > 0._wp)
            coln2o(1:kproma,1:klev) = merge(1.e-20_wp * wkl(1:kproma,4,1:klev),                                      1.e-32_wp * &
            coldry(1:kproma,1:klev),                                     wkl(1:kproma,4,1:klev) > 0._wp)
            colch4(1:kproma,1:klev) = merge(1.e-20_wp * wkl(1:kproma,6,1:klev),                                      1.e-32_wp * &
            coldry(1:kproma,1:klev),                                     wkl(1:kproma,6,1:klev) > 0._wp)
            colo2(1:kproma,1:klev) = merge(1.e-20_wp * wkl(1:kproma,7,1:klev),                                      1.e-32_wp * &
            coldry(1:kproma,1:klev),                                     wkl(1:kproma,7,1:klev) > 0._wp)
            colmol(1:kproma,1:klev) = 1.e-20_wp * coldry(1:kproma,1:klev) + colh2o(1:kproma,1:klev)
            ! ------------------------------------------
            ! Interpolation coefficients
            !
            forfac(1:kproma,1:klev) = scalefac(1:kproma,1:klev) / (1._wp+water(1:kproma,1:klev))
            !
            !  Set up factors needed to separately include the water vapor
            !  self-continuum in the calculation of absorption coefficient.
            !
            selffac(1:kproma,1:klev) = water(1:kproma,1:klev) * forfac(1:kproma,1:klev)
            !
            !  If the pressure is less than ~100mb, perform a different set of species
            !  interpolations.
            !
            factor(1:kproma,1:klev) = (332.0_wp-tlay(1:kproma,1:klev))/36.0_wp
            indfor(1:kproma,1:klev) = merge(3,                                            min(2, max(1, int(factor(1:kproma,&
            1:klev)))),             plog(1:kproma,1:klev) <= 4.56_wp)
            forfrac(1:kproma,1:klev) = merge((tlay(1:kproma,1:klev)-188.0_wp)/36.0_wp - 1.0_wp,             factor(1:kproma,&
            1:klev) - float(indfor(1:kproma,1:klev)),                  plog(1:kproma,1:klev) <= 4.56_wp)
            ! In RRTMG code, this calculation is done only in the lower atmosphere (plog > 4.56)
            !
            factor(1:kproma,1:klev) = (tlay(1:kproma,1:klev)-188.0_wp)/7.2_wp
            indself(1:kproma,1:klev) = min(9, max(1, int(factor(1:kproma,1:klev))-7))
            selffrac(1:kproma,1:klev) = factor(1:kproma,1:klev) - float(indself(1:kproma,1:klev) + 7)
        END SUBROUTINE srtm_coeffs
    END MODULE mo_rrtm_coeffs
