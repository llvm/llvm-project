
! KGEN-generated Fortran source file
!
! Filename    : rrtmg_lw_setcoef.f90
! Generated at: 2015-07-26 18:24:46
! KGEN version: 0.4.13



    MODULE rrtmg_lw_setcoef
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
        USE parrrtm, ONLY: nbndlw
        USE parrrtm, ONLY: mxmol
        USE rrlw_wvn, ONLY: totplnk
        USE rrlw_wvn, ONLY: totplk16
        USE rrlw_ref, only : preflog
        USE rrlw_ref, only : tref
        USE rrlw_ref, only : chi_mls
        USE rrlw_vsn, ONLY: hvrset
        IMPLICIT NONE
        PUBLIC setcoef
        CONTAINS

        ! write subroutines
        ! No subroutines
        ! No module extern variables
        !----------------------------------------------------------------------------

        SUBROUTINE setcoef(ncol, nlayers, istart, pavel, tavel, tz, tbound, semiss, coldry, wkl, wbroad, laytrop, jp, jt, jt1, &
        planklay, planklev, plankbnd, colh2o, colco2, colo3, coln2o, colco, colch4, colo2, colbrd, fac00, fac01, fac10, fac11, &
        rat_h2oco2, rat_h2oco2_1, rat_h2oo3, rat_h2oo3_1, rat_h2on2o, rat_h2on2o_1, rat_h2och4, rat_h2och4_1, rat_n2oco2, &
        rat_n2oco2_1, rat_o3co2, rat_o3co2_1, selffac, selffrac, indself, forfac, forfrac, indfor, minorfrac, scaleminor, &
        scaleminorn2, indminor)
            !----------------------------------------------------------------------------
            !
            !  Purpose:  For a given atmosphere, calculate the indices and
            !  fractions related to the pressure and temperature interpolations.
            !  Also calculate the values of the integrated Planck functions
            !  for each band at the level and layer temperatures.
            ! ------- Declarations -------
            ! ----- Input -----
            INTEGER, intent(in) :: ncol !number of simd columns
            INTEGER, intent(in) :: nlayers ! total number of layers
            INTEGER, intent(in) :: istart ! beginning band of calculation
            REAL(KIND=r8), intent(in) :: pavel(ncol,nlayers) ! layer pressures (mb)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: tavel(ncol,nlayers) ! layer temperatures (K)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: tz(ncol,0:nlayers) ! level (interface) temperatures (K)
            !    Dimensions: (0:nlayers)
            REAL(KIND=r8), intent(in) :: tbound(ncol) ! surface temperature (K)
            REAL(KIND=r8), intent(in) :: coldry(ncol,nlayers) ! dry air column density (mol/cm2)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: wbroad(ncol,nlayers) ! broadening gas column density (mol/cm2)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(in) :: wkl(ncol,mxmol,nlayers) ! molecular amounts (mol/cm-2)
            !    Dimensions: (ncol,mxmol,nlayers)
            REAL(KIND=r8), intent(in) :: semiss(ncol,nbndlw) ! lw surface emissivity
            !    Dimensions: (nbndlw)
            ! ----- Output -----
            INTEGER, intent(out), dimension(:) :: laytrop ! tropopause layer index
            INTEGER, intent(out) :: jp(ncol,nlayers) !
            !    Dimensions: (nlayers)
            INTEGER, intent(out) :: jt(ncol,nlayers) !
            !    Dimensions: (nlayers)
            INTEGER, intent(out) :: jt1(ncol,nlayers) !
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(out) :: planklay(ncol,nlayers,nbndlw) !
            !    Dimensions: (ncol,nlayers,nbndlw)
            REAL(KIND=r8), intent(out) :: planklev(ncol,0:nlayers,nbndlw) !
            !    Dimensions: (ncol,0:nlayers,nbndlw)
            REAL(KIND=r8), intent(out) :: plankbnd(ncol,nbndlw) !
            !    Dimensions: (ncol,nbndlw)
            REAL(KIND=r8), intent(out) :: colh2o(ncol,nlayers) ! column amount (h2o)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(out) :: colco2(ncol,nlayers) ! column amount (co2)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(out) :: colo3(ncol,nlayers) ! column amount (o3)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(out) :: coln2o(ncol,nlayers) ! column amount (n2o)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(out) :: colco(ncol,nlayers) ! column amount (co)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(out) :: colch4(ncol,nlayers) ! column amount (ch4)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(out) :: colo2(ncol,nlayers) ! column amount (o2)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(out) :: colbrd(ncol,nlayers) ! column amount (broadening gases)
            !    Dimensions: (nlayers)
            INTEGER, intent(out) :: indself(ncol,nlayers)
            !    Dimensions: (nlayers)
            INTEGER, intent(out) :: indfor(ncol,nlayers)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(out) :: selffac(ncol,nlayers)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(out) :: selffrac(ncol,nlayers)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(out) :: forfac(ncol,nlayers)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(out) :: forfrac(ncol,nlayers)
            !    Dimensions: (nlayers)
            INTEGER, intent(out) :: indminor(ncol,nlayers)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(out) :: minorfrac(ncol,nlayers)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(out) :: scaleminor(ncol,nlayers)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(out) :: scaleminorn2(ncol,nlayers)
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(out) :: fac00(ncol,nlayers)
            REAL(KIND=r8), intent(out) :: fac01(ncol,nlayers)
            REAL(KIND=r8), intent(out) :: fac10(ncol,nlayers)
            REAL(KIND=r8), intent(out) :: fac11(ncol,nlayers) !
            !    Dimensions: (nlayers)
            REAL(KIND=r8), intent(out) :: rat_h2och4(ncol,nlayers)
            REAL(KIND=r8), intent(out) :: rat_h2on2o(ncol,nlayers)
            REAL(KIND=r8), intent(out) :: rat_h2on2o_1(ncol,nlayers)
            REAL(KIND=r8), intent(out) :: rat_o3co2_1(ncol,nlayers)
            REAL(KIND=r8), intent(out) :: rat_h2och4_1(ncol,nlayers)
            REAL(KIND=r8), intent(out) :: rat_n2oco2_1(ncol,nlayers)
            REAL(KIND=r8), intent(out) :: rat_h2oo3_1(ncol,nlayers)
            REAL(KIND=r8), intent(out) :: rat_n2oco2(ncol,nlayers)
            REAL(KIND=r8), intent(out) :: rat_h2oco2(ncol,nlayers)
            REAL(KIND=r8), intent(out) :: rat_h2oco2_1(ncol,nlayers)
            REAL(KIND=r8), intent(out) :: rat_h2oo3(ncol,nlayers)
            REAL(KIND=r8), intent(out) :: rat_o3co2(ncol,nlayers) !
            !    Dimensions: (nlayers)
            INTEGER :: indbound(1:ncol)
            INTEGER :: indlev0(1:ncol)
            INTEGER :: lay
            INTEGER :: icol
            INTEGER :: indlay(1:ncol)
            INTEGER :: indlev(1:ncol)
            INTEGER :: iband
            INTEGER :: jp1(1:ncol,1:nlayers)
            REAL(KIND=r8) :: stpfac
            REAL(KIND=r8) :: tbndfrac(1:ncol)
            REAL(KIND=r8) :: t0frac(1:ncol)
            REAL(KIND=r8) :: tlayfrac(1:ncol)
            REAL(KIND=r8) :: tlevfrac(1:ncol)
            REAL(KIND=r8) :: dbdtlev(1:ncol)
            REAL(KIND=r8) :: dbdtlay(1:ncol)
            REAL(KIND=r8) :: plog(1:ncol)
            REAL(KIND=r8) :: fp(1:ncol)
            REAL(KIND=r8) :: ft(1:ncol)
            REAL(KIND=r8) :: ft1(1:ncol)
            REAL(KIND=r8) :: water(1:ncol)
            REAL(KIND=r8) :: scalefac(1:ncol)
            REAL(KIND=r8) :: factor(1:ncol)
            REAL(KIND=r8) :: compfp(1:ncol)
            hvrset = '$Revision: 1.2 $'
            !dir$ assume_aligned tz:64
            !dir$ assume_aligned tavel:64
            !dir$ assume_aligned pavel:64
            !dir$ assume_aligned planklay:64
            !dir$ assume_aligned planklev:64
            !dir$ assume_aligned plankbnd:64
            !dir$ assume_aligned pavel:64
            !dir$ assume_aligned jp:64
            !dir$ assume_aligned jp1:64
            !dir$ assume_aligned jt:64
            !dir$ assume_aligned jt1:64
            !dir$ assume_aligned wkl:64
            !dir$ assume_aligned coldry:64
            stpfac = 296._r8/1013._r8
            !dir$ vector aligned
            !dir$ SIMD
            do icol=1,ncol
                indbound(icol) = tbound(icol) - 159._r8
                if (indbound(icol) .lt. 1) then
                    indbound(icol) = 1
                elseif (indbound(icol) .gt. 180) then
                    indbound(icol) = 180
                endif
                tbndfrac(icol) = tbound(icol) - 159._r8 - float(indbound(icol))
                indlev0(icol) = tz(icol,0) - 159._r8
                if (indlev0(icol) .lt. 1) then
                    indlev0(icol) = 1
                elseif (indlev0(icol) .gt. 180) then
                    indlev0(icol) = 180
                endif
                t0frac(icol) = tz(icol,0) - 159._r8 - float(indlev0(icol))
                laytrop(icol) = 0
                ! Begin layer loop
                !  Calculate the integrated Planck functions for each band at the
                !  surface, level, and layer temperatures.
            end do
            do lay = 1, nlayers
                !dir$ vector aligned
                !dir$ SIMD
                do icol=1,ncol
                    indlay(icol) = tavel(icol,lay) - 159._r8
                    if (indlay(icol) .lt. 1) then
                        indlay(icol) = 1
                    elseif (indlay(icol) .gt. 180) then
                        indlay(icol) = 180
                    endif
                    tlayfrac(icol) = tavel(icol,lay) - 159._r8 - float(indlay(icol)) ! !
                    indlev(icol) = tz(icol,lay) - 159._r8
                    if (indlev(icol) .lt. 1) then
                        indlev(icol) = 1
                    elseif (indlev(icol) .gt. 180) then
                        indlev(icol) = 180
                    endif
                    tlevfrac(icol) = tz(icol,lay) - 159._r8 - float(indlev(icol)) ! !
                    ! Begin spectral band loop
                end do ! end of icol loop ! end of icol loop
                do iband = 1, 15
                    !dir$ vector aligned
                    !dir$ SIMD
                        do icol=1,ncol
                            if (lay.eq.1) then
                            !print*,'inside iband : lay = 1 loop',lay
                                dbdtlev(icol) = totplnk(indbound(icol)+1,iband) - totplnk(indbound(icol),iband)
                                plankbnd(icol,iband) = semiss(icol,iband) * &
                                    (totplnk(indbound(icol),iband) + tbndfrac(icol) * dbdtlev(icol))
                                dbdtlev(icol) = totplnk(indlev0(icol)+1,iband)-totplnk(indlev0(icol),iband)
                                planklev(icol,0,iband) = totplnk(indlev0(icol),iband) + t0frac(icol) * dbdtlev(icol)
                            endif
                        end do
                    !dir$ vector aligned
                    !dir$ SIMD
                    do icol=1,ncol  
                        dbdtlev(icol) = totplnk(indlev(icol)+1,iband) - totplnk(indlev(icol),iband)
                        dbdtlay(icol) = totplnk(indlay(icol)+1,iband) - totplnk(indlay(icol),iband)
                        planklay(icol,lay,iband) = totplnk(indlay(icol),iband) + tlayfrac(icol) * dbdtlay(icol)
                        planklev(icol,lay,iband) = totplnk(indlev(icol),iband) + tlevfrac(icol) * dbdtlev(icol)
                        !       print *,'exiting iband loop',iband
                    end do ! end of icol loop ! end of icol loop
                enddo
                !  For band 16, if radiative transfer will be performed on just
                !  this band, use integrated Planck values up to 3250 cm-1.
                !  If radiative transfer will be performed across all 16 bands,
                !  then include in the integrated Planck values for this band
                !  contributions from 2600 cm-1 to infinity.
                iband = 16
                if (istart .eq. 16) then
                    !           print*,'iband ::::',iband
                    if (lay.eq.1) then
                        !dir$ vector aligned
                        !dir$ SIMD
                        do icol=1,ncol
                            dbdtlev(icol) = totplk16(indbound(icol)+1) - totplk16(indbound(icol))
                            plankbnd(icol,iband) = semiss(icol,iband) * &
                                (totplk16(indbound(icol)) + tbndfrac(icol) * dbdtlev(icol))
                            dbdtlev(icol) = totplnk(indlev0(icol)+1,iband)-totplnk(indlev0(icol),iband)
                            planklev(icol,0,iband) = totplk16(indlev0(icol)) + &
                                t0frac(icol) * dbdtlev(icol)
                        end do
                    endif
                    !dir$ vector aligned
                    !dir$ SIMD
                    do icol=1,ncol
                        dbdtlev(icol) = totplk16(indlev(icol)+1) - totplk16(indlev(icol))
                        dbdtlay(icol) = totplk16(indlay(icol)+1) - totplk16(indlay(icol))
                        planklay(icol,lay,iband) = totplk16(indlay(icol)) + tlayfrac(icol) * dbdtlay(icol)
                        planklev(icol,lay,iband) = totplk16(indlev(icol)) + tlevfrac(icol) * dbdtlev(icol)
                    end do
                else
                    if (lay.eq.1) then
                        !dir$ vector aligned
                        !dir$ SIMD
                        do icol=1,ncol
                            dbdtlev(icol) = totplnk(indbound(icol)+1,iband) - totplnk(indbound(icol),iband)
                            plankbnd(icol,iband) = semiss(icol,iband) * &
                                (totplnk(indbound(icol),iband) + tbndfrac(icol) * dbdtlev(icol))
                            dbdtlev(icol) = totplnk(indlev0(icol)+1,iband)-totplnk(indlev0(icol),iband)
                            planklev(icol,0,iband) = totplnk(indlev0(icol),iband) + t0frac(icol) * dbdtlev(icol)
                        end do
                    endif
                    !dir$ vector aligned
                    !dir$ SIMD
                    do icol=1,ncol
                        dbdtlev(icol) = totplnk(indlev(icol)+1,iband) - totplnk(indlev(icol),iband)
                        dbdtlay(icol) = totplnk(indlay(icol)+1,iband) - totplnk(indlay(icol),iband)
                        planklay(icol,lay,iband) = totplnk(indlay(icol),iband) + tlayfrac(icol) * dbdtlay(icol)
                        planklev(icol,lay,iband) = totplnk(indlev(icol),iband) + tlevfrac(icol) * dbdtlev(icol)
                    end do
                endif
                !  Find the two reference pressures on either side of the
                !  layer pressure.  Store them in JP and JP1.  Store in FP the
                !  fraction of the difference (in ln(pressure)) between these
                !  two values that the layer pressure lies.
                !         plog = alog(pavel(lay))
                !dir$ vector aligned
                !dir$ SIMD
                do icol=1,ncol
                    plog(icol) = dlog(pavel(icol,lay))
                    jp(icol,lay) = int(36._r8 - 5*(plog(icol)+0.04_r8))
                    if (jp(icol,lay) .lt. 1) then
                        jp(icol,lay) = 1
                    elseif (jp(icol,lay) .gt. 58) then
                        jp(icol,lay) = 58
                    endif
                    jp1(icol,lay) = jp(icol,lay) + 1
                    fp(icol) = 5._r8 *(preflog(jp(icol,lay)) - plog(icol))
                    !  Determine, for each reference pressure (JP and JP1), which
                    !  reference temperature (these are different for each
                    !  reference pressure) is nearest the layer temperature but does
                    !  not exceed it.  Store these indices in JT and JT1, resp.
                    !  Store in FT (resp. FT1) the fraction of the way between JT
                    !  (JT1) and the next highest reference temperature that the
                    !  layer temperature falls.
                    !  (JT1) and the next highest reference temperature that the
                    !  layer temperature falls.
                    jt(icol,lay) = int(3._r8 + (tavel(icol,lay)-tref(jp(icol,lay)))/15._r8)
                    if (jt(icol,lay) .lt. 1) then
                        jt(icol,lay) = 1
                    elseif (jt(icol,lay) .gt. 4) then
                        jt(icol,lay) = 4
                    endif
                    ft(icol) = ((tavel(icol,lay)-tref(jp(icol,lay)))/15._r8) - float(jt(icol,lay)-3)
                    jt1(icol,lay) = int(3._r8 + (tavel(icol,lay)-tref(jp1(icol,lay)))/15._r8)
                    if (jt1(icol,lay) .lt. 1) then
                        jt1(icol,lay) = 1
                    elseif (jt1(icol,lay) .gt. 4) then
                        jt1(icol,lay) = 4
                    endif
                    ft1(icol) = ((tavel(icol,lay)-tref(jp1(icol,lay)))/15._r8) - float(jt1(icol,lay)-3)
                    water(icol) = wkl(icol,1,lay)/coldry(icol,lay)
                    scalefac(icol) = pavel(icol,lay) * stpfac / tavel(icol,lay)
                    !  If the pressure is less than ~100mb, perform a different
                    !  set of species interpolations.
                    if (plog(icol) .le. 4.56_r8) then 
                        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        forfac(icol,lay) = scalefac(icol) / (1.+water(icol))
                        factor(icol) = (tavel(icol,lay)-188.0_r8)/36.0_r8
                        indfor(icol,lay) = 3
                        forfrac(icol,lay) = factor(icol) - 1.0_r8
                        !  Set up factors needed to separately include the water vapor
                        !  self-continuum in the calculation of absorption coefficient.
                        selffac(icol,lay) = water(icol) * forfac(icol,lay)
                        !  Set up factors needed to separately include the minor gases
                        !  in the calculation of absorption coefficient
                        scaleminor(icol,lay) = pavel(icol,lay)/tavel(icol,lay)
                        scaleminorn2(icol,lay) = (pavel(icol,lay)/tavel(icol,lay)) &
                             * (wbroad(icol,lay)/(coldry(icol,lay)+wkl(icol,1,lay)))
                        factor(icol) = (tavel(icol,lay)-180.8_r8)/7.2_r8
                        indminor(icol,lay) = min(18, max(1, int(factor(icol))))
                        minorfrac(icol,lay) = factor(icol) - float(indminor(icol,lay))
                        !  Setup reference ratio to be used in calculation of binary
                        !  species parameter in upper atmosphere.
                        rat_h2oco2(icol,lay)=chi_mls(1,jp(icol,lay))/chi_mls(2,jp(icol,lay))
                        rat_h2oco2_1(icol,lay)=chi_mls(1,jp(icol,lay)+1)/chi_mls(2,jp(icol,lay)+1)
                        rat_o3co2(icol,lay)=chi_mls(3,jp(icol,lay))/chi_mls(2,jp(icol,lay))
                        rat_o3co2_1(icol,lay)=chi_mls(3,jp(icol,lay)+1)/chi_mls(2,jp(icol,lay)+1)
                        !  Calculate needed column amounts.
                        !  Calculate needed column amounts.
                        colh2o(icol,lay) = 1.e-20_r8 * wkl(icol,1,lay)
                        colco2(icol,lay) = 1.e-20_r8 * wkl(icol,2,lay)
                        colo3(icol,lay) = 1.e-20_r8 * wkl(icol,3,lay)
                        coln2o(icol,lay) = 1.e-20_r8 * wkl(icol,4,lay)
                        colco(icol,lay) = 1.e-20_r8 * wkl(icol,5,lay)
                        colch4(icol,lay) = 1.e-20_r8 * wkl(icol,6,lay)
                        colo2(icol,lay) = 1.e-20_r8 * wkl(icol,7,lay)
                        if (colco2(icol,lay) .eq. 0._r8) colco2(icol,lay) = 1.e-32_r8 * coldry(icol,lay)
                        if (colo3(icol,lay) .eq. 0._r8) colo3(icol,lay) = 1.e-32_r8 * coldry(icol,lay)
                        if (coln2o(icol,lay) .eq. 0._r8) coln2o(icol,lay) = 1.e-32_r8 * coldry(icol,lay)
                        if (colco(icol,lay)  .eq. 0._r8) colco(icol,lay) = 1.e-32_r8 * coldry(icol,lay)
                        if (colch4(icol,lay) .eq. 0._r8) colch4(icol,lay) = 1.e-32_r8 * coldry(icol,lay)
                        colbrd(icol,lay) = 1.e-20_r8 * wbroad(icol,lay)
                        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    else
                        laytrop(icol) =  laytrop(icol) + 1
                        forfac(icol,lay) = scalefac(icol) / (1.+water(icol))
                        factor(icol) = (332.0_r8-tavel(icol,lay))/36.0_r8
                        indfor(icol,lay) = min(2, max(1, int(factor(icol))))
                        forfrac(icol,lay) = factor(icol) - float(indfor(icol,lay))
                        !  Set up factors needed to separately include the water vapor
                        !  self-continuum in the calculation of absorption coefficient.
                        selffac(icol,lay) = water(icol) * forfac(icol,lay)
                        factor(icol) = (tavel(icol,lay)-188.0_r8)/7.2_r8
                        indself(icol,lay) = min(9, max(1, int(factor(icol))-7))
                        selffrac(icol,lay) = factor(icol) - float(indself(icol,lay) + 7)
                        indself(icol,lay) = min(9, max(1, int(factor(icol))-7))
                        selffrac(icol,lay) = factor(icol) - float(indself(icol,lay) + 7)
                        !  Set up factors needed to separately include the minor gases
                        !  in the calculation of absorption coefficient
                        scaleminor(icol,lay) = pavel(icol,lay)/tavel(icol,lay)
                        scaleminorn2(icol,lay) = (pavel(icol,lay)/tavel(icol,lay)) &
                         *(wbroad(icol,lay)/(coldry(icol,lay)+wkl(icol,1,lay)))
                        factor(icol) = (tavel(icol,lay)-180.8_r8)/7.2_r8
                        indminor(icol,lay) = min(18, max(1, int(factor(icol))))
                        minorfrac(icol,lay) = factor(icol) - float(indminor(icol,lay))
                        !  Setup reference ratio to be used in calculation of binary
                        !  species parameter in lower atmosphere.
                        rat_h2oco2(icol,lay)=chi_mls(1,jp(icol,lay))/chi_mls(2,jp(icol,lay))
                        rat_h2oco2_1(icol,lay)=chi_mls(1,jp(icol,lay)+1)/chi_mls(2,jp(icol,lay)+1)
                        rat_h2oo3(icol,lay)=chi_mls(1,jp(icol,lay))/chi_mls(3,jp(icol,lay))
                        rat_h2oo3_1(icol,lay)=chi_mls(1,jp(icol,lay)+1)/chi_mls(3,jp(icol,lay)+1)
                        rat_h2on2o(icol,lay)=chi_mls(1,jp(icol,lay))/chi_mls(4,jp(icol,lay))
                        rat_h2on2o_1(icol,lay)=chi_mls(1,jp(icol,lay)+1)/chi_mls(4,jp(icol,lay)+1)
                        rat_h2och4(icol,lay)=chi_mls(1,jp(icol,lay))/chi_mls(6,jp(icol,lay))
                        rat_h2och4_1(icol,lay)=chi_mls(1,jp(icol,lay)+1)/chi_mls(6,jp(icol,lay)+1)
                        rat_n2oco2(icol,lay)=chi_mls(4,jp(icol,lay))/chi_mls(2,jp(icol,lay))
                        rat_n2oco2_1(icol,lay)=chi_mls(4,jp(icol,lay)+1)/chi_mls(2,jp(icol,lay)+1)
                        !  Calculate needed column amounts.
                        colh2o(icol,lay) = 1.e-20_r8 * wkl(icol,1,lay)
                        colco2(icol,lay) = 1.e-20_r8 * wkl(icol,2,lay)
                        colo3(icol,lay) = 1.e-20_r8 * wkl(icol,3,lay)
                        coln2o(icol,lay) = 1.e-20_r8 * wkl(icol,4,lay)
                        colco(icol,lay) = 1.e-20_r8 * wkl(icol,5,lay)
                        colch4(icol,lay) = 1.e-20_r8 * wkl(icol,6,lay)
                        colo2(icol,lay) = 1.e-20_r8 * wkl(icol,7,lay)
                        if (colco2(icol,lay) .eq. 0._r8) colco2(icol,lay) = 1.e-32_r8 * coldry(icol,lay)
                        if (colo3(icol,lay) .eq. 0._r8) colo3(icol,lay) = 1.e-32_r8 * coldry(icol,lay)
                        if (coln2o(icol,lay) .eq. 0._r8) coln2o(icol,lay) = 1.e-32_r8 * coldry(icol,lay)
                        if (colco(icol,lay) .eq. 0._r8) colco(icol,lay) = 1.e-32_r8 * coldry(icol,lay)
                        if (coln2o(icol,lay) .eq. 0._r8) coln2o(icol,lay) = 1.e-32_r8 * coldry(icol,lay)
                        if (colco(icol,lay) .eq. 0._r8) colco(icol,lay) = 1.e-32_r8 * coldry(icol,lay)
                        if (colch4(icol,lay) .eq. 0._r8) colch4(icol,lay) = 1.e-32_r8 * coldry(icol,lay)
                        colbrd(icol,lay) = 1.e-20_r8 * wbroad(icol,lay)
                        !go to 5400
                        !  Above laytrop.
                    endif
                    !5300   continue
                    !5400    continue
                    !  We have now isolated the layer ln pressure and temperature,
                    !  between two reference pressures and two reference temperatures
                    !  (for each reference pressure).  We multiply the pressure
                    !  fraction FP with the appropriate temperature fractions to get
                    !  the factors that will be needed for the interpolation that yields
                    !  the optical depths (performed in routines TAUGBn for band n).`
                        compfp(icol) = 1. - fp(icol)
                        fac10(icol,lay) = compfp(icol)* ft(icol)
                        fac00(icol,lay) = compfp(icol) * (1._r8 - ft(icol))
                        fac11(icol,lay) = fp(icol) * ft1(icol)
                        fac01(icol,lay) = fp(icol) * (1._r8 - ft1(icol))
                    !  Rescale selffac and forfac for use in taumol
                        selffac(icol,lay) = colh2o(icol,lay)*selffac(icol,lay)
                        forfac(icol,lay) = colh2o(icol,lay)*forfac(icol,lay)
                    ! End layer loop
                    !print*,'exiting lay loop',lay
                    end do
            end do
            !print*,'exiting icol loop',icol
        END SUBROUTINE setcoef
        !***************************************************************************

        !***************************************************************************

    END MODULE rrtmg_lw_setcoef
