    MODULE resolvers

    ! RESOLVER SPECS
    INTEGER, PARAMETER :: r8 = selected_real_kind(12)
    INTEGER, PARAMETER :: nmol = 7
    INTEGER, PARAMETER :: maxxsec = 4
    INTEGER, PARAMETER :: nbndlw = 16
    INTEGER, PARAMETER :: ngptlw = 140
    INTEGER, PARAMETER :: mxmol = 38
    INTEGER, PARAMETER :: maxinpx = 38

    END MODULE

    MODULE subprograms

    CONTAINS


    ! KERNEL DRIVER SUBROUTINE
    SUBROUTINE kernel_driver(taucmcl, ch4vmr, icld, emis, tlay, reicmcl, nlay, cfc11vmr, tsfc, relqmcl, o3vmr, n2ovmr, plev, play, tauaer, clwpmcl, o2vmr, co2vmr, ccl4vmr, iceflglw, cfc12vmr, tlev, h2ovmr, inflglw, ciwpmcl, cldfmcl, liqflglw, cfc22vmr, kgen_unit)
    USE resolvers

    IMPLICIT NONE
    INTEGER, INTENT(IN) :: kgen_unit
    INTEGER, DIMENSION(2,10) :: kgen_bound


    ! STATE SPECS
    REAL(KIND = r8), INTENT(IN) :: taucmcl(:, :, :)
    INTEGER :: iceflag
    REAL(KIND = r8) :: wkl(mxmol, nlay)
    REAL(KIND = r8) :: coldry(nlay)
    REAL(KIND = r8), INTENT(IN) :: ch4vmr(:, :)
    REAL(KIND = r8) :: clwpmc(ngptlw, nlay)
    INTEGER, INTENT(INOUT) :: icld
    REAL(KIND = r8), INTENT(IN) :: emis(:, :)
    REAL(KIND = r8) :: avogad
    REAL(KIND = r8) :: cldfmc(ngptlw, nlay)
    REAL(KIND = r8) :: relqmc(nlay)
    REAL(KIND = r8) :: ciwpmc(ngptlw, nlay)
    REAL(KIND = r8) :: wbrodl(nlay)
    REAL(KIND = r8), INTENT(IN) :: tlay(:, :)
    REAL(KIND = r8), INTENT(IN) :: reicmcl(:, :)
    INTEGER, INTENT(IN) :: nlay
    REAL(KIND = r8) :: tavel(nlay)
    INTEGER :: liqflag
    REAL(KIND = r8) :: tz(0 : nlay)
    REAL(KIND = r8), INTENT(IN) :: cfc11vmr(:, :)
    REAL(KIND = r8), INTENT(IN) :: tsfc(:)
    REAL(KIND = r8) :: pz(0 : nlay)
    REAL(KIND = r8), INTENT(IN) :: relqmcl(:, :)
    REAL(KIND = r8), INTENT(IN) :: o3vmr(:, :)
    REAL(KIND = r8) :: tbound
    INTEGER :: iaer
    REAL(KIND = r8), INTENT(IN) :: n2ovmr(:, :)
    REAL(KIND = r8) :: reicmc(nlay)
    REAL(KIND = r8), INTENT(IN) :: plev(:, :)
    REAL(KIND = r8), INTENT(IN) :: play(:, :)
    REAL(KIND = r8), INTENT(IN) :: tauaer(:, :, :)
    REAL(KIND = r8) :: semiss(nbndlw)
    REAL(KIND = r8) :: pavel(nlay)
    REAL(KIND = r8), INTENT(IN) :: clwpmcl(:, :, :)
    REAL(KIND = r8), INTENT(IN) :: o2vmr(:, :)
    REAL(KIND = r8) :: dgesmc(nlay)
    REAL(KIND = r8) :: pwvcm
    REAL(KIND = r8), INTENT(IN) :: co2vmr(:, :)
    INTEGER :: inflag
    REAL(KIND = r8) :: wx(maxxsec, nlay)
    REAL(KIND = r8), INTENT(IN) :: ccl4vmr(:, :)
    REAL(KIND = r8) :: taua(nlay, nbndlw)
    INTEGER, INTENT(IN) :: iceflglw
    REAL(KIND = r8), INTENT(IN) :: cfc12vmr(:, :)
    REAL(KIND = r8), INTENT(IN) :: tlev(:, :)
    REAL(KIND = r8) :: grav
    REAL(KIND = r8) :: taucmc(ngptlw, nlay)
    REAL(KIND = r8), INTENT(IN) :: h2ovmr(:, :)
    INTEGER :: iplon
    INTEGER, INTENT(IN) :: inflglw
    REAL(KIND = r8), INTENT(IN) :: ciwpmcl(:, :, :)
    INTEGER :: ixindx(maxinpx)
    REAL(KIND = r8), INTENT(IN) :: cldfmcl(:, :, :)
    INTEGER, INTENT(IN) :: liqflglw
    REAL(KIND = r8), INTENT(IN) :: cfc22vmr(:, :)
    INTEGER :: outstate_iceflag
    REAL(KIND = r8) :: outstate_wkl(mxmol, nlay)
    REAL(KIND = r8) :: outstate_coldry(nlay)
    REAL(KIND = r8) :: outstate_clwpmc(ngptlw, nlay)
    REAL(KIND = r8) :: outstate_cldfmc(ngptlw, nlay)
    REAL(KIND = r8) :: outstate_relqmc(nlay)
    REAL(KIND = r8) :: outstate_ciwpmc(ngptlw, nlay)
    REAL(KIND = r8) :: outstate_wbrodl(nlay)
    REAL(KIND = r8) :: outstate_tavel(nlay)
    INTEGER :: outstate_liqflag
    REAL(KIND = r8) :: outstate_tz(0 : nlay)
    REAL(KIND = r8) :: outstate_pz(0 : nlay)
    REAL(KIND = r8) :: outstate_tbound
    REAL(KIND = r8) :: outstate_reicmc(nlay)
    REAL(KIND = r8) :: outstate_semiss(nbndlw)
    REAL(KIND = r8) :: outstate_pavel(nlay)
    REAL(KIND = r8) :: outstate_dgesmc(nlay)
    REAL(KIND = r8) :: outstate_pwvcm
    INTEGER :: outstate_inflag
    REAL(KIND = r8) :: outstate_wx(maxxsec, nlay)
    REAL(KIND = r8) :: outstate_taua(nlay, nbndlw)
    REAL(KIND = r8) :: outstate_taucmc(ngptlw, nlay)


    ! READ CALLER INSTATE
    READ(UNIT = kgen_unit) iaer
    READ(UNIT = kgen_unit) iplon


    ! READ CALLEE INSTATE
    READ(UNIT = kgen_unit) avogad
    READ(UNIT = kgen_unit) grav
    READ(UNIT = kgen_unit) ixindx


    ! READ CALLEE OUTSTATE


    ! READ CALLER OUTSTATE
    READ(UNIT = kgen_unit) outstate_iceflag
    READ(UNIT = kgen_unit) outstate_wkl
    READ(UNIT = kgen_unit) outstate_coldry
    READ(UNIT = kgen_unit) outstate_clwpmc
    READ(UNIT = kgen_unit) outstate_cldfmc
    READ(UNIT = kgen_unit) outstate_relqmc
    READ(UNIT = kgen_unit) outstate_ciwpmc
    READ(UNIT = kgen_unit) outstate_wbrodl
    READ(UNIT = kgen_unit) outstate_tavel
    READ(UNIT = kgen_unit) outstate_liqflag
    READ(UNIT = kgen_unit) outstate_tz
    READ(UNIT = kgen_unit) outstate_pz
    READ(UNIT = kgen_unit) outstate_tbound
    READ(UNIT = kgen_unit) outstate_reicmc
    READ(UNIT = kgen_unit) outstate_semiss
    READ(UNIT = kgen_unit) outstate_pavel
    READ(UNIT = kgen_unit) outstate_dgesmc
    READ(UNIT = kgen_unit) outstate_pwvcm
    READ(UNIT = kgen_unit) outstate_inflag
    READ(UNIT = kgen_unit) outstate_wx
    READ(UNIT = kgen_unit) outstate_taua
    READ(UNIT = kgen_unit) outstate_taucmc


    ! KERNEL RUN
    CALL inatm(iplon, nlay, icld, iaer, play, plev, tlay, tlev, tsfc, h2ovmr, o3vmr, co2vmr, ch4vmr, o2vmr, n2ovmr, cfc11vmr, cfc12vmr, cfc22vmr, ccl4vmr, emis, inflglw, iceflglw, liqflglw, cldfmcl, taucmcl, ciwpmcl, clwpmcl, reicmcl, relqmcl, tauaer, pavel, pz, tavel, tz, tbound, semiss, coldry, wkl, wbrodl, wx, pwvcm, inflag, iceflag, liqflag, cldfmc, taucmc, ciwpmc, clwpmc, reicmc, dgesmc, relqmc, taua)


    ! STATE VERIFICATION
    IF ( outstate_iceflag == iceflag ) THEN
        WRITE(*,*) "iceflag is IDENTICAL( ", outstate_iceflag, " )."
    ELSE
        WRITE(*,*) "iceflag is NOT IDENTICAL."
        WRITE(*,*) "STATE : ", outstate_iceflag
        WRITE(*,*) "KERNEL: ", iceflag
    END IF
    IF ( ALL( outstate_wkl == wkl ) ) THEN
        WRITE(*,*) "All elements of wkl are IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_wkl
        !WRITE(*,*) "KERNEL: ", wkl
        IF ( ALL( outstate_wkl == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        WRITE(*,*) "wkl is NOT IDENTICAL."
        WRITE(*,*) count( outstate_wkl /= wkl), " of ", size( wkl ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_wkl - wkl)**2)/real(size(outstate_wkl)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_wkl - wkl))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_wkl - wkl))
        WRITE(*,*) "Mean value of kernel-generated outstate_wkl is ", sum(wkl)/real(size(wkl))
        WRITE(*,*) "Mean value of original outstate_wkl is ", sum(outstate_wkl)/real(size(outstate_wkl))
        WRITE(*,*) ""
    END IF
    IF ( ALL( outstate_coldry == coldry ) ) THEN
        WRITE(*,*) "All elements of coldry are IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_coldry
        !WRITE(*,*) "KERNEL: ", coldry
        IF ( ALL( outstate_coldry == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        WRITE(*,*) "coldry is NOT IDENTICAL."
        WRITE(*,*) count( outstate_coldry /= coldry), " of ", size( coldry ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_coldry - coldry)**2)/real(size(outstate_coldry)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_coldry - coldry))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_coldry - coldry))
        WRITE(*,*) "Mean value of kernel-generated outstate_coldry is ", sum(coldry)/real(size(coldry))
        WRITE(*,*) "Mean value of original outstate_coldry is ", sum(outstate_coldry)/real(size(outstate_coldry))
        WRITE(*,*) ""
    END IF
    IF ( ALL( outstate_clwpmc == clwpmc ) ) THEN
        WRITE(*,*) "All elements of clwpmc are IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_clwpmc
        !WRITE(*,*) "KERNEL: ", clwpmc
        IF ( ALL( outstate_clwpmc == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        WRITE(*,*) "clwpmc is NOT IDENTICAL."
        WRITE(*,*) count( outstate_clwpmc /= clwpmc), " of ", size( clwpmc ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_clwpmc - clwpmc)**2)/real(size(outstate_clwpmc)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_clwpmc - clwpmc))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_clwpmc - clwpmc))
        WRITE(*,*) "Mean value of kernel-generated outstate_clwpmc is ", sum(clwpmc)/real(size(clwpmc))
        WRITE(*,*) "Mean value of original outstate_clwpmc is ", sum(outstate_clwpmc)/real(size(outstate_clwpmc))
        WRITE(*,*) ""
    END IF
    IF ( ALL( outstate_cldfmc == cldfmc ) ) THEN
        WRITE(*,*) "All elements of cldfmc are IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_cldfmc
        !WRITE(*,*) "KERNEL: ", cldfmc
        IF ( ALL( outstate_cldfmc == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        WRITE(*,*) "cldfmc is NOT IDENTICAL."
        WRITE(*,*) count( outstate_cldfmc /= cldfmc), " of ", size( cldfmc ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_cldfmc - cldfmc)**2)/real(size(outstate_cldfmc)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_cldfmc - cldfmc))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_cldfmc - cldfmc))
        WRITE(*,*) "Mean value of kernel-generated outstate_cldfmc is ", sum(cldfmc)/real(size(cldfmc))
        WRITE(*,*) "Mean value of original outstate_cldfmc is ", sum(outstate_cldfmc)/real(size(outstate_cldfmc))
        WRITE(*,*) ""
    END IF
    IF ( ALL( outstate_relqmc == relqmc ) ) THEN
        WRITE(*,*) "All elements of relqmc are IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_relqmc
        !WRITE(*,*) "KERNEL: ", relqmc
        IF ( ALL( outstate_relqmc == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        WRITE(*,*) "relqmc is NOT IDENTICAL."
        WRITE(*,*) count( outstate_relqmc /= relqmc), " of ", size( relqmc ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_relqmc - relqmc)**2)/real(size(outstate_relqmc)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_relqmc - relqmc))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_relqmc - relqmc))
        WRITE(*,*) "Mean value of kernel-generated outstate_relqmc is ", sum(relqmc)/real(size(relqmc))
        WRITE(*,*) "Mean value of original outstate_relqmc is ", sum(outstate_relqmc)/real(size(outstate_relqmc))
        WRITE(*,*) ""
    END IF
    IF ( ALL( outstate_ciwpmc == ciwpmc ) ) THEN
        WRITE(*,*) "All elements of ciwpmc are IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_ciwpmc
        !WRITE(*,*) "KERNEL: ", ciwpmc
        IF ( ALL( outstate_ciwpmc == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        WRITE(*,*) "ciwpmc is NOT IDENTICAL."
        WRITE(*,*) count( outstate_ciwpmc /= ciwpmc), " of ", size( ciwpmc ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_ciwpmc - ciwpmc)**2)/real(size(outstate_ciwpmc)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_ciwpmc - ciwpmc))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_ciwpmc - ciwpmc))
        WRITE(*,*) "Mean value of kernel-generated outstate_ciwpmc is ", sum(ciwpmc)/real(size(ciwpmc))
        WRITE(*,*) "Mean value of original outstate_ciwpmc is ", sum(outstate_ciwpmc)/real(size(outstate_ciwpmc))
        WRITE(*,*) ""
    END IF
    IF ( ALL( outstate_wbrodl == wbrodl ) ) THEN
        WRITE(*,*) "All elements of wbrodl are IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_wbrodl
        !WRITE(*,*) "KERNEL: ", wbrodl
        IF ( ALL( outstate_wbrodl == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        WRITE(*,*) "wbrodl is NOT IDENTICAL."
        WRITE(*,*) count( outstate_wbrodl /= wbrodl), " of ", size( wbrodl ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_wbrodl - wbrodl)**2)/real(size(outstate_wbrodl)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_wbrodl - wbrodl))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_wbrodl - wbrodl))
        WRITE(*,*) "Mean value of kernel-generated outstate_wbrodl is ", sum(wbrodl)/real(size(wbrodl))
        WRITE(*,*) "Mean value of original outstate_wbrodl is ", sum(outstate_wbrodl)/real(size(outstate_wbrodl))
        WRITE(*,*) ""
    END IF
    IF ( ALL( outstate_tavel == tavel ) ) THEN
        WRITE(*,*) "All elements of tavel are IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_tavel
        !WRITE(*,*) "KERNEL: ", tavel
        IF ( ALL( outstate_tavel == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        WRITE(*,*) "tavel is NOT IDENTICAL."
        WRITE(*,*) count( outstate_tavel /= tavel), " of ", size( tavel ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_tavel - tavel)**2)/real(size(outstate_tavel)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_tavel - tavel))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_tavel - tavel))
        WRITE(*,*) "Mean value of kernel-generated outstate_tavel is ", sum(tavel)/real(size(tavel))
        WRITE(*,*) "Mean value of original outstate_tavel is ", sum(outstate_tavel)/real(size(outstate_tavel))
        WRITE(*,*) ""
    END IF
    IF ( outstate_liqflag == liqflag ) THEN
        WRITE(*,*) "liqflag is IDENTICAL( ", outstate_liqflag, " )."
    ELSE
        WRITE(*,*) "liqflag is NOT IDENTICAL."
        WRITE(*,*) "STATE : ", outstate_liqflag
        WRITE(*,*) "KERNEL: ", liqflag
    END IF
    IF ( ALL( outstate_tz == tz ) ) THEN
        WRITE(*,*) "All elements of tz are IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_tz
        !WRITE(*,*) "KERNEL: ", tz
        IF ( ALL( outstate_tz == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        WRITE(*,*) "tz is NOT IDENTICAL."
        WRITE(*,*) count( outstate_tz /= tz), " of ", size( tz ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_tz - tz)**2)/real(size(outstate_tz)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_tz - tz))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_tz - tz))
        WRITE(*,*) "Mean value of kernel-generated outstate_tz is ", sum(tz)/real(size(tz))
        WRITE(*,*) "Mean value of original outstate_tz is ", sum(outstate_tz)/real(size(outstate_tz))
        WRITE(*,*) ""
    END IF
    IF ( ALL( outstate_pz == pz ) ) THEN
        WRITE(*,*) "All elements of pz are IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_pz
        !WRITE(*,*) "KERNEL: ", pz
        IF ( ALL( outstate_pz == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        WRITE(*,*) "pz is NOT IDENTICAL."
        WRITE(*,*) count( outstate_pz /= pz), " of ", size( pz ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_pz - pz)**2)/real(size(outstate_pz)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_pz - pz))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_pz - pz))
        WRITE(*,*) "Mean value of kernel-generated outstate_pz is ", sum(pz)/real(size(pz))
        WRITE(*,*) "Mean value of original outstate_pz is ", sum(outstate_pz)/real(size(outstate_pz))
        WRITE(*,*) ""
    END IF
    IF ( outstate_tbound == tbound ) THEN
        WRITE(*,*) "tbound is IDENTICAL( ", outstate_tbound, " )."
    ELSE
        WRITE(*,*) "tbound is NOT IDENTICAL."
        WRITE(*,*) "STATE : ", outstate_tbound
        WRITE(*,*) "KERNEL: ", tbound
    END IF
    IF ( ALL( outstate_reicmc == reicmc ) ) THEN
        WRITE(*,*) "All elements of reicmc are IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_reicmc
        !WRITE(*,*) "KERNEL: ", reicmc
        IF ( ALL( outstate_reicmc == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        WRITE(*,*) "reicmc is NOT IDENTICAL."
        WRITE(*,*) count( outstate_reicmc /= reicmc), " of ", size( reicmc ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_reicmc - reicmc)**2)/real(size(outstate_reicmc)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_reicmc - reicmc))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_reicmc - reicmc))
        WRITE(*,*) "Mean value of kernel-generated outstate_reicmc is ", sum(reicmc)/real(size(reicmc))
        WRITE(*,*) "Mean value of original outstate_reicmc is ", sum(outstate_reicmc)/real(size(outstate_reicmc))
        WRITE(*,*) ""
    END IF
    IF ( ALL( outstate_semiss == semiss ) ) THEN
        WRITE(*,*) "All elements of semiss are IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_semiss
        !WRITE(*,*) "KERNEL: ", semiss
        IF ( ALL( outstate_semiss == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        WRITE(*,*) "semiss is NOT IDENTICAL."
        WRITE(*,*) count( outstate_semiss /= semiss), " of ", size( semiss ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_semiss - semiss)**2)/real(size(outstate_semiss)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_semiss - semiss))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_semiss - semiss))
        WRITE(*,*) "Mean value of kernel-generated outstate_semiss is ", sum(semiss)/real(size(semiss))
        WRITE(*,*) "Mean value of original outstate_semiss is ", sum(outstate_semiss)/real(size(outstate_semiss))
        WRITE(*,*) ""
    END IF
    IF ( ALL( outstate_pavel == pavel ) ) THEN
        WRITE(*,*) "All elements of pavel are IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_pavel
        !WRITE(*,*) "KERNEL: ", pavel
        IF ( ALL( outstate_pavel == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        WRITE(*,*) "pavel is NOT IDENTICAL."
        WRITE(*,*) count( outstate_pavel /= pavel), " of ", size( pavel ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_pavel - pavel)**2)/real(size(outstate_pavel)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_pavel - pavel))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_pavel - pavel))
        WRITE(*,*) "Mean value of kernel-generated outstate_pavel is ", sum(pavel)/real(size(pavel))
        WRITE(*,*) "Mean value of original outstate_pavel is ", sum(outstate_pavel)/real(size(outstate_pavel))
        WRITE(*,*) ""
    END IF
    IF ( ALL( outstate_dgesmc == dgesmc ) ) THEN
        WRITE(*,*) "All elements of dgesmc are IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_dgesmc
        !WRITE(*,*) "KERNEL: ", dgesmc
        IF ( ALL( outstate_dgesmc == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        WRITE(*,*) "dgesmc is NOT IDENTICAL."
        WRITE(*,*) count( outstate_dgesmc /= dgesmc), " of ", size( dgesmc ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_dgesmc - dgesmc)**2)/real(size(outstate_dgesmc)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_dgesmc - dgesmc))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_dgesmc - dgesmc))
        WRITE(*,*) "Mean value of kernel-generated outstate_dgesmc is ", sum(dgesmc)/real(size(dgesmc))
        WRITE(*,*) "Mean value of original outstate_dgesmc is ", sum(outstate_dgesmc)/real(size(outstate_dgesmc))
        WRITE(*,*) ""
    END IF
    IF ( outstate_pwvcm == pwvcm ) THEN
        WRITE(*,*) "pwvcm is IDENTICAL( ", outstate_pwvcm, " )."
    ELSE
        WRITE(*,*) "pwvcm is NOT IDENTICAL."
        WRITE(*,*) "STATE : ", outstate_pwvcm
        WRITE(*,*) "KERNEL: ", pwvcm
    END IF
    IF ( outstate_inflag == inflag ) THEN
        WRITE(*,*) "inflag is IDENTICAL( ", outstate_inflag, " )."
    ELSE
        WRITE(*,*) "inflag is NOT IDENTICAL."
        WRITE(*,*) "STATE : ", outstate_inflag
        WRITE(*,*) "KERNEL: ", inflag
    END IF
    IF ( ALL( outstate_wx == wx ) ) THEN
        WRITE(*,*) "All elements of wx are IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_wx
        !WRITE(*,*) "KERNEL: ", wx
        IF ( ALL( outstate_wx == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        WRITE(*,*) "wx is NOT IDENTICAL."
        WRITE(*,*) count( outstate_wx /= wx), " of ", size( wx ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_wx - wx)**2)/real(size(outstate_wx)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_wx - wx))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_wx - wx))
        WRITE(*,*) "Mean value of kernel-generated outstate_wx is ", sum(wx)/real(size(wx))
        WRITE(*,*) "Mean value of original outstate_wx is ", sum(outstate_wx)/real(size(outstate_wx))
        WRITE(*,*) ""
    END IF
    IF ( ALL( outstate_taua == taua ) ) THEN
        WRITE(*,*) "All elements of taua are IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_taua
        !WRITE(*,*) "KERNEL: ", taua
        IF ( ALL( outstate_taua == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        WRITE(*,*) "taua is NOT IDENTICAL."
        WRITE(*,*) count( outstate_taua /= taua), " of ", size( taua ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_taua - taua)**2)/real(size(outstate_taua)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_taua - taua))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_taua - taua))
        WRITE(*,*) "Mean value of kernel-generated outstate_taua is ", sum(taua)/real(size(taua))
        WRITE(*,*) "Mean value of original outstate_taua is ", sum(outstate_taua)/real(size(outstate_taua))
        WRITE(*,*) ""
    END IF
    IF ( ALL( outstate_taucmc == taucmc ) ) THEN
        WRITE(*,*) "All elements of taucmc are IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_taucmc
        !WRITE(*,*) "KERNEL: ", taucmc
        IF ( ALL( outstate_taucmc == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        WRITE(*,*) "taucmc is NOT IDENTICAL."
        WRITE(*,*) count( outstate_taucmc /= taucmc), " of ", size( taucmc ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_taucmc - taucmc)**2)/real(size(outstate_taucmc)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_taucmc - taucmc))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_taucmc - taucmc))
        WRITE(*,*) "Mean value of kernel-generated outstate_taucmc is ", sum(taucmc)/real(size(taucmc))
        WRITE(*,*) "Mean value of original outstate_taucmc is ", sum(outstate_taucmc)/real(size(outstate_taucmc))
        WRITE(*,*) ""
    END IF


    ! DEALLOCATE INSTATE


    ! DEALLOCATE OUTSTATE


    ! DEALLOCATE CALLEE INSTATE
    ! DEALLOCATE INSTATE


    ! DEALLOCATE CALEE OUTSTATE
    ! DEALLOCATE OUTSTATE


    CONTAINS


    ! KERNEL SUBPROGRAM
    subroutine inatm (iplon, nlay, icld, iaer,               play, plev, tlay, tlev, tsfc, h2ovmr,               o3vmr, co2vmr, ch4vmr, o2vmr, n2ovmr, cfc11vmr, cfc12vmr,               cfc22vmr, ccl4vmr, emis, inflglw, iceflglw, liqflglw,               cldfmcl, taucmcl, ciwpmcl, clwpmcl, reicmcl, relqmcl, tauaer,               pavel, pz, tavel, tz, tbound, semiss, coldry,               wkl, wbrodl, wx, pwvcm, inflag, iceflag, liqflag,               cldfmc, taucmc, ciwpmc, clwpmc, reicmc, dgesmc, relqmc, taua)
        integer, intent(in) :: iplon
        integer, intent(in) :: nlay
        integer, intent(in) :: icld
        integer, intent(in) :: iaer
        real(kind=r8), intent(in) :: play(:,:)
        real(kind=r8), intent(in) :: plev(:,:)
        real(kind=r8), intent(in) :: tlay(:,:)
        real(kind=r8), intent(in) :: tlev(:,:)
        real(kind=r8), intent(in) :: tsfc(:)
        real(kind=r8), intent(in) :: h2ovmr(:,:)
        real(kind=r8), intent(in) :: o3vmr(:,:)
        real(kind=r8), intent(in) :: co2vmr(:,:)
        real(kind=r8), intent(in) :: ch4vmr(:,:)
        real(kind=r8), intent(in) :: o2vmr(:,:)
        real(kind=r8), intent(in) :: n2ovmr(:,:)
        real(kind=r8), intent(in) :: cfc11vmr(:,:)
        real(kind=r8), intent(in) :: cfc12vmr(:,:)
        real(kind=r8), intent(in) :: cfc22vmr(:,:)
        real(kind=r8), intent(in) :: ccl4vmr(:,:)
        real(kind=r8), intent(in) :: emis(:,:)
        integer, intent(in) :: inflglw
        integer, intent(in) :: iceflglw
        integer, intent(in) :: liqflglw
        real(kind=r8), intent(in) :: cldfmcl(:,:,:)
        real(kind=r8), intent(in) :: ciwpmcl(:,:,:)
        real(kind=r8), intent(in) :: clwpmcl(:,:,:)
        real(kind=r8), intent(in) :: reicmcl(:,:)
        real(kind=r8), intent(in) :: relqmcl(:,:)
        real(kind=r8), intent(in) :: taucmcl(:,:,:)
        real(kind=r8), intent(in) :: tauaer(:,:,:)
        real(kind=r8), intent(out) :: pavel(:)
        real(kind=r8), intent(out) :: tavel(:)
        real(kind=r8), intent(out) :: pz(0:)
        real(kind=r8), intent(out) :: tz(0:)
        real(kind=r8), intent(out) :: tbound
        real(kind=r8), intent(out) :: coldry(:)
        real(kind=r8), intent(out) :: wbrodl(:)
        real(kind=r8), intent(out) :: wkl(:,:)
        real(kind=r8), intent(out) :: wx(:,:)
        real(kind=r8), intent(out) :: pwvcm
        real(kind=r8), intent(out) :: semiss(:)
        integer, intent(out) :: inflag
        integer, intent(out) :: iceflag
        integer, intent(out) :: liqflag
        real(kind=r8), intent(out) :: cldfmc(:,:)
        real(kind=r8), intent(out) :: ciwpmc(:,:)
        real(kind=r8), intent(out) :: clwpmc(:,:)
        real(kind=r8), intent(out) :: relqmc(:)
        real(kind=r8), intent(out) :: reicmc(:)
        real(kind=r8), intent(out) :: dgesmc(:)
        real(kind=r8), intent(out) :: taucmc(:,:)
        real(kind=r8), intent(out) :: taua(:,:)
        real(kind=r8), parameter :: amd = 28.9660_r8
        real(kind=r8), parameter :: amw = 18.0160_r8
        real(kind=r8), parameter :: amdw = 1.607793_r8
        real(kind=r8), parameter :: amdc = 0.658114_r8
        real(kind=r8), parameter :: amdo = 0.603428_r8
        real(kind=r8), parameter :: amdm = 1.805423_r8
        real(kind=r8), parameter :: amdn = 0.658090_r8
        real(kind=r8), parameter :: amdc1 = 0.210852_r8
        real(kind=r8), parameter :: amdc2 = 0.239546_r8
        real(kind=r8), parameter :: sbc = 5.67e-08_r8
        integer :: isp, l, ix, n, imol, ib, ig
        real(kind=r8) :: amm, amttl, wvttl, wvsh, summol
        wkl(:,:) = 0.0_r8
        wx(:,:) = 0.0_r8
        cldfmc(:,:) = 0.0_r8
        taucmc(:,:) = 0.0_r8
        ciwpmc(:,:) = 0.0_r8
        clwpmc(:,:) = 0.0_r8
        reicmc(:) = 0.0_r8
        dgesmc(:) = 0.0_r8
        relqmc(:) = 0.0_r8
        taua(:,:) = 0.0_r8
        amttl = 0.0_r8
        wvttl = 0.0_r8
        tbound = tsfc(iplon)
        pz(0) = plev(iplon,nlay+1)
        tz(0) = tlev(iplon,nlay+1)
        do l = 1, nlay
            pavel(l) = play(iplon,nlay-l+1)
            tavel(l) = tlay(iplon,nlay-l+1)
            pz(l) = plev(iplon,nlay-l+1)
            tz(l) = tlev(iplon,nlay-l+1)
            wkl(1,l) = h2ovmr(iplon,nlay-l+1)
            wkl(2,l) = co2vmr(iplon,nlay-l+1)
            wkl(3,l) = o3vmr(iplon,nlay-l+1)
            wkl(4,l) = n2ovmr(iplon,nlay-l+1)
            wkl(6,l) = ch4vmr(iplon,nlay-l+1)
            wkl(7,l) = o2vmr(iplon,nlay-l+1)
            amm = (1._r8 - wkl(1,l)) * amd + wkl(1,l) * amw
            coldry(l) = (pz(l-1)-pz(l)) * 1.e3_r8 * avogad /                      (1.e2_r8 * grav * amm * (1._r8 + wkl(1,l)))
            wx(1,l) = ccl4vmr(iplon,nlay-l+1)
            wx(2,l) = cfc11vmr(iplon,nlay-l+1)
            wx(3,l) = cfc12vmr(iplon,nlay-l+1)
            wx(4,l) = cfc22vmr(iplon,nlay-l+1)
        enddo
        coldry(nlay) = (pz(nlay-1)) * 1.e3_r8 * avogad /                         (1.e2_r8 * grav * amm * (1._r8 + wkl(1,nlay-1)))
        do l = 1, nlay
            summol = 0.0_r8
            do imol = 2, nmol
                summol = summol + wkl(imol,l)
            enddo
            wbrodl(l) = coldry(l) * (1._r8 - summol)
            do imol = 1, nmol
                wkl(imol,l) = coldry(l) * wkl(imol,l)
            enddo
            amttl = amttl + coldry(l)+wkl(1,l)
            wvttl = wvttl + wkl(1,l)
            do ix = 1,maxxsec
                if (ixindx(ix) .ne. 0) then
                    wx(ixindx(ix),l) = coldry(l) * wx(ix,l) * 1.e-20_r8
                endif
            enddo
        enddo
        wvsh = (amw * wvttl) / (amd * amttl)
        pwvcm = wvsh * (1.e3_r8 * pz(0)) / (1.e2_r8 * grav)
        do n=1,nbndlw
            semiss(n) = emis(iplon,n)
        enddo
        if (iaer .ge. 1) then
            do l = 1, nlay-1
                do ib = 1, nbndlw
                    taua(l,ib) = tauaer(iplon,nlay-l,ib)
                enddo
            enddo
        endif
        if (icld .ge. 1) then
            inflag = inflglw
            iceflag = iceflglw
            liqflag = liqflglw
            do l = 1, nlay-1
                do ig = 1, ngptlw
                    cldfmc(ig,l) = cldfmcl(ig,iplon,nlay-l)
                    taucmc(ig,l) = taucmcl(ig,iplon,nlay-l)
                    ciwpmc(ig,l) = ciwpmcl(ig,iplon,nlay-l)
                    clwpmc(ig,l) = clwpmcl(ig,iplon,nlay-l)
                enddo
                reicmc(l) = reicmcl(iplon,nlay-l)
                if (iceflag .eq. 3) then
                    dgesmc(l) = 1.5396_r8 * reicmcl(iplon,nlay-l)
                endif
                relqmc(l) = relqmcl(iplon,nlay-l)
            enddo
            cldfmc(:,nlay) = 0.0_r8
            taucmc(:,nlay) = 0.0_r8
            ciwpmc(:,nlay) = 0.0_r8
            clwpmc(:,nlay) = 0.0_r8
            reicmc(nlay) = 0.0_r8
            dgesmc(nlay) = 0.0_r8
            relqmc(nlay) = 0.0_r8
            taua(nlay,:) = 0.0_r8
        endif
    end subroutine inatm


    END SUBROUTINE kernel_driver


    ! RESOLVER SUBPROGRAMS
    
    FUNCTION kgen_get_newunit(seed) RESULT(new_unit)
       INTEGER, PARAMETER :: UNIT_MIN=100, UNIT_MAX=1000000
       LOGICAL :: is_opened
       INTEGER :: nunit, new_unit, counter
       INTEGER, INTENT(IN) :: seed
    
       new_unit = -1
       
       DO counter=UNIT_MIN, UNIT_MAX
           inquire(UNIT=counter, OPENED=is_opened)
           IF (.NOT. is_opened) THEN
               new_unit = counter
               EXIT
           END IF
       END DO
    END FUNCTION

    
    SUBROUTINE kgen_error_stop( msg )
        IMPLICIT NONE
        CHARACTER(LEN=*), INTENT(IN) :: msg
    
        WRITE (*,*) msg
        STOP 1
    END SUBROUTINE
    END MODULE

    PROGRAM kernel_inatm
    USE resolvers
    USE subprograms

    IMPLICIT NONE


    INTEGER :: kgen_mpi_rank
    CHARACTER(LEN=16) ::kgen_mpi_rank_conv
    INTEGER, DIMENSION(1), PARAMETER :: kgen_mpi_rank_at = (/ 0 /)
    INTEGER :: kgen_ierr, kgen_unit
    INTEGER :: kgen_repeat_counter
    INTEGER :: kgen_counter
    CHARACTER(LEN=16) :: kgen_counter_conv
    INTEGER, DIMENSION(1), PARAMETER :: kgen_counter_at = (/ 1 /)
    CHARACTER(LEN=1024) :: kgen_filepath
    INTEGER, DIMENSION(2,10) :: kgen_bound

    ! DRIVER SPECS
    REAL(KIND = r8), ALLOCATABLE :: taucmcl(:, :, :)
    REAL(KIND = r8), ALLOCATABLE :: ch4vmr(:, :)
    INTEGER :: icld
    REAL(KIND = r8), ALLOCATABLE :: emis(:, :)
    REAL(KIND = r8), ALLOCATABLE :: tlay(:, :)
    REAL(KIND = r8), ALLOCATABLE :: reicmcl(:, :)
    INTEGER :: nlay
    REAL(KIND = r8), ALLOCATABLE :: cfc11vmr(:, :)
    REAL(KIND = r8), ALLOCATABLE :: tsfc(:)
    REAL(KIND = r8), ALLOCATABLE :: relqmcl(:, :)
    REAL(KIND = r8), ALLOCATABLE :: o3vmr(:, :)
    REAL(KIND = r8), ALLOCATABLE :: n2ovmr(:, :)
    REAL(KIND = r8), ALLOCATABLE :: plev(:, :)
    REAL(KIND = r8), ALLOCATABLE :: play(:, :)
    REAL(KIND = r8), ALLOCATABLE :: tauaer(:, :, :)
    REAL(KIND = r8), ALLOCATABLE :: clwpmcl(:, :, :)
    REAL(KIND = r8), ALLOCATABLE :: o2vmr(:, :)
    REAL(KIND = r8), ALLOCATABLE :: co2vmr(:, :)
    REAL(KIND = r8), ALLOCATABLE :: ccl4vmr(:, :)
    INTEGER :: iceflglw
    REAL(KIND = r8), ALLOCATABLE :: cfc12vmr(:, :)
    REAL(KIND = r8), ALLOCATABLE :: tlev(:, :)
    REAL(KIND = r8), ALLOCATABLE :: h2ovmr(:, :)
    INTEGER :: inflglw
    REAL(KIND = r8), ALLOCATABLE :: ciwpmcl(:, :, :)
    REAL(KIND = r8), ALLOCATABLE :: cldfmcl(:, :, :)
    INTEGER :: liqflglw
    REAL(KIND = r8), ALLOCATABLE :: cfc22vmr(:, :)


    DO kgen_repeat_counter = 1, 1
        kgen_counter = kgen_counter_at(mod(kgen_repeat_counter, 1)+1)
        WRITE( kgen_counter_conv, * ) kgen_counter
        kgen_mpi_rank = kgen_mpi_rank_at(mod(kgen_repeat_counter, 1)+1)
        WRITE( kgen_mpi_rank_conv, * ) kgen_mpi_rank


        kgen_filepath = "../data/inatm." // trim(adjustl(kgen_counter_conv)) // "." // trim(adjustl(kgen_mpi_rank_conv))
        kgen_unit = kgen_get_newunit(kgen_mpi_rank+kgen_counter)
        OPEN (UNIT=kgen_unit, FILE=kgen_filepath, STATUS="OLD", ACCESS="STREAM", FORM="UNFORMATTED", ACTION="READ", IOSTAT=kgen_ierr, CONVERT="BIG_ENDIAN")
        WRITE (*,*) "Kernel output is being verified against " // trim(adjustl(kgen_filepath))
        IF ( kgen_ierr /= 0 ) THEN
            CALL kgen_error_stop( "FILE OPEN ERROR: " // trim(adjustl(kgen_filepath)) )
        END IF


        ! READ DRIVER INSTATE
        READ(UNIT = kgen_unit) kgen_bound(1, 1)
        READ(UNIT = kgen_unit) kgen_bound(2, 1)
        READ(UNIT = kgen_unit) kgen_bound(1, 2)
        READ(UNIT = kgen_unit) kgen_bound(2, 2)
        READ(UNIT = kgen_unit) kgen_bound(1, 3)
        READ(UNIT = kgen_unit) kgen_bound(2, 3)
        ALLOCATE(taucmcl(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1, kgen_bound(2, 3) - kgen_bound(1, 3) + 1))
        READ(UNIT = kgen_unit) taucmcl
        READ(UNIT = kgen_unit) kgen_bound(1, 1)
        READ(UNIT = kgen_unit) kgen_bound(2, 1)
        READ(UNIT = kgen_unit) kgen_bound(1, 2)
        READ(UNIT = kgen_unit) kgen_bound(2, 2)
        ALLOCATE(ch4vmr(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
        READ(UNIT = kgen_unit) ch4vmr
        READ(UNIT = kgen_unit) icld
        READ(UNIT = kgen_unit) kgen_bound(1, 1)
        READ(UNIT = kgen_unit) kgen_bound(2, 1)
        READ(UNIT = kgen_unit) kgen_bound(1, 2)
        READ(UNIT = kgen_unit) kgen_bound(2, 2)
        ALLOCATE(emis(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
        READ(UNIT = kgen_unit) emis
        READ(UNIT = kgen_unit) kgen_bound(1, 1)
        READ(UNIT = kgen_unit) kgen_bound(2, 1)
        READ(UNIT = kgen_unit) kgen_bound(1, 2)
        READ(UNIT = kgen_unit) kgen_bound(2, 2)
        ALLOCATE(tlay(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
        READ(UNIT = kgen_unit) tlay
        READ(UNIT = kgen_unit) kgen_bound(1, 1)
        READ(UNIT = kgen_unit) kgen_bound(2, 1)
        READ(UNIT = kgen_unit) kgen_bound(1, 2)
        READ(UNIT = kgen_unit) kgen_bound(2, 2)
        ALLOCATE(reicmcl(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
        READ(UNIT = kgen_unit) reicmcl
        READ(UNIT = kgen_unit) nlay
        READ(UNIT = kgen_unit) kgen_bound(1, 1)
        READ(UNIT = kgen_unit) kgen_bound(2, 1)
        READ(UNIT = kgen_unit) kgen_bound(1, 2)
        READ(UNIT = kgen_unit) kgen_bound(2, 2)
        ALLOCATE(cfc11vmr(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
        READ(UNIT = kgen_unit) cfc11vmr
        READ(UNIT = kgen_unit) kgen_bound(1, 1)
        READ(UNIT = kgen_unit) kgen_bound(2, 1)
        ALLOCATE(tsfc(kgen_bound(2, 1) - kgen_bound(1, 1) + 1))
        READ(UNIT = kgen_unit) tsfc
        READ(UNIT = kgen_unit) kgen_bound(1, 1)
        READ(UNIT = kgen_unit) kgen_bound(2, 1)
        READ(UNIT = kgen_unit) kgen_bound(1, 2)
        READ(UNIT = kgen_unit) kgen_bound(2, 2)
        ALLOCATE(relqmcl(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
        READ(UNIT = kgen_unit) relqmcl
        READ(UNIT = kgen_unit) kgen_bound(1, 1)
        READ(UNIT = kgen_unit) kgen_bound(2, 1)
        READ(UNIT = kgen_unit) kgen_bound(1, 2)
        READ(UNIT = kgen_unit) kgen_bound(2, 2)
        ALLOCATE(o3vmr(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
        READ(UNIT = kgen_unit) o3vmr
        READ(UNIT = kgen_unit) kgen_bound(1, 1)
        READ(UNIT = kgen_unit) kgen_bound(2, 1)
        READ(UNIT = kgen_unit) kgen_bound(1, 2)
        READ(UNIT = kgen_unit) kgen_bound(2, 2)
        ALLOCATE(n2ovmr(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
        READ(UNIT = kgen_unit) n2ovmr
        READ(UNIT = kgen_unit) kgen_bound(1, 1)
        READ(UNIT = kgen_unit) kgen_bound(2, 1)
        READ(UNIT = kgen_unit) kgen_bound(1, 2)
        READ(UNIT = kgen_unit) kgen_bound(2, 2)
        ALLOCATE(plev(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
        READ(UNIT = kgen_unit) plev
        READ(UNIT = kgen_unit) kgen_bound(1, 1)
        READ(UNIT = kgen_unit) kgen_bound(2, 1)
        READ(UNIT = kgen_unit) kgen_bound(1, 2)
        READ(UNIT = kgen_unit) kgen_bound(2, 2)
        ALLOCATE(play(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
        READ(UNIT = kgen_unit) play
        READ(UNIT = kgen_unit) kgen_bound(1, 1)
        READ(UNIT = kgen_unit) kgen_bound(2, 1)
        READ(UNIT = kgen_unit) kgen_bound(1, 2)
        READ(UNIT = kgen_unit) kgen_bound(2, 2)
        READ(UNIT = kgen_unit) kgen_bound(1, 3)
        READ(UNIT = kgen_unit) kgen_bound(2, 3)
        ALLOCATE(tauaer(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1, kgen_bound(2, 3) - kgen_bound(1, 3) + 1))
        READ(UNIT = kgen_unit) tauaer
        READ(UNIT = kgen_unit) kgen_bound(1, 1)
        READ(UNIT = kgen_unit) kgen_bound(2, 1)
        READ(UNIT = kgen_unit) kgen_bound(1, 2)
        READ(UNIT = kgen_unit) kgen_bound(2, 2)
        READ(UNIT = kgen_unit) kgen_bound(1, 3)
        READ(UNIT = kgen_unit) kgen_bound(2, 3)
        ALLOCATE(clwpmcl(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1, kgen_bound(2, 3) - kgen_bound(1, 3) + 1))
        READ(UNIT = kgen_unit) clwpmcl
        READ(UNIT = kgen_unit) kgen_bound(1, 1)
        READ(UNIT = kgen_unit) kgen_bound(2, 1)
        READ(UNIT = kgen_unit) kgen_bound(1, 2)
        READ(UNIT = kgen_unit) kgen_bound(2, 2)
        ALLOCATE(o2vmr(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
        READ(UNIT = kgen_unit) o2vmr
        READ(UNIT = kgen_unit) kgen_bound(1, 1)
        READ(UNIT = kgen_unit) kgen_bound(2, 1)
        READ(UNIT = kgen_unit) kgen_bound(1, 2)
        READ(UNIT = kgen_unit) kgen_bound(2, 2)
        ALLOCATE(co2vmr(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
        READ(UNIT = kgen_unit) co2vmr
        READ(UNIT = kgen_unit) kgen_bound(1, 1)
        READ(UNIT = kgen_unit) kgen_bound(2, 1)
        READ(UNIT = kgen_unit) kgen_bound(1, 2)
        READ(UNIT = kgen_unit) kgen_bound(2, 2)
        ALLOCATE(ccl4vmr(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
        READ(UNIT = kgen_unit) ccl4vmr
        READ(UNIT = kgen_unit) iceflglw
        READ(UNIT = kgen_unit) kgen_bound(1, 1)
        READ(UNIT = kgen_unit) kgen_bound(2, 1)
        READ(UNIT = kgen_unit) kgen_bound(1, 2)
        READ(UNIT = kgen_unit) kgen_bound(2, 2)
        ALLOCATE(cfc12vmr(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
        READ(UNIT = kgen_unit) cfc12vmr
        READ(UNIT = kgen_unit) kgen_bound(1, 1)
        READ(UNIT = kgen_unit) kgen_bound(2, 1)
        READ(UNIT = kgen_unit) kgen_bound(1, 2)
        READ(UNIT = kgen_unit) kgen_bound(2, 2)
        ALLOCATE(tlev(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
        READ(UNIT = kgen_unit) tlev
        READ(UNIT = kgen_unit) kgen_bound(1, 1)
        READ(UNIT = kgen_unit) kgen_bound(2, 1)
        READ(UNIT = kgen_unit) kgen_bound(1, 2)
        READ(UNIT = kgen_unit) kgen_bound(2, 2)
        ALLOCATE(h2ovmr(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
        READ(UNIT = kgen_unit) h2ovmr
        READ(UNIT = kgen_unit) inflglw
        READ(UNIT = kgen_unit) kgen_bound(1, 1)
        READ(UNIT = kgen_unit) kgen_bound(2, 1)
        READ(UNIT = kgen_unit) kgen_bound(1, 2)
        READ(UNIT = kgen_unit) kgen_bound(2, 2)
        READ(UNIT = kgen_unit) kgen_bound(1, 3)
        READ(UNIT = kgen_unit) kgen_bound(2, 3)
        ALLOCATE(ciwpmcl(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1, kgen_bound(2, 3) - kgen_bound(1, 3) + 1))
        READ(UNIT = kgen_unit) ciwpmcl
        READ(UNIT = kgen_unit) kgen_bound(1, 1)
        READ(UNIT = kgen_unit) kgen_bound(2, 1)
        READ(UNIT = kgen_unit) kgen_bound(1, 2)
        READ(UNIT = kgen_unit) kgen_bound(2, 2)
        READ(UNIT = kgen_unit) kgen_bound(1, 3)
        READ(UNIT = kgen_unit) kgen_bound(2, 3)
        ALLOCATE(cldfmcl(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1, kgen_bound(2, 3) - kgen_bound(1, 3) + 1))
        READ(UNIT = kgen_unit) cldfmcl
        READ(UNIT = kgen_unit) liqflglw
        READ(UNIT = kgen_unit) kgen_bound(1, 1)
        READ(UNIT = kgen_unit) kgen_bound(2, 1)
        READ(UNIT = kgen_unit) kgen_bound(1, 2)
        READ(UNIT = kgen_unit) kgen_bound(2, 2)
        ALLOCATE(cfc22vmr(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
        READ(UNIT = kgen_unit) cfc22vmr


        ! KERNEL DRIVER RUN
        CALL kernel_driver(taucmcl, ch4vmr, icld, emis, tlay, reicmcl, nlay, cfc11vmr, tsfc, relqmcl, o3vmr, n2ovmr, plev, play, tauaer, clwpmcl, o2vmr, co2vmr, ccl4vmr, iceflglw, cfc12vmr, tlev, h2ovmr, inflglw, ciwpmcl, cldfmcl, liqflglw, cfc22vmr, kgen_unit)

        CLOSE (UNIT=kgen_unit)

        WRITE (*,*)
    END DO

    END PROGRAM kernel_inatm
