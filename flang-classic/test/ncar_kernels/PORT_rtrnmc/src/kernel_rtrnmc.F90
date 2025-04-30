    MODULE resolvers

    ! RESOLVER SPECS
    INTEGER, PARAMETER :: r8 = selected_real_kind(12)
    INTEGER, PARAMETER :: ngptlw = 140
    INTEGER, PARAMETER :: nbndlw = 16
    REAL(KIND = r8), PARAMETER :: tblint = 10000.0_r8
    INTEGER, PARAMETER :: ntbl = 10000

    END MODULE

    PROGRAM kernel_rtrnmc
    USE resolvers

    IMPLICIT NONE


    INTEGER :: kgen_mpi_rank
    CHARACTER(LEN=16) ::kgen_mpi_rank_conv
    INTEGER, DIMENSION(2), PARAMETER :: kgen_mpi_rank_at = (/ 0,1 /)
    INTEGER :: kgen_ierr, kgen_unit, kgen_get_newunit
    INTEGER :: kgen_repeat_counter
    INTEGER :: kgen_counter
    CHARACTER(LEN=16) :: kgen_counter_conv
    INTEGER, DIMENSION(1), PARAMETER :: kgen_counter_at = (/ 10 /)
    CHARACTER(LEN=1024) :: kgen_filepath
    INTEGER, DIMENSION(2,10) :: kgen_bound

    ! DRIVER SPECS
    INTEGER :: nlay

    DO kgen_repeat_counter = 1, 2
        kgen_counter = kgen_counter_at(mod(kgen_repeat_counter, 1)+1)
        WRITE( kgen_counter_conv, * ) kgen_counter
        kgen_mpi_rank = kgen_mpi_rank_at(mod(kgen_repeat_counter, 2)+1)
        WRITE( kgen_mpi_rank_conv, * ) kgen_mpi_rank

        kgen_filepath = "../data/rtrnmc." // trim(adjustl(kgen_counter_conv)) // "." // trim(adjustl(kgen_mpi_rank_conv))
        kgen_unit = kgen_get_newunit(kgen_mpi_rank+kgen_counter)
        OPEN (UNIT=kgen_unit, FILE=kgen_filepath, STATUS="OLD", ACCESS="STREAM", FORM="UNFORMATTED", ACTION="READ", IOSTAT=kgen_ierr, CONVERT="BIG_ENDIAN")
        WRITE (*,*) "Kernel output is being verified against " // trim(adjustl(kgen_filepath))
        IF ( kgen_ierr /= 0 ) THEN
            CALL kgen_error_stop( "FILE OPEN ERROR: " // trim(adjustl(kgen_filepath)) )
        END IF
        ! READ DRIVER INSTATE

        READ(UNIT = kgen_unit) nlay

        ! KERNEL DRIVER RUN
        CALL kernel_driver(nlay, kgen_unit)
        CLOSE (UNIT=kgen_unit)

        WRITE (*,*)
    END DO
    END PROGRAM kernel_rtrnmc

    ! KERNEL DRIVER SUBROUTINE
    SUBROUTINE kernel_driver(nlay, kgen_unit)
    USE resolvers

    IMPLICIT NONE
    INTEGER, INTENT(IN) :: kgen_unit
    INTEGER, DIMENSION(2,10) :: kgen_bound

    ! STATE SPECS
    CHARACTER*18 :: hvrrtc
    INTEGER, INTENT(IN) :: nlay
    REAL(KIND = r8) :: pwvcm
    REAL(KIND = r8) :: bpade
    INTEGER :: ncbands
    REAL(KIND = r8), DIMENSION(0 : ntbl) :: exp_tbl
    REAL(KIND = r8) :: totdflux(0 : nlay)
    REAL(KIND = r8) :: fnetc(0 : nlay)
    REAL(KIND = r8) :: htr(0 : nlay)
    REAL(KIND = r8) :: plankbnd(nbndlw)
    INTEGER :: istart
    INTEGER :: ngb(ngptlw)
    REAL(KIND = r8) :: pz(0 : nlay)
    REAL(KIND = r8) :: totdclfl(0 : nlay)
    REAL(KIND = r8) :: fracs(nlay, ngptlw)
    INTEGER :: ngs(nbndlw)
    REAL(KIND = r8) :: totdfluxs(nbndlw, 0 : nlay)
    REAL(KIND = r8) :: fluxfac
    REAL(KIND = r8) :: heatfac
    REAL(KIND = r8) :: taut(nlay, ngptlw)
    REAL(KIND = r8) :: semiss(nbndlw)
    REAL(KIND = r8) :: totufluxs(nbndlw, 0 : nlay)
    REAL(KIND = r8) :: taucmc(ngptlw, nlay)
    REAL(KIND = r8) :: planklay(nlay, nbndlw)
    REAL(KIND = r8) :: totuclfl(0 : nlay)
    REAL(KIND = r8) :: htrc(0 : nlay)
    REAL(KIND = r8), DIMENSION(0 : ntbl) :: tfn_tbl
    REAL(KIND = r8) :: fnet(0 : nlay)
    REAL(KIND = r8) :: planklev(0 : nlay, nbndlw)
    INTEGER :: iout
    REAL(KIND = r8) :: cldfmc(ngptlw, nlay)
    REAL(KIND = r8) :: totuflux(0 : nlay)
    REAL(KIND = r8), DIMENSION(0 : ntbl) :: tau_tbl
    REAL(KIND = r8) :: delwave(nbndlw)
    INTEGER :: iend
    INTEGER :: outstate_ncbands
    REAL(KIND = r8) :: outstate_totdflux(0 : nlay)
    REAL(KIND = r8) :: outstate_fnetc(0 : nlay)
    REAL(KIND = r8) :: outstate_htr(0 : nlay)
    REAL(KIND = r8) :: outstate_totdclfl(0 : nlay)
    REAL(KIND = r8) :: outstate_totdfluxs(nbndlw, 0 : nlay)
    REAL(KIND = r8) :: outstate_totufluxs(nbndlw, 0 : nlay)
    REAL(KIND = r8) :: outstate_totuclfl(0 : nlay)
    REAL(KIND = r8) :: outstate_htrc(0 : nlay)
    REAL(KIND = r8) :: outstate_fnet(0 : nlay)
    REAL(KIND = r8) :: outstate_totuflux(0 : nlay)

    LOGICAL :: lstatus = .TRUE.
    ! READ CALLER INSTATE

    READ(UNIT = kgen_unit) pwvcm
    READ(UNIT = kgen_unit) ncbands
    READ(UNIT = kgen_unit) plankbnd
    READ(UNIT = kgen_unit) istart
    READ(UNIT = kgen_unit) pz
    READ(UNIT = kgen_unit) fracs
    READ(UNIT = kgen_unit) taut
    READ(UNIT = kgen_unit) semiss
    READ(UNIT = kgen_unit) taucmc
    READ(UNIT = kgen_unit) planklay
    READ(UNIT = kgen_unit) planklev
    READ(UNIT = kgen_unit) iout
    READ(UNIT = kgen_unit) cldfmc
    READ(UNIT = kgen_unit) iend
    ! READ CALLEE INSTATE

    READ(UNIT = kgen_unit) hvrrtc
    READ(UNIT = kgen_unit) bpade
    READ(UNIT = kgen_unit) exp_tbl
    READ(UNIT = kgen_unit) ngb
    READ(UNIT = kgen_unit) ngs
    READ(UNIT = kgen_unit) fluxfac
    READ(UNIT = kgen_unit) heatfac
    READ(UNIT = kgen_unit) tfn_tbl
    READ(UNIT = kgen_unit) tau_tbl
    READ(UNIT = kgen_unit) delwave
    ! READ CALLEE OUTSTATE

    ! READ CALLER OUTSTATE

    READ(UNIT = kgen_unit) outstate_ncbands
    READ(UNIT = kgen_unit) outstate_totdflux
    READ(UNIT = kgen_unit) outstate_fnetc
    READ(UNIT = kgen_unit) outstate_htr
    READ(UNIT = kgen_unit) outstate_totdclfl
    READ(UNIT = kgen_unit) outstate_totdfluxs
    READ(UNIT = kgen_unit) outstate_totufluxs
    READ(UNIT = kgen_unit) outstate_totuclfl
    READ(UNIT = kgen_unit) outstate_htrc
    READ(UNIT = kgen_unit) outstate_fnet
    READ(UNIT = kgen_unit) outstate_totuflux

    ! KERNEL RUN
    CALL rtrnmc(nlay, istart, iend, iout, pz, semiss, ncbands, cldfmc, &
                taucmc, planklay, planklev, plankbnd, pwvcm, fracs, taut, &
                totuflux, totdflux, fnet, htr, totuclfl, totdclfl, fnetc, &
                htrc, totufluxs, totdfluxs)

    ! STATE VERIFICATION
    IF ( outstate_ncbands == ncbands ) THEN
        WRITE(*,*) "ncbands is IDENTICAL( ", outstate_ncbands, " )."
    ELSE
        lstatus = .FALSE.
        WRITE(*,*) "ncbands is NOT IDENTICAL."
        WRITE(*,*) "STATE : ", outstate_ncbands
        WRITE(*,*) "KERNEL: ", ncbands
    END IF
    IF ( ALL( outstate_totdflux == totdflux ) ) THEN
        WRITE(*,*) "All elements of totdflux are IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_totdflux
        !WRITE(*,*) "KERNEL: ", totdflux
        IF ( ALL( outstate_totdflux == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        lstatus = .FALSE.
        WRITE(*,*) "totdflux is NOT IDENTICAL."
        WRITE(*,*) count( outstate_totdflux /= totdflux), " of ", size( totdflux ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_totdflux - totdflux)**2)/real(size(outstate_totdflux)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_totdflux - totdflux))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_totdflux - totdflux))
        WRITE(*,*) "Mean value of kernel-generated outstate_totdflux is ", sum(totdflux)/real(size(totdflux))
        WRITE(*,*) "Mean value of original outstate_totdflux is ", sum(outstate_totdflux)/real(size(outstate_totdflux))
        WRITE(*,*) ""
    END IF
    IF ( ALL( outstate_fnetc == fnetc ) ) THEN
        WRITE(*,*) "All elements of fnetc are IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_fnetc
        !WRITE(*,*) "KERNEL: ", fnetc
        IF ( ALL( outstate_fnetc == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        lstatus = .FALSE.
        WRITE(*,*) "fnetc is NOT IDENTICAL."
        WRITE(*,*) count( outstate_fnetc /= fnetc), " of ", size( fnetc ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_fnetc - fnetc)**2)/real(size(outstate_fnetc)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_fnetc - fnetc))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_fnetc - fnetc))
        WRITE(*,*) "Mean value of kernel-generated outstate_fnetc is ", sum(fnetc)/real(size(fnetc))
        WRITE(*,*) "Mean value of original outstate_fnetc is ", sum(outstate_fnetc)/real(size(outstate_fnetc))
        WRITE(*,*) ""
    END IF
    IF ( ALL( outstate_htr == htr ) ) THEN
        WRITE(*,*) "All elements of htr are IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_htr
        !WRITE(*,*) "KERNEL: ", htr
        IF ( ALL( outstate_htr == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        lstatus = .FALSE.
        WRITE(*,*) "htr is NOT IDENTICAL."
        WRITE(*,*) count( outstate_htr /= htr), " of ", size( htr ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_htr - htr)**2)/real(size(outstate_htr)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_htr - htr))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_htr - htr))
        WRITE(*,*) "Mean value of kernel-generated outstate_htr is ", sum(htr)/real(size(htr))
        WRITE(*,*) "Mean value of original outstate_htr is ", sum(outstate_htr)/real(size(outstate_htr))
        WRITE(*,*) ""
    END IF
    IF ( ALL( outstate_totdclfl == totdclfl ) ) THEN
        WRITE(*,*) "All elements of totdclfl are IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_totdclfl
        !WRITE(*,*) "KERNEL: ", totdclfl
        IF ( ALL( outstate_totdclfl == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        lstatus = .FALSE.
        WRITE(*,*) "totdclfl is NOT IDENTICAL."
        WRITE(*,*) count( outstate_totdclfl /= totdclfl), " of ", size( totdclfl ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_totdclfl - totdclfl)**2)/real(size(outstate_totdclfl)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_totdclfl - totdclfl))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_totdclfl - totdclfl))
        WRITE(*,*) "Mean value of kernel-generated outstate_totdclfl is ", sum(totdclfl)/real(size(totdclfl))
        WRITE(*,*) "Mean value of original outstate_totdclfl is ", sum(outstate_totdclfl)/real(size(outstate_totdclfl))
        WRITE(*,*) ""
    END IF
    IF ( ALL( outstate_totdfluxs == totdfluxs ) ) THEN
        WRITE(*,*) "All elements of totdfluxs are IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_totdfluxs
        !WRITE(*,*) "KERNEL: ", totdfluxs
        IF ( ALL( outstate_totdfluxs == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        lstatus = .FALSE.
        WRITE(*,*) "totdfluxs is NOT IDENTICAL."
        WRITE(*,*) count( outstate_totdfluxs /= totdfluxs), " of ", size( totdfluxs ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_totdfluxs - totdfluxs)**2)/real(size(outstate_totdfluxs)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_totdfluxs - totdfluxs))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_totdfluxs - totdfluxs))
        WRITE(*,*) "Mean value of kernel-generated outstate_totdfluxs is ", sum(totdfluxs)/real(size(totdfluxs))
        WRITE(*,*) "Mean value of original outstate_totdfluxs is ", sum(outstate_totdfluxs)/real(size(outstate_totdfluxs))
        WRITE(*,*) ""
    END IF
    IF ( ALL( outstate_totufluxs == totufluxs ) ) THEN
        WRITE(*,*) "All elements of totufluxs are IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_totufluxs
        !WRITE(*,*) "KERNEL: ", totufluxs
        IF ( ALL( outstate_totufluxs == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        lstatus = .FALSE.
        WRITE(*,*) "totufluxs is NOT IDENTICAL."
        WRITE(*,*) count( outstate_totufluxs /= totufluxs), " of ", size( totufluxs ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_totufluxs - totufluxs)**2)/real(size(outstate_totufluxs)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_totufluxs - totufluxs))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_totufluxs - totufluxs))
        WRITE(*,*) "Mean value of kernel-generated outstate_totufluxs is ", sum(totufluxs)/real(size(totufluxs))
        WRITE(*,*) "Mean value of original outstate_totufluxs is ", sum(outstate_totufluxs)/real(size(outstate_totufluxs))
        WRITE(*,*) ""
    END IF
    IF ( ALL( outstate_totuclfl == totuclfl ) ) THEN
        WRITE(*,*) "All elements of totuclfl are IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_totuclfl
        !WRITE(*,*) "KERNEL: ", totuclfl
        IF ( ALL( outstate_totuclfl == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        lstatus = .FALSE.
        WRITE(*,*) "totuclfl is NOT IDENTICAL."
        WRITE(*,*) count( outstate_totuclfl /= totuclfl), " of ", size( totuclfl ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_totuclfl - totuclfl)**2)/real(size(outstate_totuclfl)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_totuclfl - totuclfl))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_totuclfl - totuclfl))
        WRITE(*,*) "Mean value of kernel-generated outstate_totuclfl is ", sum(totuclfl)/real(size(totuclfl))
        WRITE(*,*) "Mean value of original outstate_totuclfl is ", sum(outstate_totuclfl)/real(size(outstate_totuclfl))
        WRITE(*,*) ""
    END IF
    IF ( ALL( outstate_htrc == htrc ) ) THEN
        WRITE(*,*) "All elements of htrc are IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_htrc
        !WRITE(*,*) "KERNEL: ", htrc
        IF ( ALL( outstate_htrc == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        lstatus = .FALSE.
        WRITE(*,*) "htrc is NOT IDENTICAL."
        WRITE(*,*) count( outstate_htrc /= htrc), " of ", size( htrc ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_htrc - htrc)**2)/real(size(outstate_htrc)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_htrc - htrc))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_htrc - htrc))
        WRITE(*,*) "Mean value of kernel-generated outstate_htrc is ", sum(htrc)/real(size(htrc))
        WRITE(*,*) "Mean value of original outstate_htrc is ", sum(outstate_htrc)/real(size(outstate_htrc))
        WRITE(*,*) ""
    END IF
    IF ( ALL( outstate_fnet == fnet ) ) THEN
        WRITE(*,*) "All elements of fnet are IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_fnet
        !WRITE(*,*) "KERNEL: ", fnet
        IF ( ALL( outstate_fnet == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        lstatus = .FALSE.
        WRITE(*,*) "fnet is NOT IDENTICAL."
        WRITE(*,*) count( outstate_fnet /= fnet), " of ", size( fnet ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_fnet - fnet)**2)/real(size(outstate_fnet)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_fnet - fnet))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_fnet - fnet))
        WRITE(*,*) "Mean value of kernel-generated outstate_fnet is ", sum(fnet)/real(size(fnet))
        WRITE(*,*) "Mean value of original outstate_fnet is ", sum(outstate_fnet)/real(size(outstate_fnet))
        WRITE(*,*) ""
    END IF
    IF ( ALL( outstate_totuflux == totuflux ) ) THEN
        WRITE(*,*) "All elements of totuflux are IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_totuflux
        !WRITE(*,*) "KERNEL: ", totuflux
        IF ( ALL( outstate_totuflux == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        lstatus = .FALSE.
        WRITE(*,*) "totuflux is NOT IDENTICAL."
        WRITE(*,*) count( outstate_totuflux /= totuflux), " of ", size( totuflux ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_totuflux - totuflux)**2)/real(size(outstate_totuflux)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_totuflux - totuflux))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_totuflux - totuflux))
        WRITE(*,*) "Mean value of kernel-generated outstate_totuflux is ", sum(totuflux)/real(size(totuflux))
        WRITE(*,*) "Mean value of original outstate_totuflux is ", sum(outstate_totuflux)/real(size(outstate_totuflux))
        WRITE(*,*) ""
    END IF

    IF ( lstatus ) THEN
        WRITE(*,*) "PASSED"
    ELSE
        WRITE(*,*) "FAILED"
    END IF

    ! DEALLOCATE INSTATE

    ! DEALLOCATE OUTSTATE
    ! DEALLOCATE CALLEE INSTATE

    ! DEALLOCATE INSTATE
    ! DEALLOCATE CALEE OUTSTATE

    ! DEALLOCATE OUTSTATE

    CONTAINS


    ! KERNEL SUBPROGRAM
    subroutine rtrnmc(nlayers, istart, iend, iout, pz, semiss, ncbands,&
                         cldfmc, taucmc, planklay, planklev, plankbnd,&
                         pwvcm, fracs, taut,&
                         totuflux, totdflux, fnet, htr,&
                         totuclfl, totdclfl, fnetc, htrc, totufluxs, totdfluxs )
        integer, intent(in) :: nlayers
        integer, intent(in) :: istart
        integer, intent(in) :: iend
        integer, intent(in) :: iout
        real(kind=r8), intent(in) :: pz(0:)
        real(kind=r8), intent(in) :: pwvcm
        real(kind=r8), intent(in) :: semiss(:)
        real(kind=r8), intent(in) :: planklay(:,:)
        real(kind=r8), intent(in) :: planklev(0:,:)
        real(kind=r8), intent(in) :: plankbnd(:)
        real(kind=r8), intent(in) :: fracs(:,:)
        real(kind=r8), intent(in) :: taut(:,:)
        integer, intent(in) :: ncbands
        real(kind=r8), intent(in) :: cldfmc(:,:)
        real(kind=r8), intent(in) :: taucmc(:,:)
        real(kind=r8), intent(out) :: totuflux(0:)
        real(kind=r8), intent(out) :: totdflux(0:)
        real(kind=r8), intent(out) :: fnet(0:)
        real(kind=r8), intent(out) :: htr(0:)
        real(kind=r8), intent(out) :: totuclfl(0:)
        real(kind=r8), intent(out) :: totdclfl(0:)
        real(kind=r8), intent(out) :: fnetc(0:)
        real(kind=r8), intent(out) :: htrc(0:)
        real(kind=r8), intent(out) :: totufluxs(:,0:)
        real(kind=r8), intent(out) :: totdfluxs(:,0:)
        real(kind=r8) :: abscld(nlayers,ngptlw)
        real(kind=r8) :: atot(nlayers)
        real(kind=r8) :: atrans(nlayers)
        real(kind=r8) :: bbugas(nlayers)
        real(kind=r8) :: bbutot(nlayers)
        real(kind=r8) :: clrurad(0:nlayers)
        real(kind=r8) :: clrdrad(0:nlayers)
        real(kind=r8) :: efclfrac(nlayers,ngptlw)
        real(kind=r8) :: uflux(0:nlayers)
        real(kind=r8) :: dflux(0:nlayers)
        real(kind=r8) :: urad(0:nlayers)
        real(kind=r8) :: drad(0:nlayers)
        real(kind=r8) :: uclfl(0:nlayers)
        real(kind=r8) :: dclfl(0:nlayers)
        real(kind=r8) :: odcld(nlayers,ngptlw)
        real(kind=r8) :: secdiff(nbndlw)
        real(kind=r8) :: a0(nbndlw),a1(nbndlw),a2(nbndlw)
        real(kind=r8) :: wtdiff, rec_6
        real(kind=r8) :: transcld, radld, radclrd, plfrac, blay, dplankup, dplankdn
        real(kind=r8) :: odepth, odtot, odepth_rec, odtot_rec, gassrc
        real(kind=r8) :: tblind, tfactot, bbd, bbdtot, tfacgas, transc, tausfac
        real(kind=r8) :: rad0, reflect, radlu, radclru
        integer :: icldlyr(nlayers)
        integer :: ibnd, ib, iband, lay, lev, l, ig
        integer :: igc
        integer :: iclddn
        integer :: ittot, itgas, itr
        data wtdiff /0.5_r8/
        data rec_6 /0.166667_r8/
        data a0 / 1.66_r8,  1.55_r8,  1.58_r8,  1.66_r8,                 1.54_r8, 1.454_r8,  1.89_r8,  1.33_r8,                1.668_r8,  1.66_r8,  1.66_r8,  1.66_r8,                 1.66_r8,  1.66_r8,  1.66_r8,  1.66_r8 /
        data a1 / 0.00_r8,  0.25_r8,  0.22_r8,  0.00_r8,                 0.13_r8, 0.446_r8, -0.10_r8,  0.40_r8,               -0.006_r8,  0.00_r8,  0.00_r8,  0.00_r8,                 0.00_r8,  0.00_r8,  0.00_r8,  0.00_r8 /
        data a2 / 0.00_r8, -12.0_r8, -11.7_r8,  0.00_r8,                -0.72_r8,-0.243_r8,  0.19_r8,-0.062_r8,                0.414_r8,  0.00_r8,  0.00_r8,  0.00_r8,                 0.00_r8,  0.00_r8,  0.00_r8,  0.00_r8 /
        hvrrtc = '$Revision$'
        do ibnd = 1,nbndlw
            if (ibnd.eq.1 .or. ibnd.eq.4 .or. ibnd.ge.10) then
                secdiff(ibnd) = 1.66_r8
                else
                secdiff(ibnd) = a0(ibnd) + a1(ibnd)*exp(a2(ibnd)*pwvcm)
            endif
        enddo
        if (pwvcm.lt.1.0) secdiff(6) = 1.80_r8
        if (pwvcm.gt.7.1) secdiff(7) = 1.50_r8
        urad(0) = 0.0_r8
        drad(0) = 0.0_r8
        totuflux(0) = 0.0_r8
        totdflux(0) = 0.0_r8
        clrurad(0) = 0.0_r8
        clrdrad(0) = 0.0_r8
        totuclfl(0) = 0.0_r8
        totdclfl(0) = 0.0_r8
        do lay = 1, nlayers
            urad(lay) = 0.0_r8
            drad(lay) = 0.0_r8
            totuflux(lay) = 0.0_r8
            totdflux(lay) = 0.0_r8
            clrurad(lay) = 0.0_r8
            clrdrad(lay) = 0.0_r8
            totuclfl(lay) = 0.0_r8
            totdclfl(lay) = 0.0_r8
            icldlyr(lay) = 0
            do ig = 1, ngptlw
                if (cldfmc(ig,lay) .eq. 1._r8) then
                    ib = ngb(ig)
                    odcld(lay,ig) = secdiff(ib) * taucmc(ig,lay)
                    transcld = exp(-odcld(lay,ig))
                    abscld(lay,ig) = 1._r8 - transcld
                    efclfrac(lay,ig) = abscld(lay,ig) * cldfmc(ig,lay)
                    icldlyr(lay) = 1
                    else
                    odcld(lay,ig) = 0.0_r8
                    abscld(lay,ig) = 0.0_r8
                    efclfrac(lay,ig) = 0.0_r8
                endif
            enddo
        enddo
        igc = 1
        do iband = istart, iend
            if (iout.gt.0.and.iband.ge.2) igc = ngs(iband-1)+1
            1000 continue
            radld = 0._r8
            radclrd = 0._r8
            iclddn = 0
            do lev = nlayers, 1, -1
                plfrac = fracs(lev,igc)
                blay = planklay(lev,iband)
                dplankup = planklev(lev,iband) - blay
                dplankdn = planklev(lev-1,iband) - blay
                odepth = secdiff(iband) * taut(lev,igc)
                if (odepth .lt. 0.0_r8) odepth = 0.0_r8
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
                        radld = radld - radld * (atrans(lev) +                          efclfrac(lev,igc) * (1. - atrans(lev))) +                          gassrc + cldfmc(igc,lev) *                          (bbdtot * atot(lev) - gassrc)
                        drad(lev-1) = drad(lev-1) + radld
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
                        radld = radld - radld * (atrans(lev) +                          efclfrac(lev,igc) * (1._r8 - atrans(lev))) +                          gassrc + cldfmc(igc,lev) *                          (bbdtot * atot(lev) - gassrc)
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
                        radld = radld - radld * (atrans(lev) +                     efclfrac(lev,igc) * (1._r8 - atrans(lev))) +                     gassrc + cldfmc(igc,lev) *                     (bbdtot * atot(lev) - gassrc)
                        drad(lev-1) = drad(lev-1) + radld
                        bbugas(lev) = plfrac * (blay + tfacgas * dplankup)
                        bbutot(lev) = plfrac * (blay + tfactot * dplankup)
                    endif
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
                if (iclddn.eq.1) then
                    radclrd = radclrd + (bbd-radclrd) * atrans(lev)
                    clrdrad(lev-1) = clrdrad(lev-1) + radclrd
                    else
                    radclrd = radld
                    clrdrad(lev-1) = drad(lev-1)
                endif
            enddo
            rad0 = fracs(1,igc) * plankbnd(iband)
            reflect = 1._r8 - semiss(iband)
            radlu = rad0 + reflect * radld
            radclru = rad0 + reflect * radclrd
            urad(0) = urad(0) + radlu
            clrurad(0) = clrurad(0) + radclru
            do lev = 1, nlayers
                if (icldlyr(lev) .eq. 1) then
                    gassrc = bbugas(lev) * atrans(lev)
                    radlu = radlu - radlu * (atrans(lev) +                    efclfrac(lev,igc) * (1._r8 - atrans(lev))) +                    gassrc + cldfmc(igc,lev) *                    (bbutot(lev) * atot(lev) - gassrc)
                    urad(lev) = urad(lev) + radlu
                    else
                    radlu = radlu + (bbugas(lev)-radlu)*atrans(lev)
                    urad(lev) = urad(lev) + radlu
                endif
                if (iclddn.eq.1) then
                    radclru = radclru + (bbugas(lev)-radclru)*atrans(lev)
                    clrurad(lev) = clrurad(lev) + radclru
                    else
                    radclru = radlu
                    clrurad(lev) = urad(lev)
                endif
            enddo
            igc = igc + 1
            if (igc .le. ngs(iband)) go to 1000
            do lev = nlayers, 0, -1
                uflux(lev) = urad(lev)*wtdiff
                dflux(lev) = drad(lev)*wtdiff
                urad(lev) = 0.0_r8
                drad(lev) = 0.0_r8
                totuflux(lev) = totuflux(lev) + uflux(lev) * delwave(iband)
                totdflux(lev) = totdflux(lev) + dflux(lev) * delwave(iband)
                uclfl(lev) = clrurad(lev)*wtdiff
                dclfl(lev) = clrdrad(lev)*wtdiff
                clrurad(lev) = 0.0_r8
                clrdrad(lev) = 0.0_r8
                totuclfl(lev) = totuclfl(lev) + uclfl(lev) * delwave(iband)
                totdclfl(lev) = totdclfl(lev) + dclfl(lev) * delwave(iband)
                totufluxs(iband,lev) = uflux(lev) * delwave(iband)
                totdfluxs(iband,lev) = dflux(lev) * delwave(iband)
            enddo
        enddo
        totuflux(0) = totuflux(0) * fluxfac
        totdflux(0) = totdflux(0) * fluxfac
        totufluxs(:,0) = totufluxs(:,0) * fluxfac
        totdfluxs(:,0) = totdfluxs(:,0) * fluxfac
        fnet(0) = totuflux(0) - totdflux(0)
        totuclfl(0) = totuclfl(0) * fluxfac
        totdclfl(0) = totdclfl(0) * fluxfac
        fnetc(0) = totuclfl(0) - totdclfl(0)
        do lev = 1, nlayers
            totuflux(lev) = totuflux(lev) * fluxfac
            totdflux(lev) = totdflux(lev) * fluxfac
            totufluxs(:,lev) = totufluxs(:,lev) * fluxfac
            totdfluxs(:,lev) = totdfluxs(:,lev) * fluxfac
            fnet(lev) = totuflux(lev) - totdflux(lev)
            totuclfl(lev) = totuclfl(lev) * fluxfac
            totdclfl(lev) = totdclfl(lev) * fluxfac
            fnetc(lev) = totuclfl(lev) - totdclfl(lev)
            l = lev - 1
            htr(l)=heatfac*(fnet(l)-fnet(lev))/(pz(l)-pz(lev))
            htrc(l)=heatfac*(fnetc(l)-fnetc(lev))/(pz(l)-pz(lev))
        enddo
        htr(nlayers) = 0.0_r8
        htrc(nlayers) = 0.0_r8
    end subroutine rtrnmc

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
