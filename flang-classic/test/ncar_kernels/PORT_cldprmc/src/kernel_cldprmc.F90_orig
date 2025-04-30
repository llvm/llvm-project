    MODULE resolvers

    ! RESOLVER SPECS
    INTEGER, PARAMETER :: r8 = selected_real_kind(12)
    INTEGER, PARAMETER :: ngptlw = 140

    END MODULE

    PROGRAM kernel_cldprmc
    USE resolvers

    IMPLICIT NONE


    INTEGER :: kgen_mpi_rank
    CHARACTER(LEN=16) ::kgen_mpi_rank_conv
    INTEGER, DIMENSION(2), PARAMETER :: kgen_mpi_rank_at = (/ 1,2 /)
    INTEGER :: kgen_ierr, kgen_unit, kgen_get_newunit
    INTEGER :: kgen_repeat_counter
    INTEGER :: kgen_counter
    CHARACTER(LEN=16) :: kgen_counter_conv
    INTEGER, DIMENSION(2), PARAMETER :: kgen_counter_at = (/ 10,20 /)
    CHARACTER(LEN=1024) :: kgen_filepath
    INTEGER, DIMENSION(2,10) :: kgen_bound

    ! DRIVER SPECS
    INTEGER :: nlay

    DO kgen_repeat_counter = 1, 4
        kgen_counter = kgen_counter_at(mod(kgen_repeat_counter, 2)+1)
        WRITE( kgen_counter_conv, * ) kgen_counter
        kgen_mpi_rank = kgen_mpi_rank_at(mod(kgen_repeat_counter, 2)+1)
        WRITE( kgen_mpi_rank_conv, * ) kgen_mpi_rank

        kgen_filepath = "../data/cldprmc." // trim(adjustl(kgen_counter_conv)) // "." // trim(adjustl(kgen_mpi_rank_conv))
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
    END PROGRAM kernel_cldprmc

    ! KERNEL DRIVER SUBROUTINE
    SUBROUTINE kernel_driver(nlay, kgen_unit)
    USE resolvers

    IMPLICIT NONE
    INTEGER, INTENT(IN) :: kgen_unit
    INTEGER, DIMENSION(2,10) :: kgen_bound

    ! STATE SPECS
    REAL(KIND = r8), DIMENSION(2) :: absice0
    REAL(KIND = r8), DIMENSION(2, 5) :: absice1
    CHARACTER*18 :: hvrclc
    REAL(KIND = r8), DIMENSION(46, 16) :: absice3
    INTEGER :: iceflag
    REAL(KIND = r8) :: absliq0
    INTEGER :: ngb(ngptlw)
    INTEGER :: ncbands
    REAL(KIND = r8) :: clwpmc(ngptlw, nlay)
    REAL(KIND = r8), DIMENSION(43, 16) :: absice2
    REAL(KIND = r8) :: taucmc(ngptlw, nlay)
    REAL(KIND = r8) :: relqmc(nlay)
    INTEGER :: liqflag
    REAL(KIND = r8) :: dgesmc(nlay)
    REAL(KIND = r8) :: reicmc(nlay)
    REAL(KIND = r8) :: ciwpmc(ngptlw, nlay)
    INTEGER, INTENT(IN) :: nlay
    REAL(KIND = r8), DIMENSION(58, 16) :: absliq1
    INTEGER :: inflag
    REAL(KIND = r8) :: cldfmc(ngptlw, nlay)
    INTEGER :: outstate_ncbands
    REAL(KIND = r8) :: outstate_taucmc(ngptlw, nlay)
    ! READ CALLER INSTATE

    READ(UNIT = kgen_unit) iceflag
    READ(UNIT = kgen_unit) clwpmc
    READ(UNIT = kgen_unit) taucmc
    READ(UNIT = kgen_unit) relqmc
    READ(UNIT = kgen_unit) liqflag
    READ(UNIT = kgen_unit) dgesmc
    READ(UNIT = kgen_unit) reicmc
    READ(UNIT = kgen_unit) ciwpmc
    READ(UNIT = kgen_unit) inflag
    READ(UNIT = kgen_unit) cldfmc
    ! READ CALLEE INSTATE

    READ(UNIT = kgen_unit) absice0
    READ(UNIT = kgen_unit) absice1
    READ(UNIT = kgen_unit) hvrclc
    READ(UNIT = kgen_unit) absice3
    READ(UNIT = kgen_unit) absliq0
    READ(UNIT = kgen_unit) ngb
    READ(UNIT = kgen_unit) absice2
    READ(UNIT = kgen_unit) absliq1
    ! READ CALLEE OUTSTATE

    ! READ CALLER OUTSTATE

    READ(UNIT = kgen_unit) outstate_ncbands
    READ(UNIT = kgen_unit) outstate_taucmc

    ! KERNEL RUN
    CALL cldprmc(nlay, inflag, iceflag, liqflag, cldfmc, ciwpmc, clwpmc, reicmc, dgesmc, relqmc, ncbands, taucmc)

    ! STATE VERIFICATION
    IF ( outstate_ncbands == ncbands ) THEN
        WRITE(*,*) "ncbands is IDENTICAL( ", outstate_ncbands, " )."
    ELSE
        WRITE(*,*) "ncbands is NOT IDENTICAL."
        WRITE(*,*) "STATE : ", outstate_ncbands
        WRITE(*,*) "KERNEL: ", ncbands
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
    subroutine cldprmc(nlayers, inflag, iceflag, liqflag, cldfmc,                          ciwpmc, clwpmc, reicmc, dgesmc, relqmc, ncbands, taucmc)
        integer, intent(in) :: nlayers
        integer, intent(in) :: inflag
        integer, intent(in) :: iceflag
        integer, intent(in) :: liqflag
        real(kind=r8), intent(in) :: cldfmc(:,:)
        real(kind=r8), intent(in) :: ciwpmc(:,:)
        real(kind=r8), intent(in) :: clwpmc(:,:)
        real(kind=r8), intent(in) :: relqmc(:)
        real(kind=r8), intent(in) :: reicmc(:)
        real(kind=r8), intent(in) :: dgesmc(:)
        integer, intent(out) :: ncbands
        real(kind=r8), intent(inout) :: taucmc(:,:)
        integer :: lay
        integer :: ib
        integer :: ig
        integer :: index
        real(kind=r8) :: abscoice(ngptlw)
        real(kind=r8) :: abscoliq(ngptlw)
        real(kind=r8) :: cwp
        real(kind=r8) :: radice
        real(kind=r8) :: dgeice
        real(kind=r8) :: factor
        real(kind=r8) :: fint
        real(kind=r8) :: radliq
        real(kind=r8), parameter :: eps = 1.e-6_r8
        real(kind=r8), parameter :: cldmin = 1.e-80_r8
        hvrclc = '$Revision$'
        ncbands = 1
        do lay = 1, nlayers
            do ig = 1, ngptlw
                cwp = ciwpmc(ig,lay) + clwpmc(ig,lay)
                if (cldfmc(ig,lay) .ge. cldmin .and.              (cwp .ge. cldmin .or. taucmc(ig,lay) .ge. cldmin)) then
                    if (inflag .eq. 0) then
                        return
                        elseif(inflag .eq. 1) then
                        stop 'INFLAG = 1 OPTION NOT AVAILABLE WITH MCICA'
                        elseif(inflag .eq. 2) then
                        radice = reicmc(lay)
                        if (ciwpmc(ig,lay) .eq. 0.0_r8) then
                            abscoice(ig) = 0.0_r8
                            elseif (iceflag .eq. 0) then
                            if (radice .lt. 10.0_r8) stop 'ICE RADIUS TOO SMALL'
                            abscoice(ig) = absice0(1) + absice0(2)/radice
                            elseif (iceflag .eq. 1) then
                            ncbands = 5
                            ib = ngb(ig)
                            abscoice(ig) = absice1(1,ib) + absice1(2,ib)/radice
                            elseif (iceflag .eq. 2) then
                            if (radice .lt. 5.0_r8) stop 'ICE RADIUS OUT OF BOUNDS'
                            if (radice .ge. 5.0_r8 .and. radice .le. 131._r8) then
                                ncbands = 16
                                factor = (radice - 2._r8)/3._r8
                                index = int(factor)
                                if (index .eq. 43) index = 42
                                fint = factor - float(index)
                                ib = ngb(ig)
                                abscoice(ig) =                          absice2(index,ib) + fint *                          (absice2(index+1,ib) - (absice2(index,ib)))
                                elseif (radice .gt. 131._r8) then
                                abscoice(ig) = absice0(1) + absice0(2)/radice
                            endif
                            elseif (iceflag .eq. 3) then
                            dgeice = dgesmc(lay)
                            if (dgeice .lt. 5.0_r8) stop 'ICE GENERALIZED EFFECTIVE SIZE OUT OF BOUNDS'
                            if (dgeice .ge. 5.0_r8 .and. dgeice .le. 140._r8) then
                                ncbands = 16
                                factor = (dgeice - 2._r8)/3._r8
                                index = int(factor)
                                if (index .eq. 46) index = 45
                                fint = factor - float(index)
                                ib = ngb(ig)
                                abscoice(ig) =                          absice3(index,ib) + fint *                          (absice3(index+1,ib) - (absice3(index,ib)))
                                elseif (dgeice .gt. 140._r8) then
                                abscoice(ig) = absice0(1) + absice0(2)/radice
                            endif
                        endif
                        if (clwpmc(ig,lay) .eq. 0.0_r8) then
                            abscoliq(ig) = 0.0_r8
                            elseif (liqflag .eq. 0) then
                            abscoliq(ig) = absliq0
                            elseif (liqflag .eq. 1) then
                            radliq = relqmc(lay)
                            if (radliq .lt. 1.5_r8 .or. radliq .gt. 60._r8) stop                        'LIQUID EFFECTIVE RADIUS OUT OF BOUNDS'
                            index = radliq - 1.5_r8
                            if (index .eq. 58) index = 57
                            if (index .eq. 0) index = 1
                            fint = radliq - 1.5_r8 - index
                            ib = ngb(ig)
                            abscoliq(ig) =                         absliq1(index,ib) + fint *                         (absliq1(index+1,ib) - (absliq1(index,ib)))
                        endif
                        taucmc(ig,lay) = ciwpmc(ig,lay) * abscoice(ig) +                                 clwpmc(ig,lay) * abscoliq(ig)
                    endif
                endif
            enddo
        enddo
    end subroutine cldprmc

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
