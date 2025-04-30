    MODULE resolvers

    ! RESOLVER SPECS
    INTEGER, PARAMETER :: r8 = selected_real_kind(12)
    INTEGER, PARAMETER :: pcols = 16
    INTEGER, PARAMETER :: ncoef = 5
    INTEGER, PARAMETER :: prefr = 7
    INTEGER, PARAMETER :: prefi = 10

    END MODULE

    PROGRAM kernel_binterp
    USE resolvers
    USE omp_lib
    IMPLICIT NONE


    INTEGER :: kgen_mpi_rank
    CHARACTER(LEN=16) ::kgen_mpi_rank_conv
    INTEGER, DIMENSION(3), PARAMETER :: kgen_mpi_rank_at = (/ 0,1,2 /)
    INTEGER :: kgen_ierr, kgen_unit, kgen_get_newunit
    INTEGER :: kgen_repeat_counter
    INTEGER :: kgen_counter
    CHARACTER(LEN=16) :: kgen_counter_conv
    INTEGER, DIMENSION(1), PARAMETER :: kgen_counter_at = (/ 1 /)
    CHARACTER(LEN=1024) :: kgen_filepath
    INTEGER, DIMENSION(2,10) :: kgen_bound

    ! DRIVER SPECS

    DO kgen_repeat_counter = 1, 1
        kgen_counter = kgen_counter_at(mod(kgen_repeat_counter, 1)+1)
        WRITE( kgen_counter_conv, * ) kgen_counter
        kgen_mpi_rank = kgen_mpi_rank_at(mod(kgen_repeat_counter, 3)+1)
        WRITE( kgen_mpi_rank_conv, * ) kgen_mpi_rank

        kgen_filepath = "../data/binterp." // trim(adjustl(kgen_counter_conv)) // "." // trim(adjustl(kgen_mpi_rank_conv))
        kgen_unit = kgen_get_newunit(kgen_mpi_rank+kgen_counter)
        OPEN (UNIT=kgen_unit, FILE=kgen_filepath, STATUS="OLD", ACCESS="STREAM", FORM="UNFORMATTED", ACTION="READ", IOSTAT=kgen_ierr, CONVERT="BIG_ENDIAN")
        IF ( kgen_ierr /= 0 ) THEN
            CALL kgen_error_stop( "FILE OPEN ERROR: " // trim(adjustl(kgen_filepath)) )
        END IF
        ! READ DRIVER INSTATE


        ! KERNEL DRIVER RUN
        CALL kernel_driver(kgen_unit)
        CLOSE (UNIT=kgen_unit)

    END DO
    END PROGRAM kernel_binterp

    ! KERNEL DRIVER SUBROUTINE
    SUBROUTINE kernel_driver(kgen_unit)
    USE resolvers

    IMPLICIT NONE
    INTEGER, INTENT(IN) :: kgen_unit
    INTEGER, DIMENSION(2,10) :: kgen_bound

    ! STATE SPECS
    INTEGER :: itab(pcols)
    REAL(KIND = r8) :: refr(pcols)
    REAL(KIND = r8) :: cext(pcols, ncoef)
    REAL(KIND = r8) :: utab(pcols)
    REAL(KIND = r8), POINTER :: refitabsw(:, :)
    REAL(KIND = r8), POINTER :: refrtabsw(:, :)
    REAL(KIND = r8) :: ttab(pcols)
    REAL(KIND = r8) :: refi(pcols)
    INTEGER :: ncol
    INTEGER :: jtab(pcols)
    REAL(KIND = r8), POINTER :: extpsw(:, :, :, :)
    INTEGER :: outstate_itab(pcols)
    REAL(KIND = r8) :: outstate_refr(pcols)
    REAL(KIND = r8) :: outstate_cext(pcols, ncoef)
    REAL(KIND = r8) :: outstate_utab(pcols)
    REAL(KIND = r8), POINTER :: outstate_refitabsw(:, :)
    REAL(KIND = r8), POINTER :: outstate_refrtabsw(:, :)
    REAL(KIND = r8) :: outstate_ttab(pcols)
    REAL(KIND = r8) :: outstate_refi(pcols)
    INTEGER :: outstate_ncol
    INTEGER :: outstate_jtab(pcols)
    REAL(KIND = r8), POINTER :: outstate_extpsw(:, :, :, :)

    !JMD manual timer additions
    integer*8 c1,c2,cr,cm
    real*8 dt
    integer :: itmax=10000
    character(len=80), parameter :: kname='[kernel_binterp]'
    integer :: it
    !JMD

    ! READ CALLER INSTATE
    READ(UNIT = kgen_unit) itab
    READ(UNIT = kgen_unit) refr
    READ(UNIT = kgen_unit) cext
    READ(UNIT = kgen_unit) utab
    READ(UNIT = kgen_unit) kgen_bound(1, 1)
    READ(UNIT = kgen_unit) kgen_bound(2, 1)
    READ(UNIT = kgen_unit) kgen_bound(1, 2)
    READ(UNIT = kgen_unit) kgen_bound(2, 2)
    ALLOCATE(refitabsw(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
    READ(UNIT = kgen_unit) refitabsw
    READ(UNIT = kgen_unit) kgen_bound(1, 1)
    READ(UNIT = kgen_unit) kgen_bound(2, 1)
    READ(UNIT = kgen_unit) kgen_bound(1, 2)
    READ(UNIT = kgen_unit) kgen_bound(2, 2)
    ALLOCATE(refrtabsw(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
    READ(UNIT = kgen_unit) refrtabsw
    READ(UNIT = kgen_unit) ttab
    READ(UNIT = kgen_unit) refi
    READ(UNIT = kgen_unit) ncol
    READ(UNIT = kgen_unit) jtab
    READ(UNIT = kgen_unit) kgen_bound(1, 1)
    READ(UNIT = kgen_unit) kgen_bound(2, 1)
    READ(UNIT = kgen_unit) kgen_bound(1, 2)
    READ(UNIT = kgen_unit) kgen_bound(2, 2)
    READ(UNIT = kgen_unit) kgen_bound(1, 3)
    READ(UNIT = kgen_unit) kgen_bound(2, 3)
    READ(UNIT = kgen_unit) kgen_bound(1, 4)
    READ(UNIT = kgen_unit) kgen_bound(2, 4)
    ALLOCATE(extpsw(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1, kgen_bound(2, 3) - kgen_bound(1, 3) + 1, kgen_bound(2, 4) - kgen_bound(1, 4) + 1))
    READ(UNIT = kgen_unit) extpsw
    ! READ CALLEE INSTATE

    ! READ CALLEE OUTSTATE

    ! READ CALLER OUTSTATE

    READ(UNIT = kgen_unit) outstate_itab
    READ(UNIT = kgen_unit) outstate_refr
    READ(UNIT = kgen_unit) outstate_cext
    READ(UNIT = kgen_unit) outstate_utab
    READ(UNIT = kgen_unit) kgen_bound(1, 1)
    READ(UNIT = kgen_unit) kgen_bound(2, 1)
    READ(UNIT = kgen_unit) kgen_bound(1, 2)
    READ(UNIT = kgen_unit) kgen_bound(2, 2)
    ALLOCATE(outstate_refitabsw(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
    READ(UNIT = kgen_unit) outstate_refitabsw
    READ(UNIT = kgen_unit) kgen_bound(1, 1)
    READ(UNIT = kgen_unit) kgen_bound(2, 1)
    READ(UNIT = kgen_unit) kgen_bound(1, 2)
    READ(UNIT = kgen_unit) kgen_bound(2, 2)
    ALLOCATE(outstate_refrtabsw(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
    READ(UNIT = kgen_unit) outstate_refrtabsw
    READ(UNIT = kgen_unit) outstate_ttab
    READ(UNIT = kgen_unit) outstate_refi
    READ(UNIT = kgen_unit) outstate_ncol
    READ(UNIT = kgen_unit) outstate_jtab
    READ(UNIT = kgen_unit) kgen_bound(1, 1)
    READ(UNIT = kgen_unit) kgen_bound(2, 1)
    READ(UNIT = kgen_unit) kgen_bound(1, 2)
    READ(UNIT = kgen_unit) kgen_bound(2, 2)
    READ(UNIT = kgen_unit) kgen_bound(1, 3)
    READ(UNIT = kgen_unit) kgen_bound(2, 3)
    READ(UNIT = kgen_unit) kgen_bound(1, 4)
    READ(UNIT = kgen_unit) kgen_bound(2, 4)
    ALLOCATE(outstate_extpsw(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1, kgen_bound(2, 3) - kgen_bound(1, 3) + 1, kgen_bound(2, 4) - kgen_bound(1, 4) + 1))
    READ(UNIT = kgen_unit) outstate_extpsw

    call system_clock(c1,cr,cm)
    ! KERNEL RUN
    do it=1,itmax
       CALL binterp(extpsw, ncol, ncoef, prefr, prefi, refr, refi, refrtabsw, refitabsw, itab, jtab, ttab, utab, cext)
    enddo
    call system_clock(c2,cr,cm)
    dt = dble(c2-c1)/dble(cr)
    print *, TRIM(kname), ' total time (sec): ',dt
    print *, TRIM(kname), ' time per call (usec): ',1.e6*dt/dble(itmax)


    ! STATE VERIFICATION
    IF ( ALL( outstate_itab == itab ) ) THEN
        WRITE(*,*) "itab is IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_itab
        !WRITE(*,*) "KERNEL: ", itab
        IF ( ALL( outstate_itab == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        WRITE(*,*) "itab is NOT IDENTICAL."
        WRITE(*,*) count( outstate_itab /= itab), " of ", size( itab ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_itab - itab)**2)/real(size(outstate_itab)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_itab - itab))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_itab - itab))
        WRITE(*,*) "Mean value of kernel-generated outstate_itab is ", sum(itab)/real(size(itab))
        WRITE(*,*) "Mean value of original outstate_itab is ", sum(outstate_itab)/real(size(outstate_itab))
        WRITE(*,*) ""
    END IF
    IF ( ALL( outstate_refr == refr ) ) THEN
        WRITE(*,*) "refr is IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_refr
        !WRITE(*,*) "KERNEL: ", refr
        IF ( ALL( outstate_refr == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        WRITE(*,*) "refr is NOT IDENTICAL."
        WRITE(*,*) count( outstate_refr /= refr), " of ", size( refr ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_refr - refr)**2)/real(size(outstate_refr)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_refr - refr))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_refr - refr))
        WRITE(*,*) "Mean value of kernel-generated outstate_refr is ", sum(refr)/real(size(refr))
        WRITE(*,*) "Mean value of original outstate_refr is ", sum(outstate_refr)/real(size(outstate_refr))
        WRITE(*,*) ""
    END IF
    IF ( ALL( outstate_cext == cext ) ) THEN
        WRITE(*,*) "cext is IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_cext
        !WRITE(*,*) "KERNEL: ", cext
        IF ( ALL( outstate_cext == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        WRITE(*,*) "cext is NOT IDENTICAL."
        WRITE(*,*) count( outstate_cext /= cext), " of ", size( cext ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_cext - cext)**2)/real(size(outstate_cext)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_cext - cext))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_cext - cext))
        WRITE(*,*) "Mean value of kernel-generated outstate_cext is ", sum(cext)/real(size(cext))
        WRITE(*,*) "Mean value of original outstate_cext is ", sum(outstate_cext)/real(size(outstate_cext))
        WRITE(*,*) ""
    END IF
    IF ( ALL( outstate_utab == utab ) ) THEN
        WRITE(*,*) "utab is IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_utab
        !WRITE(*,*) "KERNEL: ", utab
        IF ( ALL( outstate_utab == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        WRITE(*,*) "utab is NOT IDENTICAL."
        WRITE(*,*) count( outstate_utab /= utab), " of ", size( utab ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_utab - utab)**2)/real(size(outstate_utab)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_utab - utab))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_utab - utab))
        WRITE(*,*) "Mean value of kernel-generated outstate_utab is ", sum(utab)/real(size(utab))
        WRITE(*,*) "Mean value of original outstate_utab is ", sum(outstate_utab)/real(size(outstate_utab))
        WRITE(*,*) ""
    END IF
    IF ( ALL( outstate_refitabsw == refitabsw ) ) THEN
        WRITE(*,*) "refitabsw is IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_refitabsw
        !WRITE(*,*) "KERNEL: ", refitabsw
        IF ( ALL( outstate_refitabsw == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        WRITE(*,*) "refitabsw is NOT IDENTICAL."
        WRITE(*,*) count( outstate_refitabsw /= refitabsw), " of ", size( refitabsw ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_refitabsw - refitabsw)**2)/real(size(outstate_refitabsw)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_refitabsw - refitabsw))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_refitabsw - refitabsw))
        WRITE(*,*) "Mean value of kernel-generated outstate_refitabsw is ", sum(refitabsw)/real(size(refitabsw))
        WRITE(*,*) "Mean value of original outstate_refitabsw is ", sum(outstate_refitabsw)/real(size(outstate_refitabsw))
        WRITE(*,*) ""
    END IF
    IF ( ALL( outstate_refrtabsw == refrtabsw ) ) THEN
        WRITE(*,*) "refrtabsw is IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_refrtabsw
        !WRITE(*,*) "KERNEL: ", refrtabsw
        IF ( ALL( outstate_refrtabsw == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        WRITE(*,*) "refrtabsw is NOT IDENTICAL."
        WRITE(*,*) count( outstate_refrtabsw /= refrtabsw), " of ", size( refrtabsw ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_refrtabsw - refrtabsw)**2)/real(size(outstate_refrtabsw)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_refrtabsw - refrtabsw))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_refrtabsw - refrtabsw))
        WRITE(*,*) "Mean value of kernel-generated outstate_refrtabsw is ", sum(refrtabsw)/real(size(refrtabsw))
        WRITE(*,*) "Mean value of original outstate_refrtabsw is ", sum(outstate_refrtabsw)/real(size(outstate_refrtabsw))
        WRITE(*,*) ""
    END IF
    IF ( ALL( outstate_ttab == ttab ) ) THEN
        WRITE(*,*) "ttab is IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_ttab
        !WRITE(*,*) "KERNEL: ", ttab
        IF ( ALL( outstate_ttab == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        WRITE(*,*) "ttab is NOT IDENTICAL."
        WRITE(*,*) count( outstate_ttab /= ttab), " of ", size( ttab ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_ttab - ttab)**2)/real(size(outstate_ttab)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_ttab - ttab))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_ttab - ttab))
        WRITE(*,*) "Mean value of kernel-generated outstate_ttab is ", sum(ttab)/real(size(ttab))
        WRITE(*,*) "Mean value of original outstate_ttab is ", sum(outstate_ttab)/real(size(outstate_ttab))
        WRITE(*,*) ""
    END IF
    IF ( ALL( outstate_refi == refi ) ) THEN
        WRITE(*,*) "refi is IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_refi
        !WRITE(*,*) "KERNEL: ", refi
        IF ( ALL( outstate_refi == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        WRITE(*,*) "refi is NOT IDENTICAL."
        WRITE(*,*) count( outstate_refi /= refi), " of ", size( refi ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_refi - refi)**2)/real(size(outstate_refi)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_refi - refi))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_refi - refi))
        WRITE(*,*) "Mean value of kernel-generated outstate_refi is ", sum(refi)/real(size(refi))
        WRITE(*,*) "Mean value of original outstate_refi is ", sum(outstate_refi)/real(size(outstate_refi))
        WRITE(*,*) ""
    END IF
    IF ( outstate_ncol == ncol ) THEN
        WRITE(*,*) "ncol is IDENTICAL."
        WRITE(*,*) "STATE : ", outstate_ncol
        WRITE(*,*) "KERNEL: ", ncol
    ELSE
        WRITE(*,*) "ncol is NOT IDENTICAL."
        WRITE(*,*) "STATE : ", outstate_ncol
        WRITE(*,*) "KERNEL: ", ncol
    END IF
    IF ( ALL( outstate_jtab == jtab ) ) THEN
        WRITE(*,*) "jtab is IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_jtab
        !WRITE(*,*) "KERNEL: ", jtab
        IF ( ALL( outstate_jtab == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        WRITE(*,*) "jtab is NOT IDENTICAL."
        WRITE(*,*) count( outstate_jtab /= jtab), " of ", size( jtab ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_jtab - jtab)**2)/real(size(outstate_jtab)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_jtab - jtab))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_jtab - jtab))
        WRITE(*,*) "Mean value of kernel-generated outstate_jtab is ", sum(jtab)/real(size(jtab))
        WRITE(*,*) "Mean value of original outstate_jtab is ", sum(outstate_jtab)/real(size(outstate_jtab))
        WRITE(*,*) ""
    END IF
    IF ( ALL( outstate_extpsw == extpsw ) ) THEN
        WRITE(*,*) "extpsw is IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_extpsw
        !WRITE(*,*) "KERNEL: ", extpsw
        IF ( ALL( outstate_extpsw == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        WRITE(*,*) "extpsw is NOT IDENTICAL."
        WRITE(*,*) count( outstate_extpsw /= extpsw), " of ", size( extpsw ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_extpsw - extpsw)**2)/real(size(outstate_extpsw)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_extpsw - extpsw))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_extpsw - extpsw))
        WRITE(*,*) "Mean value of kernel-generated outstate_extpsw is ", sum(extpsw)/real(size(extpsw))
        WRITE(*,*) "Mean value of original outstate_extpsw is ", sum(outstate_extpsw)/real(size(outstate_extpsw))
        WRITE(*,*) ""
    END IF

    ! DEALLOCATE INSTATE
    DEALLOCATE(refitabsw)
    DEALLOCATE(refrtabsw)
    DEALLOCATE(extpsw)

    ! DEALLOCATE OUTSTATE
    DEALLOCATE(outstate_refitabsw)
    DEALLOCATE(outstate_refrtabsw)
    DEALLOCATE(outstate_extpsw)
    ! DEALLOCATE CALLEE INSTATE

    ! DEALLOCATE INSTATE
    ! DEALLOCATE CALEE OUTSTATE

    ! DEALLOCATE OUTSTATE

    CONTAINS


    ! KERNEL SUBPROGRAM
    subroutine binterp(table,ncol,km,im,jm,x,y,xtab,ytab,ix,jy,t,u,out)

        !     bilinear interpolation of table
        !
        implicit none
        integer im,jm,km,ncol
        real(r8) table(km,im,jm),xtab(im),ytab(jm),out(pcols,km)
        integer i,ix(pcols),ip1,j,jy(pcols),jp1,k,ic
        real(r8) x(pcols),dx,t(pcols),y(pcols),dy,u(pcols),tu(pcols),tuc(pcols),tcu(pcols),tcuc(pcols)
        real(r8) temp1,temp2,temp3,temp4
!dir$ assume_aligned table:64
!dir$ assume_aligned xtab:64
!dir$ assume_aligned ytab:64
!dir$ assume_aligned out:64
!dir$ assume_aligned ix:64
!dir$ assume_aligned jy:64
!dir$ assume_aligned x:64
!dir$ assume_aligned t:64
!dir$ assume_aligned tu:64
!dir$ assume_aligned y:64
!dir$ assume_aligned u:64
!dir$ assume_aligned tuc:64
!dir$ assume_aligned tcu:64
!dir$ assume_aligned tcuc:64
     	!print *,km
        if(ix(1).gt.0) go to 30
        if(im.gt.1)then
!dir$ SIMD
            do ic=1,ncol
                do i=1,im
                    if(x(ic).lt.xtab(i))go to 10
                enddo
                10 ix(ic)=max0(i-1,1)
                ip1=min(ix(ic)+1,im)
                dx=(xtab(ip1)-xtab(ix(ic)))
                if(abs(dx).gt.1.e-20_r8)then
                    t(ic)=(x(ic)-xtab(ix(ic)))/dx
                else
                    t(ic)=0._r8
                endif
            end do
        else
            ix(:ncol)=1
            t(:ncol)=0._r8
        endif
        if(jm.gt.1)then
!dir$ SIMD
            do ic=1,ncol
                do j=1,jm
                    if(y(ic).lt.ytab(j))go to 20
                enddo
                20 jy(ic)=max0(j-1,1)
                jp1=min(jy(ic)+1,jm)
                dy=(ytab(jp1)-ytab(jy(ic)))
                if(abs(dy).gt.1.e-20_r8)then
                    u(ic)=(y(ic)-ytab(jy(ic)))/dy
                else
                    u(ic)=0._r8
                endif
            end do
        else
            jy(:ncol)=1
            u(:ncol)=0._r8
        endif
        30 continue
!Do not use SIMD here
        do ic=1,ncol
            tu(ic)=t(ic)*u(ic)
            tuc(ic)=t(ic)-tu(ic)
            tcuc(ic)=1._r8-tuc(ic)-u(ic)
            tcu(ic)=u(ic)-tu(ic)
            jp1=min(jy(ic)+1,jm)
            ip1=min(ix(ic)+1,im)
!dir$ SIMD
            do k=1,km
                out(ic,k) = tcuc(ic) * table(k,ix(ic),jy(ic)) + tuc(ic) * table(k,ip1,jy(ic)) + tu(ic) * table(k,ip1,jp1) + tcu(ic) * table(k,ix(ic),jp1)
	    end do
        enddo
        return
    end subroutine binterp

    END SUBROUTINE kernel_driver

    
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
