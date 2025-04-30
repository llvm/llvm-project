#ifdef __aarch64__
#define _TOL 1.E-12
#else
#define _TOL 0.0
#endif
    MODULE resolvers

    ! RESOLVER SPECS
    INTEGER, PARAMETER :: r8 = selected_real_kind(12)
    REAL(KIND = r8), PARAMETER :: tblint = 10000.0
    REAL(KIND = r8), PARAMETER :: od_lo = 0.06
    INTEGER, PARAMETER :: ntbl = 10000

    END MODULE

    PROGRAM kernel_reftra_sw
    USE resolvers

    IMPLICIT NONE


    INTEGER :: kgen_mpi_rank
    CHARACTER(LEN=16) ::kgen_mpi_rank_conv
    INTEGER, DIMENSION(1), PARAMETER :: kgen_mpi_rank_at = (/ 0 /)
    INTEGER :: kgen_ierr, kgen_unit, kgen_get_newunit
    INTEGER :: kgen_repeat_counter
    INTEGER :: kgen_counter
    CHARACTER(LEN=16) :: kgen_counter_conv
    INTEGER, DIMENSION(1), PARAMETER :: kgen_counter_at = (/ 1 /)
    CHARACTER(LEN=1024) :: kgen_filepath
    INTEGER, DIMENSION(2,10) :: kgen_bound

    ! DRIVER SPECS
    REAL(KIND = r8) :: prmu0
    INTEGER :: nlayers

    DO kgen_repeat_counter = 1, 1
        kgen_counter = kgen_counter_at(mod(kgen_repeat_counter, 1)+1)
        WRITE( kgen_counter_conv, * ) kgen_counter
        kgen_mpi_rank = kgen_mpi_rank_at(mod(kgen_repeat_counter, 1)+1)
        WRITE( kgen_mpi_rank_conv, * ) kgen_mpi_rank

        kgen_filepath = "../data/reftra_sw." // trim(adjustl(kgen_counter_conv)) // "." // trim(adjustl(kgen_mpi_rank_conv))
        kgen_unit = kgen_get_newunit(kgen_mpi_rank+kgen_counter)
        OPEN (UNIT=kgen_unit, FILE=kgen_filepath, STATUS="OLD", ACCESS="STREAM", FORM="UNFORMATTED", ACTION="READ", IOSTAT=kgen_ierr, CONVERT="BIG_ENDIAN")
        IF ( kgen_ierr /= 0 ) THEN
            CALL kgen_error_stop( "FILE OPEN ERROR: " // trim(adjustl(kgen_filepath)) )
        END IF
        ! READ DRIVER INSTATE

        READ(UNIT = kgen_unit) prmu0
        READ(UNIT = kgen_unit) nlayers

        ! KERNEL DRIVER RUN
        CALL kernel_driver(prmu0, nlayers, kgen_unit)
        CLOSE (UNIT=kgen_unit)

    END DO
    END PROGRAM kernel_reftra_sw

    ! KERNEL DRIVER SUBROUTINE
    SUBROUTINE kernel_driver(prmu0, nlayers, kgen_unit)
    USE resolvers

    IMPLICIT NONE
    INTEGER, INTENT(IN) :: kgen_unit
    INTEGER, DIMENSION(2,10) :: kgen_bound

    ! STATE SPECS
    REAL(KIND = r8) :: ztradc(nlayers + 1)
    INTEGER :: klev
    REAL(KIND = r8) :: bpade
    REAL(KIND = r8), INTENT(IN) :: prmu0
    CHARACTER*18 :: hvrrft
    REAL(KIND = r8) :: ztauc(nlayers)
    REAL(KIND = r8) :: zomcc(nlayers)
    REAL(KIND = r8), DIMENSION(0 : ntbl) :: exp_tbl
    INTEGER, INTENT(IN) :: nlayers
    REAL(KIND = r8) :: zrefdc(nlayers + 1)
    REAL(KIND = r8) :: ztrac(nlayers + 1)
    REAL(KIND = r8) :: zrefc(nlayers + 1)
    REAL(KIND = r8) :: zgcc(nlayers)
    LOGICAL :: lrtchkclr(nlayers)
    REAL(KIND = r8) :: outstate_ztradc(nlayers + 1)
    REAL(KIND = r8) :: outstate_zrefdc(nlayers + 1)
    REAL(KIND = r8) :: outstate_ztrac(nlayers + 1)
    REAL(KIND = r8) :: outstate_zrefc(nlayers + 1)

    !JMD manual timer additions
    integer*8 c1,c2,cr,cm
    real*8 dt
    integer :: itmax=100000
    character(len=80), parameter :: kname='[kernel_reftra_sw]'
    integer :: it
    !JMD
    LOGICAL :: lstatus = .TRUE.
    
    ! READ CALLER INSTATE

    READ(UNIT = kgen_unit) ztradc
    READ(UNIT = kgen_unit) klev
    READ(UNIT = kgen_unit) ztauc
    READ(UNIT = kgen_unit) zomcc
    READ(UNIT = kgen_unit) zrefdc
    READ(UNIT = kgen_unit) ztrac
    READ(UNIT = kgen_unit) zrefc
    READ(UNIT = kgen_unit) zgcc
    READ(UNIT = kgen_unit) lrtchkclr
    ! READ CALLEE INSTATE

    READ(UNIT = kgen_unit) bpade
    READ(UNIT = kgen_unit) hvrrft
    READ(UNIT = kgen_unit) exp_tbl
    ! READ CALLEE OUTSTATE

    ! READ CALLER OUTSTATE

    READ(UNIT = kgen_unit) outstate_ztradc
    READ(UNIT = kgen_unit) outstate_zrefdc
    READ(UNIT = kgen_unit) outstate_ztrac
    READ(UNIT = kgen_unit) outstate_zrefc

    call system_clock(c1,cr,cm)
    ! KERNEL RUN
    do it=1,itmax
       CALL reftra_sw(klev, lrtchkclr, zgcc, prmu0, ztauc, zomcc, zrefc, zrefdc, ztrac, ztradc)
    enddo
    call system_clock(c2,cr,cm)
    dt = dble(c2-c1)/dble(cr)
    print *, TRIM(kname), ' total time (sec): ',dt
    print *, TRIM(kname), ' time per call (usec): ',1.e6*dt/dble(itmax)
   

    ! STATE VERIFICATION
    IF ( ALL(( abs(outstate_ztradc - ztradc) ) .LE. _TOL) ) THEN
        WRITE(*,*) "ztradc is IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_ztradc
        !WRITE(*,*) "KERNEL: ", ztradc
        IF ( ALL( outstate_ztradc == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        lstatus = .FALSE.
        WRITE(*,*) "ztradc is NOT IDENTICAL."
        WRITE(*,*) count( outstate_ztradc /= ztradc), " of ", size( ztradc ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_ztradc - ztradc)**2)/real(size(outstate_ztradc)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_ztradc - ztradc))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_ztradc - ztradc))
        WRITE(*,*) "Mean value of kernel-generated outstate_ztradc is ", sum(ztradc)/real(size(ztradc))
        WRITE(*,*) "Mean value of original outstate_ztradc is ", sum(outstate_ztradc)/real(size(outstate_ztradc))
        WRITE(*,*) ""
    END IF
    IF ( ALL(( abs(outstate_zrefdc - zrefdc) ) .LE. _TOL)) THEN
        WRITE(*,*) "zrefdc is IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_zrefdc
        !WRITE(*,*) "KERNEL: ", zrefdc
        IF ( ALL( outstate_zrefdc == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        lstatus = .FALSE.
        WRITE(*,*) "zrefdc is NOT IDENTICAL."
        WRITE(*,*) count( outstate_zrefdc /= zrefdc), " of ", size( zrefdc ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_zrefdc - zrefdc)**2)/real(size(outstate_zrefdc)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_zrefdc - zrefdc))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_zrefdc - zrefdc))
        WRITE(*,*) "Mean value of kernel-generated outstate_zrefdc is ", sum(zrefdc)/real(size(zrefdc))
        WRITE(*,*) "Mean value of original outstate_zrefdc is ", sum(outstate_zrefdc)/real(size(outstate_zrefdc))
        WRITE(*,*) ""
    END IF
    IF ( ALL(( abs(outstate_ztrac - ztrac) ) .LE. _TOL)) THEN
        WRITE(*,*) "ztrac is IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_ztrac
        !WRITE(*,*) "KERNEL: ", ztrac
        IF ( ALL( outstate_ztrac == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        lstatus = .FALSE.
        WRITE(*,*) "ztrac is NOT IDENTICAL."
        WRITE(*,*) count( outstate_ztrac /= ztrac), " of ", size( ztrac ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_ztrac - ztrac)**2)/real(size(outstate_ztrac)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_ztrac - ztrac))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_ztrac - ztrac))
        WRITE(*,*) "Mean value of kernel-generated outstate_ztrac is ", sum(ztrac)/real(size(ztrac))
        WRITE(*,*) "Mean value of original outstate_ztrac is ", sum(outstate_ztrac)/real(size(outstate_ztrac))
        WRITE(*,*) ""
    END IF
    IF ( ALL(( abs(outstate_zrefc - zrefc) ) .LE. _TOL) ) THEN
        WRITE(*,*) "zrefc is IDENTICAL."
        !WRITE(*,*) "STATE : ", outstate_zrefc
        !WRITE(*,*) "KERNEL: ", zrefc
        IF ( ALL( outstate_zrefc == 0 ) ) THEN
            WRITE(*,*) "All values are zero."
        END IF
    ELSE
        lstatus = .FALSE.
        WRITE(*,*) "zrefc is NOT IDENTICAL."
        WRITE(*,*) count( outstate_zrefc /= zrefc), " of ", size( zrefc ), " elements are different."
        WRITE(*,*) "RMS of difference is ", sqrt(sum((outstate_zrefc - zrefc)**2)/real(size(outstate_zrefc)))
        WRITE(*,*) "Minimum difference is ", minval(abs(outstate_zrefc - zrefc))
        WRITE(*,*) "Maximum difference is ", maxval(abs(outstate_zrefc - zrefc))
        WRITE(*,*) "Mean value of kernel-generated outstate_zrefc is ", sum(zrefc)/real(size(zrefc))
        WRITE(*,*) "Mean value of original outstate_zrefc is ", sum(outstate_zrefc)/real(size(outstate_zrefc))
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
    subroutine reftra_sw(nlayers, lrtchk, pgg, prmuz, ptau, pw,                            pref, prefd, ptra, ptrad)
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

        integer, intent(in) :: nlayers

        logical, intent(in) :: lrtchk(:)
        ! Logical flag for reflectivity and
        ! and transmissivity calculation;
        !   Dimensions: (nlayers)

        real(kind=r8), intent(in) :: pgg(:)
        ! asymmetry parameter
        !   Dimensions: (nlayers)
        real(kind=r8), intent(in) :: ptau(:)
        ! optical depth
        !   Dimensions: (nlayers)
        real(kind=r8), intent(in) :: pw(:)
        ! single scattering albedo
        !   Dimensions: (nlayers)
        real(kind=r8), intent(in) :: prmuz
        ! cosine of solar zenith angle

        ! ------- Output -------

        real(kind=r8), intent(inout) :: pref(:)
        ! direct beam reflectivity
        !   Dimensions: (nlayers+1)
        real(kind=r8), intent(inout) :: prefd(:)
        ! diffuse beam reflectivity
        !   Dimensions: (nlayers+1)
        real(kind=r8), intent(inout) :: ptra(:)
        ! direct beam transmissivity
        !   Dimensions: (nlayers+1)
        real(kind=r8), intent(inout) :: ptrad(:)
        ! diffuse beam transmissivity
        !   Dimensions: (nlayers+1)

        ! ------- Local -------

        integer :: jk, jl, kmodts
        integer :: itind

        real(kind=r8) :: tblind
        real(kind=r8) :: za, za1, za2
        real(kind=r8) :: zbeta, zdend, zdenr, zdent
        real(kind=r8) :: ze1, ze2, zem1, zem2, zemm, zep1, zep2
        real(kind=r8) :: zg, zg3, zgamma1, zgamma2, zgamma3, zgamma4, zgt
        real(kind=r8) :: zr1, zr2, zr3, zr4, zr5
        real(kind=r8) :: zrk, zrk2, zrkg, zrm1, zrp, zrp1, zrpp
        real(kind=r8) :: zsr3, zt1, zt2, zt3, zt4, zt5, zto1
        real(kind=r8) :: zw, zwcrit, zwo

        real(kind=r8), parameter :: eps = 1.e-08_r8

        !     ------------------------------------------------------------------

        ! Initialize

        hvrrft = '$Revision$'

        zsr3=sqrt(3._r8)
        zwcrit=0.9999995_r8
        kmodts=2

        do jk=1, nlayers
            if (.not.lrtchk(jk)) then
                pref(jk) =0._r8
                ptra(jk) =1._r8
                prefd(jk)=0._r8
                ptrad(jk)=1._r8
                else
                zto1=ptau(jk)
                zw  =pw(jk)
                zg  =pgg(jk)

                ! General two-stream expressions

                zg3= 3._r8 * zg
                if (kmodts == 1) then
                    zgamma1= (7._r8 - zw * (4._r8 + zg3)) * 0.25_r8
                    zgamma2=-(1._r8 - zw * (4._r8 - zg3)) * 0.25_r8
                    zgamma3= (2._r8 - zg3 * prmuz ) * 0.25_r8
                    else if (kmodts == 2) then
                    zgamma1= (8._r8 - zw * (5._r8 + zg3)) * 0.25_r8
                    zgamma2=  3._r8 *(zw * (1._r8 - zg )) * 0.25_r8
                    zgamma3= (2._r8 - zg3 * prmuz ) * 0.25_r8
                    else if (kmodts == 3) then
                    zgamma1= zsr3 * (2._r8 - zw * (1._r8 + zg)) * 0.5_r8
                    zgamma2= zsr3 * zw * (1._r8 - zg ) * 0.5_r8
                    zgamma3= (1._r8 - zsr3 * zg * prmuz ) * 0.5_r8
                end if
                zgamma4= 1._r8 - zgamma3

                ! Recompute original s.s.a. to test for conservative solution

                zwo= zw / (1._r8 - (1._r8 - zw) * (zg / (1._r8 - zg))**2)

                if (zwo >= zwcrit) then
                    ! Conservative scattering

                    za  = zgamma1 * prmuz
                    za1 = za - zgamma3
                    zgt = zgamma1 * zto1

                    ! Homogeneous reflectance and transmittance,
                    ! collimated beam

                    ze1 = min ( zto1 / prmuz , 500._r8)
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

                    pref(jk) = (zgt - za1 * (1._r8 - ze2)) / (1._r8 + zgt)
                    ptra(jk) = 1._r8 - pref(jk)

                    ! isotropic incidence

                    prefd(jk) = zgt / (1._r8 + zgt)
                    ptrad(jk) = 1._r8 - prefd(jk)

                    ! This is applied for consistency between total (delta-scaled) and direct (unscaled)
                    ! calculations at very low optical depths (tau < 1.e-4) when the exponential lookup
                    ! table returns a transmittance of 1.0.
                    if (ze2 .eq. 1.0_r8) then
                        pref(jk) = 0.0_r8
                        ptra(jk) = 1.0_r8
                        prefd(jk) = 0.0_r8
                        ptrad(jk) = 1.0_r8
                    endif

                    else
                    ! Non-conservative scattering

                    za1 = zgamma1 * zgamma4 + zgamma2 * zgamma3
                    za2 = zgamma1 * zgamma3 + zgamma2 * zgamma4
                    zrk = sqrt ( zgamma1**2 - zgamma2**2)
                    zrp = zrk * prmuz
                    zrp1 = 1._r8 + zrp
                    zrm1 = 1._r8 - zrp
                    zrk2 = 2._r8 * zrk
                    zrpp = 1._r8 - zrp*zrp
                    zrkg = zrk + zgamma1
                    zr1  = zrm1 * (za2 + zrk * zgamma3)
                    zr2  = zrp1 * (za2 - zrk * zgamma3)
                    zr3  = zrk2 * (zgamma3 - za2 * prmuz )
                    zr4  = zrpp * zrkg
                    zr5  = zrpp * (zrk - zgamma1)
                    zt1  = zrp1 * (za1 + zrk * zgamma4)
                    zt2  = zrm1 * (za1 - zrk * zgamma4)
                    zt3  = zrk2 * (zgamma4 + za1 * prmuz )
                    zt4  = zr4
                    zt5  = zr5
                    zbeta = (zgamma1 - zrk) / zrkg
                    !- zr5 / zr4

                    ! Homogeneous reflectance and transmittance

                    ze1 = min ( zrk * zto1, 500._r8)
                    ze2 = min ( zto1 / prmuz , 500._r8)
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
                    zdent = zt4*zep1 + zt5*zem1
                    if (zdenr .ge. -eps .and. zdenr .le. eps) then
                        pref(jk) = eps
                        ptra(jk) = zem2
                        else
                        pref(jk) = zw * (zr1*zep1 - zr2*zem1 - zr3*zem2) / zdenr
                        ptra(jk) = zem2 - zem2 * zw * (zt1*zep1 - zt2*zem1 - zt3*zep2) / zdent
                    endif

                    ! diffuse beam

                    zemm = zem1*zem1
                    zdend = 1._r8 / ( (1._r8 - zbeta*zemm ) * zrkg)
                    prefd(jk) =  zgamma2 * (1._r8 - zemm) * zdend
                    ptrad(jk) =  zrk2*zem1*zdend

                endif

            endif

        enddo

    end subroutine reftra_sw

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
