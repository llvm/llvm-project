! ======================================================================================================
! This kernel represents a distillation of *part* of
! the taumol03 calculation in the gas optics part of the PSRAD
! atmospheric
! radiation code.
!
! It is meant to show conceptually how one might "SIMD-ize" swaths of
! the taumol03 code related to calculating the
! taug term, so that the impact of the conditional expression on
! specparm could be reduced and at least partial vectorization
! across columns could be achieved.
!
! I consider it at this point to be "compiling pseudo-code".
!
! By this I mean that the code as written compiles under ifort, but has
! not been tested
! for correctness, nor I have written a driver routine for it. It does
! not contain everything
! that is going on in the taug parent taumol03 code, but I don't claim
! to actually completely
! understand the physical meaning of all or even most of the inputs
! required to make it run.
!
! It has been written to vectorize, but apparently does not actually do
! that
! under the ifort V13 compiler with the -xHost -O3 level of
! optimization, even with !dir$ assume_aligned directives.
! I hypothesize that the compiler is baulking to do so for the indirect
! addressed calls into the absa
! look-up table, either that or 4 byte integers may be causing alignment
! issues relative to 8 byte reals. Anyway,
! it seems to complain about the key loop being too complex.
! ======================================================================================================
MODULE mo_taumol03
    USE mo_kind, only:wp
    USE mo_lrtm_setup, ONLY: nspa
    USE mo_lrtm_setup, ONLY: nspb
    USE rrlw_planck, ONLY: chi_mls
    USE rrlw_kg03, ONLY: selfref
    USE rrlw_kg03, ONLY: forref
    USE rrlw_kg03, ONLY: ka_mn2o
    USE rrlw_kg03, ONLY: absa
    USE rrlw_kg03, ONLY: fracrefa
    USE rrlw_kg03, ONLY: kb_mn2o
    USE rrlw_kg03, ONLY: absb
    USE rrlw_kg03, ONLY: fracrefb
    IMPLICIT NONE
    PRIVATE
    PUBLIC taumol03_lwr,taumol03_upr
    CONTAINS
    SUBROUTINE taumol03_lwr(ncol, laytrop, nlayers,         &
                rat_h2oco2, colco2, colh2o, coln2o, coldry, &
                fac0, fac1, minorfrac,                      &
                selffac,selffrac,forfac,forfrac,            &
                jp, jt, ig, indself, &
                indfor, indminor, &
                taug, fracs)
        IMPLICIT NONE

        real(kind=wp), PARAMETER :: oneminus = 1.0_wp - 1.0e-06_wp
        integer, intent(in) :: ncol                  ! number of simd columns
        integer, intent(in) :: laytrop               ! number of layers forwer atmosphere kernel
        integer, intent(in) :: nlayers               ! total number of layers
        real(kind=wp), intent(in), dimension(ncol,0:1,nlayers) :: rat_h2oco2,fac0,fac1    ! not sure of ncol depend
        real(kind=wp), intent(in), dimension(ncol,nlayers) :: colco2,colh2o,coln2o,coldry ! these appear to be gas concentrations

        real(kind=wp), intent(in), dimension(ncol,nlayers) :: selffac,selffrac            ! not sure of ncol depend
        real(kind=wp), intent(in), dimension(ncol,nlayers) :: forfac,forfrac              ! not sure of ncol depend
        real(kind=wp), intent(in), dimension(ncol,nlayers) :: minorfrac                   ! not sure of ncol depend

        ! Look up tables and related lookup indices
        ! I assume all lookup indices depend on 3D position
        ! =================================================

        integer, intent(in) :: jp(ncol,nlayers)      ! I assume jp depends on ncol
        integer, intent(in) :: jt(ncol,0:1,nlayers)  ! likewise for jt
        integer, intent(in) :: ig                    ! ig indexes into lookup tables
        integer, intent(in) :: indself(ncol,nlayers) ! self index array
        integer, intent(in) :: indfor(ncol,nlayers)  ! for index array
        integer, intent(in) :: indminor(ncol,nlayers) ! ka_mn2o index array
        real(kind=wp), intent(out), dimension(ncol,nlayers) :: taug  ! kernel result
        real(kind=wp), intent(out), dimension(ncol,nlayers) :: fracs ! kernel result

        ! Local variable
        ! ==============

        integer :: lay   ! layer index
        integer :: i     ! specparm types index
        integer :: icol  ! column index

        ! vector temporaries
        ! ====================

        integer, dimension(1:3,1:3) :: caseTypeOperations
        integer, dimension(ncol)    :: caseType
        real(kind=wp), dimension(ncol) :: p, p4, fs
        real(kind=wp), dimension(ncol) :: fmn2o,fmn2omf
        real(kind=wp), dimension(ncol) :: fpl
        real(kind=wp), dimension(ncol) :: specmult, speccomb, specparm
        real(kind=wp), dimension(ncol) :: specmult_mn2o, speccomb_mn2o,specparm_mn2o
        real(kind=wp), dimension(ncol) :: specmult_planck, speccomb_planck,specparm_planck
        real(kind=wp), dimension(ncol) :: n2om1,n2om2,absn2o,adjcoln2o,adjfac,chi_n2o,ratn2o
        real(kind=wp), dimension(ncol,0:1) :: tau_major
        real(kind=wp), dimension(ncol) :: taufor,tauself
        integer, dimension(ncol) :: js, ind0, ind00, ind01, ind02, jmn2o, jpl

        ! Register temporaries
        ! ====================

        real(kind=wp) :: p2,fk0,fk1,fk2
        real(kind=wp) :: refrat_planck_a, refrat_planck_b
        real(kind=wp) :: refrat_m_a, refrat_m_b
        integer ::       rrpk_counter=0

        !dir$ assume_aligned jp:64
        !dir$ assume_aligned jt:64
        !dir$ assume_aligned rat_h2oco2:64
        !dir$ assume_aligned colco2:64
        !dir$ assume_aligned colh2o:64
        !dir$ assume_aligned fac0:64
        !dir$ assume_aligned fac1:64
        !dir$ assume_aligned taug:64

        !dir$ assume_aligned p:64
        !dir$ assume_aligned p4:64
        !dir$ assume_aligned specmult:64
        !dir$ assume_aligned speccomb:64
        !dir$ assume_aligned specparm:64
        !dir$ assume_aligned specmult_mn2o:64
        !dir$ assume_aligned speccomb_mn2o:64
        !dir$ assume_aligned specparm_mn2o:64
        !dir$ assume_aligned specmult_planck:64
        !dir$ assume_aligned speccomb_planck:64
        !dir$ assume_aligned specparm_planck:64
        !dir$ assume_aligned indself:64
        !dir$ assume_aligned indfor:64
        !dir$ assume_aligned indminor:64
        !dir$ assume_aligned fs:64
        !dir$ assume_aligned tau_major:64

        !dir$ assume_aligned js:64
        !dir$ assume_aligned ind0:64
        !dir$ assume_aligned ind00:64
        !dir$ assume_aligned ind01:64
        !dir$ assume_aligned ind02:64

        !dir$ assume_aligned caseTypeOperations:64
        !dir$ assume_aligned caseType:64

        ! Initialize Case type operations
        !=================================

        caseTypeOperations(1:3,1) = (/0, 1, 2/)
        caseTypeOperations(1:3,2) = (/1, 0,-1/)
        caseTypeOperations(1:3,3) = (/0, 1, 1/)

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

        ! Lower atmosphere loop
        ! =====================

        DO lay = 1,laytrop  ! loop over layers

        ! Compute tau_major term
        ! ======================

            DO i=0,1         ! loop over specparm types

                ! This loop should vectorize
                ! =============================
                !dir$ SIMD
                DO icol=1,ncol  ! Vectorizes with dir - 14.0.2
                    speccomb(icol) = colh2o(icol,lay) + rat_h2oco2(icol,i,lay)*colco2(icol,lay)
                    specparm(icol) = colh2o(icol,lay)/speccomb(icol)
                    IF (specparm(icol) .GE. oneminus) specparm(icol) = oneminus
                    specmult(icol) = 8._wp*(specparm(icol))
                    js(icol) = 1 + INT(specmult(icol))
                    fs(icol) = MOD(specmult(icol),1.0_wp)
                END DO

                ! The only conditional loop
                ! =========================

                DO icol=1,ncol  ! Vectorizes as is 14.0.2
                    IF (specparm(icol) .LT. 0.125_wp) THEN
                        caseType(icol)=1
                        p(icol)  = fs(icol) - 1.0_wp
                        p2 = p(icol)*p(icol)
                        p4(icol) = p2*p2
                    ELSE IF (specparm(icol) .GT. 0.875_wp) THEN
                        caseType(icol)=2
                        p(icol)  = -fs(icol)
                        p2 = p(icol)*p(icol)
                        p4(icol) = p2*p2
                    ELSE
                        caseType(icol)=3
                        ! SIMD way of writing fk0= 1-fs and fk1 = fs, fk2=zero
                        ! ===========================================================

                        p4(icol) = 1.0_wp - fs(icol)
                        p(icol) = -p4(icol)          ! complicated way of getting fk2 = zero for ELSE case
                    ENDIF
                END DO

                ! Vector/SIMD index loop calculation
                ! ==================================

                !dir$ SIMD
                DO icol=1,ncol  ! Vectorizes with dir 14.0.2
                    ind0(icol) = ((jp(icol,lay)-(1*MOD(i+1,2)))*5+(jt(icol,i,lay)-1))*nspa(3) +js(icol)
                    ind00(icol) = ind0(icol) + caseTypeOperations(1,caseType(icol))
                    ind01(icol) = ind0(icol) + caseTypeOperations(2,caseType(icol))
                    ind02(icol) = ind0(icol) + caseTypeOperations(3,caseType(icol))
                END DO

                ! What we've been aiming for a nice flop intensive
                ! SIMD/vectorizable loop!
                ! 17 flops
                !
                ! Albeit at the cost of a couple extra flops for the fk2 term
                ! and a repeated lookup table access for the fk2 term in the
                ! the ELSE case when specparm or specparm1 is  (> .125 && < .875)
                ! ===============================================================

                !dir$ SIMD
                DO icol=1,ncol ! Vectorizes with dir 14.0.2

                    fk0 = p4(icol)
                    fk1 = 1.0_wp - p(icol) - 2.0_wp*p4(icol)
                    fk2 = p(icol) + p4(icol)
                    tau_major(icol,i) = speccomb(icol) * ( &
                    fac0(icol,i,lay)*(fk0*absa(ind00(icol),ig)  + &
                    fk1*absa(ind01(icol),ig)  + &
                    fk2*absa(ind02(icol),ig)) + &
                    fac1(icol,i,lay)*(fk0*absa(ind00(icol)+9,ig)  + &
                    fk1*absa(ind01(icol)+9,ig)  + &
                    fk2*absa(ind02(icol)+9,ig)))
                END DO

            END DO ! end loop over specparm types for tau_major calculation

            ! Compute taufor and tauself terms:
            ! Note the use of 1D bilinear interpolation of selfref and forref
            ! lookup table values
            ! ===================================================================================

            !dir$ SIMD
            DO icol=1,ncol  ! Vectorizes with dir 14.0.2
                tauself(icol) = selffac(icol,lay)*(selfref(indself(icol,lay),ig) +&
                selffrac(icol,lay)*(selfref(indself(icol,lay)+1,ig)- selfref(indself(icol,lay),ig)))
                taufor(icol) =  forfac(icol,lay)*( forref(indfor(icol,lay),ig) +&
                forfrac(icol,lay)*( forref(indfor(icol,lay)+1,ig) -forref(indfor(icol,lay),ig)))
            END DO

            ! Compute absn2o term:
            ! Note the use of 2D bilinear interpolation ka_mn2o lookup table
            ! values
            ! =====================================================================

            !dir$ SIMD
            DO icol=1,ncol !vectorizes with dir 14.0.2
                speccomb_mn2o(icol) = colh2o(icol,lay) +refrat_m_a*colco2(icol,lay)
                specparm_mn2o(icol) = colh2o(icol,lay)/speccomb_mn2o(icol)
            END DO

            do icol=1,ncol ! vectorizes as is 14.0.2
                IF (specparm_mn2o(icol) .GE. oneminus) specparm_mn2o(icol) =oneminus
            end do

            !dir$ SIMD ! vectorizes with dir 14.0.2
            DO icol=1,ncol
                specmult_mn2o(icol) = 8.0_wp*specparm_mn2o(icol)
                jmn2o(icol) = 1 + INT(specmult_mn2o(icol))
                fmn2o(icol) = MOD(specmult_mn2o(icol),1.0_wp)
                fmn2omf(icol) = minorfrac(icol,lay)*fmn2o(icol)
            END DO

            !
            ! 2D bilinear interpolation
            ! =========================

            !dir$ SIMD
            do icol=1,ncol ! vectorizes with dir 14.0.2
                n2om1(icol)   = ka_mn2o(jmn2o(icol),indminor(icol,lay)  ,ig) + &
                fmn2o(icol)*(ka_mn2o(jmn2o(icol)+1,indminor(icol,lay),ig) - &
                ka_mn2o(jmn2o(icol),indminor(icol,lay)  ,ig))
                n2om2(icol)   = ka_mn2o(jmn2o(icol),indminor(icol,lay)+1,ig) + &
                fmn2o(icol)*(ka_mn2o(jmn2o(icol)+1,indminor(icol,lay)+1,ig)- &
                ka_mn2o(jmn2o(icol),indminor(icol,lay)+1,ig))
                absn2o(icol)  = n2om1(icol) + minorfrac(icol,lay)*(n2om2(icol) -n2om1(icol))
            end do

            !  In atmospheres where the amount of N2O is too great to be
            !  considered
            !  a minor species, adjust the column amount of N2O by an empirical
            !  factor
            !  to obtain the proper contribution.
            ! ========================================================================

            !dir$ SIMD
            do icol=1,ncol  ! vectorized with dir 14.0.2
                chi_n2o(icol) = coln2o(icol,lay)/coldry(icol,lay)
                ratn2o(icol) = 1.e20*chi_n2o(icol)/chi_mls(4,jp(icol,lay)+1)
            end do

            do icol=1,ncol ! vectorizes as is 14.0.2
                IF (ratn2o(icol) .GT. 1.5_wp) THEN
                    adjfac(icol) = 0.5_wp+(ratn2o(icol)-0.5_wp)**0.65_wp
                    adjcoln2o(icol) =adjfac(icol)*chi_mls(4,jp(icol,lay)+1)*coldry(icol,lay)*1.e-20_wp
                ELSE
                    adjcoln2o(icol) = coln2o(icol,lay)
                ENDIF
            end do

            ! Compute taug, one of two terms returned by the lower atmosphere
            ! kernel (the other is fracs)
            ! This loop could be parallelized over specparm types (i) but might
            ! produce
            ! different results for different thread counts
            ! ===========================================================================================

            !dir$ SIMD  ! DOES NOT VECTORIZE even with SIMD dir 14.0.2
            DO icol=1,ncol
                taug(icol,lay) = tau_major(icol,0) + tau_major(icol,1) +tauself(icol) + taufor(icol) + adjcoln2o(icol)*absn2o(icol)
            END DO

            !dir$ SIMD ! vectorizes with dir 14.0.2
            DO icol=1,ncol
                speccomb_planck(icol) = colh2o(icol,lay)+refrat_planck_a*colco2(icol,lay)
                specparm_planck(icol) = colh2o(icol,lay)/speccomb_planck(icol)
            END DO

            DO icol=1,ncol ! vectorizes as is 14.0.2
                IF (specparm_planck(icol) .GE. oneminus) specparm_planck(icol)=oneminus
            END DO

            !dir$ SIMD
            DO icol=1,ncol !vectorizes with dir 14.0.2
                specmult_planck(icol) = 8.0_wp*specparm_planck(icol)
                jpl(icol)= 1 + INT(specmult_planck(icol))
                fpl(icol) = MOD(specmult_planck(icol),1.0_wp)
                fracs(icol,lay) = fracrefa(ig,jpl(icol)) + fpl(icol)*(fracrefa(ig,jpl(icol)+1)-fracrefa(ig,jpl(icol)))
            END DO
            rrpk_counter=rrpk_counter+1
        END DO ! end lower atmosphere loop
    END SUBROUTINE taumol03_lwr


    SUBROUTINE taumol03_upr(ncol, laytrop, nlayers,                     &
                rat_h2oco2, colco2, colh2o, coln2o, coldry, &
                fac0, fac1, minorfrac, &
                forfac,forfrac,        &
                jp, jt, ig,        &
                indfor, indminor, &
                taug, fracs)

        use mo_kind, only : wp
        USE mo_lrtm_setup, ONLY: nspa
        USE mo_lrtm_setup, ONLY: nspb
        USE rrlw_planck, ONLY: chi_mls
        USE rrlw_kg03, ONLY: selfref
        USE rrlw_kg03, ONLY: forref
        USE rrlw_kg03, ONLY: ka_mn2o
        USE rrlw_kg03, ONLY: absa
        USE rrlw_kg03, ONLY: fracrefa
        USE rrlw_kg03, ONLY: kb_mn2o
        USE rrlw_kg03, ONLY: absb
        USE rrlw_kg03, ONLY: fracrefb

        IMPLICIT NONE

        real(kind=wp), PARAMETER :: oneminus = 1.0_wp - 1.0e-06_wp

        integer, intent(in) :: ncol                  ! number of simd columns
        integer, intent(in) :: laytrop               ! number of layers for lower atmosphere kernel
        integer, intent(in) :: nlayers               ! total number of layers
        real(kind=wp), intent(in), dimension(ncol,0:1,nlayers) :: rat_h2oco2,fac0,fac1    ! not sure of ncol depend
        real(kind=wp), intent(in), dimension(ncol,nlayers) :: colco2,colh2o,coln2o,coldry ! these appear to be gas concentrations
        real(kind=wp), intent(in), dimension(ncol,nlayers) :: forfac,forfrac ! not sure of ncol depend
        real(kind=wp), intent(in), dimension(ncol,nlayers) :: minorfrac      ! not sure of ncol depend

        ! Look up tables and related lookup indices
        ! I assume all lookup indices depend on 3D position
        ! =================================================

        integer, intent(in) :: jp(ncol,nlayers)      ! I assume jp depends on ncol
        integer, intent(in) :: jt(ncol,0:1,nlayers)  ! likewise for jt
        integer, intent(in) :: ig                    ! ig indexes into lookup tables
        integer, intent(in) :: indfor(ncol,nlayers)  ! for index array
        integer, intent(in) :: indminor(ncol,nlayers) ! ka_mn2o index array
        real(kind=wp), intent(out), dimension(ncol,nlayers) :: taug  ! kernel result
        real(kind=wp), intent(out), dimension(ncol,nlayers) :: fracs ! kernel result

        ! Local variable
        ! ==============

        integer :: lay   ! layer index
        integer :: i     ! specparm types index
        integer :: icol  ! column index

        ! vector temporaries
        ! ====================

        real(kind=wp), dimension(ncol) :: fs
        real(kind=wp), dimension(ncol) :: fmn2o,fmn2omf
        real(kind=wp), dimension(ncol) :: fpl
        real(kind=wp), dimension(ncol) :: specmult, speccomb, specparm
        real(kind=wp), dimension(ncol) :: specmult_mn2o, speccomb_mn2o, specparm_mn2o
        real(kind=wp), dimension(ncol) :: specmult_planck, speccomb_planck,specparm_planck
        real(kind=wp), dimension(ncol) :: n2om1,n2om2,absn2o,adjcoln2o,adjfac,chi_n2o,ratn2o
        real(kind=wp), dimension(ncol,0:1) :: tau_major
        real(kind=wp), dimension(ncol) :: taufor,tauself
        integer, dimension(ncol) :: js, ind0, jmn2o, jpl

        ! Register temporaries
        ! ====================

        real(kind=wp) :: p2,fk0,fk1,fk2
        real(kind=wp) :: refrat_planck_a, refrat_planck_b
        real(kind=wp) :: refrat_m_a, refrat_m_b

        !dir$ assume_aligned jp:64
        !dir$ assume_aligned jt:64
        !dir$ assume_aligned rat_h2oco2:64
        !dir$ assume_aligned colco2:64
        !dir$ assume_aligned colh2o:64
        !dir$ assume_aligned fac0:64
        !dir$ assume_aligned fac1:64
        !dir$ assume_aligned taug:64

        !dir$ assume_aligned specmult:64
        !dir$ assume_aligned speccomb:64
        !dir$ assume_aligned specparm:64
        !dir$ assume_aligned specmult_mn2o:64
        !dir$ assume_aligned speccomb_mn2o:64
        !dir$ assume_aligned specparm_mn2o:64
        !dir$ assume_aligned specmult_planck:64
        !dir$ assume_aligned speccomb_planck:64
        !dir$ assume_aligned specparm_planck:64
        !dir$ assume_aligned fs:64
        !dir$ assume_aligned tau_major:64
        !dir$ assume_aligned chi_n2o:64

        !dir$ assume_aligned js:64
        !dir$ assume_aligned ind0:64
        !dir$ assume_aligned jpl:64
        !dir$ assume_aligned fpl:64

        !dir$ assume_aligned absn2o:64
        !dir$ assume_aligned adjcoln2o:64
        !dir$ assume_aligned adjfac:64
        !dir$ assume_aligned ratn2o:64

        ! Upper atmosphere loop
        ! ========================
        refrat_planck_b = chi_mls(1,13)/chi_mls(2,13)
        refrat_m_b = chi_mls(1,13)/chi_mls(2,13)
        DO lay = laytrop+1, nlayers

            DO i=0,1         ! loop over specparm types

            ! This loop should vectorize
            ! =============================

            !dir$ SIMD
                DO icol=1,ncol ! Vectorizes with dir 14.0.2
                    speccomb(icol) = colh2o(icol,lay) + rat_h2oco2(icol,i,lay)*colco2(icol,lay)
                    specparm(icol) = colh2o(icol,lay)/speccomb(icol)
                    IF (specparm(icol) .ge. oneminus) specparm(icol) = oneminus
                    specmult(icol) = 4.0_wp*(specparm(icol))
                    js(icol) = 1 + INT(specmult(icol))
                    fs(icol) = MOD(specmult(icol),1.0_wp)
                    ind0(icol) = ((jp(icol,lay)-13+i)*5+(jt(icol,i,lay)-1))*nspb(3) +js(icol)
                END DO

                !dir$ SIMD
                DO icol=1,ncol ! Vectorizes with dir 14.0.2
                    tau_major(icol,i) = speccomb(icol) * &
                    ((1.0_wp - fs(icol))*fac0(icol,i,lay)*absb(ind0(icol)  ,ig) + &
                    fs(icol) *fac0(icol,i,lay)*absb(ind0(icol)+1,ig) + &
                    (1.0_wp - fs(icol))*fac1(icol,i,lay)*absb(ind0(icol)+5,ig) + &
                    fs(icol) *fac1(icol,i,lay)*absb(ind0(icol)+6,ig))
                END DO

            END DO ! end loop over specparm types for tau_major calculation

            ! Compute taufor terms
            ! Note the use of 1D bilinear interpolation of selfref and forref lookup
            ! table values
            ! ===================================================================================
            !dir$ SIMD
            DO icol=1,ncol ! Vectorizes with dir 14.0.2
                taufor(icol)  =  forfac(icol,lay)*( forref(indfor(icol,lay),ig) +  &
                forfrac(icol,lay)*( forref(indfor(icol,lay)+1,ig) - forref(indfor(icol,lay),ig)))
            END DO

            ! Compute absn2o term:
            ! Note the use of 2D bilinear interpolation ka_mn2o lookup table values
            ! =====================================================================
            !$DIR SIMD
            DO icol=1,ncol ! Vectorizes with dir 14.0.2
                speccomb_mn2o(icol) = colh2o(icol,lay) + refrat_m_b*colco2(icol,lay)
                specparm_mn2o(icol) = colh2o(icol,lay)/speccomb_mn2o(icol)
                IF (specparm_mn2o(icol) .GE. oneminus) specparm_mn2o(icol) = oneminus
                specmult_mn2o(icol) = 4.0_wp*specparm_mn2o(icol)
                jmn2o(icol) = 1 + INT(specmult_mn2o(icol))
                fmn2o(icol) = MOD(specmult_mn2o(icol),1.0_wp)
                fmn2omf(icol) = minorfrac(icol,lay)*fmn2o(icol)
            END DO

            !  In atmospheres where the amount of N2O is too great to be considered
            !  a minor species, adjust the column amount of N2O by an empirical factor
            !  to obtain the proper contribution.
            ! ========================================================================

            !dir$ SIMD
            DO icol=1,ncol ! loop vectorized with directive 14.0.2
                chi_n2o(icol) = coln2o(icol,lay)/coldry(icol,lay)
                ratn2o(icol) = 1.e20*chi_n2o(icol)/chi_mls(4,jp(icol,lay)+1)
            END DO

            DO icol=1,ncol ! Loop vectorized as is 14.0.2
                IF (ratn2o(icol) .GT. 1.5_wp) THEN
                    adjfac(icol) = 0.5_wp+(ratn2o(icol)-0.5_wp)**0.65_wp
                    adjcoln2o(icol) = adjfac(icol)*chi_mls(4,jp(icol,lay)+1)*coldry(icol,lay)*1.e-20_wp
                ELSE
                    adjcoln2o(icol) = coln2o(icol,lay)
                ENDIF
            END DO

            !
            ! 2D bilinear interpolation
            ! =========================

            !dir$ SIMD
            DO icol=1,ncol ! loop vectorizes with directive 14.0.2
                n2om1(icol)   = kb_mn2o(jmn2o(icol),indminor(icol,lay)  ,ig) + &
                fmn2o(icol)*(kb_mn2o(jmn2o(icol)+1,indminor(icol,lay),ig) - &
                kb_mn2o(jmn2o(icol)  ,indminor(icol,lay)  ,ig))
                n2om2(icol)   = kb_mn2o(jmn2o(icol),indminor(icol,lay)+1,ig) + &
                fmn2o(icol)*(kb_mn2o(jmn2o(icol)+1,indminor(icol,lay)+1,ig)- &
                kb_mn2o(jmn2o(icol),indminor(icol,lay)+1,ig))
                absn2o(icol)  = n2om1(icol) + minorfrac(icol,lay)*(n2om2(icol) -n2om1(icol))
            END DO

            !dir$ SIMD
            DO icol=1,ncol ! loop vectorizes with directive 14.0.2
                taug(icol,lay) =  tau_major(icol,0) + tau_major(icol,1) + taufor(icol) + adjcoln2o(icol)*absn2o(icol)
            END DO

            !dir$ SIMD
            DO icol=1,ncol ! loop vectorizes with directive 14.0.2
                speccomb_planck(icol) = colh2o(icol,lay)+refrat_planck_b*colco2(icol,lay)
                specparm_planck(icol) = colh2o(icol,lay)/speccomb_planck(icol)
            END DO

            !dir$ SIMD
            DO icol=1,ncol  ! loop vectorizes with directive 14.0.2
                IF (specparm_planck(icol) .GE. oneminus) specparm_planck(icol)=oneminus
                specmult_planck(icol) = 4.0_wp*specparm_planck(icol)
                jpl(icol)= 1 + INT(specmult_planck(icol))
                fpl(icol) = MOD(specmult_planck(icol),1.0_wp)
                fracs(icol,lay) =  fracrefb(ig,jpl(icol)) + fpl(icol)*(fracrefb(ig,jpl(icol)+1)-fracrefb(ig,jpl(icol)))
            END DO
        END DO  ! nlayers loop

    END SUBROUTINE taumol03_upr

END MODULE mo_taumol03
