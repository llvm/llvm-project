
! KGEN-generated Fortran source file
!
! Filename    : mo_imp_sol.F90
! Generated at: 2015-05-13 11:02:22
! KGEN version: 0.4.10



    MODULE mo_imp_sol
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        USE chem_mods, ONLY: gas_pcnst
        USE chem_mods, ONLY: clscnt4
        USE chem_mods, ONLY: clsmap
        IMPLICIT NONE
        PRIVATE
        PUBLIC imp_sol
        !-----------------------------------------------------------------------
        ! Newton-Raphson iteration limits
        !-----------------------------------------------------------------------
        INTEGER, parameter :: itermax = 11
        INTEGER, parameter :: cut_limit = 5
        REAL(KIND=r8) :: small
        REAL(KIND=r8) :: epsilon(clscnt4)
        LOGICAL :: factor(itermax)
        INTEGER :: ox_ndx
        INTEGER :: o1d_ndx = -1
        INTEGER :: h2o_ndx = -1
        INTEGER :: ch3co3_ndx
        INTEGER :: ho2_ndx
        INTEGER :: ch3o2_ndx
        INTEGER :: po2_ndx
        INTEGER :: oh_ndx
        INTEGER :: macro2_ndx
        INTEGER :: mco3_ndx
        INTEGER :: c2h5o2_ndx
        INTEGER :: c3h7o2_ndx
        INTEGER :: isopo2_ndx
        INTEGER :: xo2_ndx
        INTEGER :: ro2_ndx
        INTEGER :: no2_ndx
        INTEGER :: n2o5_ndx
        INTEGER :: no3_ndx
        INTEGER :: no_ndx
        INTEGER :: mvk_ndx
        INTEGER :: c2h4_ndx
        INTEGER :: c3h6_ndx
        INTEGER :: isop_ndx
        INTEGER :: c10h16_ndx
        INTEGER :: ox_p2_ndx
        INTEGER :: ox_p5_ndx
        INTEGER :: ox_p1_ndx
        INTEGER :: ox_p3_ndx
        INTEGER :: ox_p4_ndx
        INTEGER :: ox_p7_ndx
        INTEGER :: ox_p8_ndx
        INTEGER :: ox_p9_ndx
        INTEGER :: ox_p6_ndx
        INTEGER :: ox_p10_ndx
        INTEGER :: ox_p11_ndx
        INTEGER :: ox_l1_ndx
        INTEGER :: ox_l3_ndx
        INTEGER :: ox_l4_ndx
        INTEGER :: ox_l5_ndx
        INTEGER :: ox_l2_ndx
        INTEGER :: ox_l7_ndx
        INTEGER :: ox_l8_ndx
        INTEGER :: ox_l9_ndx
        INTEGER :: ox_l6_ndx
        INTEGER :: usr4_ndx
        INTEGER :: c2o3_ndx
        INTEGER :: ole_ndx
        INTEGER :: usr16_ndx
        INTEGER :: usr17_ndx
        INTEGER :: eneo2_ndx
        INTEGER :: meko2_ndx
        INTEGER :: eo2_ndx
        INTEGER :: terpo2_ndx
        INTEGER :: alko2_ndx
        INTEGER :: tolo2_ndx
        INTEGER :: ox_p17_ndx
        INTEGER :: ox_p12_ndx
        INTEGER :: ox_p14_ndx
        INTEGER :: ox_p13_ndx
        INTEGER :: ox_p16_ndx
        INTEGER :: ox_p15_ndx
        LOGICAL :: full_ozone_chem = .false.
        LOGICAL :: middle_atm_chem = .false.
        LOGICAL :: reduced_ozone_chem = .false.
        ! for xnox ozone chemistry diagnostics
        INTEGER :: o3a_ndx
        INTEGER :: o1da_ndx
        INTEGER :: xno2no3_ndx
        INTEGER :: xno2_ndx
        INTEGER :: xno3_ndx
        INTEGER :: no2xno3_ndx
        INTEGER :: xno_ndx
        INTEGER :: usr16b_ndx
        INTEGER :: usr4a_ndx
        INTEGER :: usr16a_ndx
        INTEGER :: usr17b_ndx
            PUBLIC kgen_read_externs_mo_imp_sol
        CONTAINS

        ! write subroutines

        ! module extern variables

        SUBROUTINE kgen_read_externs_mo_imp_sol(kgen_unit)
            INTEGER, INTENT(IN) :: kgen_unit
            READ(UNIT=kgen_unit) small
            READ(UNIT=kgen_unit) epsilon
            READ(UNIT=kgen_unit) factor
            READ(UNIT=kgen_unit) ox_ndx
            READ(UNIT=kgen_unit) o1d_ndx
            READ(UNIT=kgen_unit) h2o_ndx
            READ(UNIT=kgen_unit) ch3co3_ndx
            READ(UNIT=kgen_unit) ho2_ndx
            READ(UNIT=kgen_unit) ch3o2_ndx
            READ(UNIT=kgen_unit) po2_ndx
            READ(UNIT=kgen_unit) oh_ndx
            READ(UNIT=kgen_unit) macro2_ndx
            READ(UNIT=kgen_unit) mco3_ndx
            READ(UNIT=kgen_unit) c2h5o2_ndx
            READ(UNIT=kgen_unit) c3h7o2_ndx
            READ(UNIT=kgen_unit) isopo2_ndx
            READ(UNIT=kgen_unit) xo2_ndx
            READ(UNIT=kgen_unit) ro2_ndx
            READ(UNIT=kgen_unit) no2_ndx
            READ(UNIT=kgen_unit) n2o5_ndx
            READ(UNIT=kgen_unit) no3_ndx
            READ(UNIT=kgen_unit) no_ndx
            READ(UNIT=kgen_unit) mvk_ndx
            READ(UNIT=kgen_unit) c2h4_ndx
            READ(UNIT=kgen_unit) c3h6_ndx
            READ(UNIT=kgen_unit) isop_ndx
            READ(UNIT=kgen_unit) c10h16_ndx
            READ(UNIT=kgen_unit) ox_p2_ndx
            READ(UNIT=kgen_unit) ox_p5_ndx
            READ(UNIT=kgen_unit) ox_p1_ndx
            READ(UNIT=kgen_unit) ox_p3_ndx
            READ(UNIT=kgen_unit) ox_p4_ndx
            READ(UNIT=kgen_unit) ox_p7_ndx
            READ(UNIT=kgen_unit) ox_p8_ndx
            READ(UNIT=kgen_unit) ox_p9_ndx
            READ(UNIT=kgen_unit) ox_p6_ndx
            READ(UNIT=kgen_unit) ox_p10_ndx
            READ(UNIT=kgen_unit) ox_p11_ndx
            READ(UNIT=kgen_unit) ox_l1_ndx
            READ(UNIT=kgen_unit) ox_l3_ndx
            READ(UNIT=kgen_unit) ox_l4_ndx
            READ(UNIT=kgen_unit) ox_l5_ndx
            READ(UNIT=kgen_unit) ox_l2_ndx
            READ(UNIT=kgen_unit) ox_l7_ndx
            READ(UNIT=kgen_unit) ox_l8_ndx
            READ(UNIT=kgen_unit) ox_l9_ndx
            READ(UNIT=kgen_unit) ox_l6_ndx
            READ(UNIT=kgen_unit) usr4_ndx
            READ(UNIT=kgen_unit) c2o3_ndx
            READ(UNIT=kgen_unit) ole_ndx
            READ(UNIT=kgen_unit) usr16_ndx
            READ(UNIT=kgen_unit) usr17_ndx
            READ(UNIT=kgen_unit) eneo2_ndx
            READ(UNIT=kgen_unit) meko2_ndx
            READ(UNIT=kgen_unit) eo2_ndx
            READ(UNIT=kgen_unit) terpo2_ndx
            READ(UNIT=kgen_unit) alko2_ndx
            READ(UNIT=kgen_unit) tolo2_ndx
            READ(UNIT=kgen_unit) ox_p17_ndx
            READ(UNIT=kgen_unit) ox_p12_ndx
            READ(UNIT=kgen_unit) ox_p14_ndx
            READ(UNIT=kgen_unit) ox_p13_ndx
            READ(UNIT=kgen_unit) ox_p16_ndx
            READ(UNIT=kgen_unit) ox_p15_ndx
            READ(UNIT=kgen_unit) full_ozone_chem
            READ(UNIT=kgen_unit) middle_atm_chem
            READ(UNIT=kgen_unit) reduced_ozone_chem
            READ(UNIT=kgen_unit) o3a_ndx
            READ(UNIT=kgen_unit) o1da_ndx
            READ(UNIT=kgen_unit) xno2no3_ndx
            READ(UNIT=kgen_unit) xno2_ndx
            READ(UNIT=kgen_unit) xno3_ndx
            READ(UNIT=kgen_unit) no2xno3_ndx
            READ(UNIT=kgen_unit) xno_ndx
            READ(UNIT=kgen_unit) usr16b_ndx
            READ(UNIT=kgen_unit) usr4a_ndx
            READ(UNIT=kgen_unit) usr16a_ndx
            READ(UNIT=kgen_unit) usr17b_ndx
        END SUBROUTINE kgen_read_externs_mo_imp_sol



        SUBROUTINE imp_sol(base_sol, reaction_rates, het_rates, extfrc, delt, xhnm, ncol, lchnk, ltrop, o3s_loss)
            !-----------------------------------------------------------------------
            ! ... imp_sol advances the volumetric mixing ratio
            ! forward one time step via the fully implicit euler scheme.
            ! this source is meant for small l1 cache machines such as
            ! the intel pentium and itanium cpus
            !-----------------------------------------------------------------------
            USE chem_mods, ONLY: extcnt
            USE chem_mods, ONLY: rxntot
            USE chem_mods, ONLY: nzcnt
            USE chem_mods, ONLY: cls_rxt_cnt
            USE chem_mods, ONLY: permute
            USE mo_tracname, ONLY: solsym
            USE ppgrid, ONLY: pver
            USE mo_lin_matrix, ONLY: linmat
            USE mo_nln_matrix, ONLY: nlnmat
            USE mo_lu_factor, ONLY: lu_fac
            USE mo_lu_solve, ONLY: lu_slv
            USE mo_prod_loss, ONLY: imp_prod_loss
            USE mo_indprd, ONLY: indprd
            IMPLICIT NONE
            !-----------------------------------------------------------------------
            ! ... dummy args
            !-----------------------------------------------------------------------
            INTEGER, intent(in) :: ncol ! columns in chunck
            INTEGER, intent(in) :: lchnk ! chunk id
            REAL(KIND=r8), intent(in) :: delt ! time step (s)
            REAL(KIND=r8), intent(in) :: reaction_rates(ncol,pver,max(1,rxntot))
            REAL(KIND=r8), intent(in) :: extfrc(ncol,pver,max(1,extcnt))
            REAL(KIND=r8), intent(in) :: het_rates(ncol,pver,max(1,gas_pcnst)) ! rxt rates (1/cm^3/s)
            ! external in-situ forcing (1/cm^3/s)
            ! washout rates (1/s)
            REAL(KIND=r8), intent(inout) :: base_sol(ncol,pver,gas_pcnst) ! species mixing ratios (vmr)
            REAL(KIND=r8), intent(in) :: xhnm(ncol,pver)
            INTEGER, intent(in) :: ltrop(ncol) ! chemistry troposphere boundary (index)
            REAL(KIND=r8), optional, intent(out) :: o3s_loss(ncol,pver)
            !-----------------------------------------------------------------------
            ! ... local variables
            !-----------------------------------------------------------------------
            INTEGER :: m
            INTEGER :: lev
            INTEGER :: i
            INTEGER :: k
            INTEGER :: j
            INTEGER :: nr_iter
            INTEGER :: cut_cnt
            INTEGER :: fail_cnt
            INTEGER :: stp_con_cnt
            INTEGER :: nstep
            REAL(KIND=r8) :: dt
            REAL(KIND=r8) :: interval_done
            REAL(KIND=r8) :: dti
            REAL(KIND=r8) :: max_delta(max(1,clscnt4))
            REAL(KIND=r8) :: sys_jac(max(1,nzcnt))
            REAL(KIND=r8) :: lin_jac(max(1,nzcnt))
            REAL(KIND=r8), dimension(max(1,clscnt4)) :: solution
            REAL(KIND=r8), dimension(max(1,clscnt4)) :: iter_invariant
            REAL(KIND=r8), dimension(max(1,clscnt4)) :: prod
            REAL(KIND=r8), dimension(max(1,clscnt4)) :: loss
            REAL(KIND=r8), dimension(max(1,clscnt4)) :: forcing
            REAL(KIND=r8) :: lrxt(max(1,rxntot))
            REAL(KIND=r8) :: lsol(max(1,gas_pcnst))
            REAL(KIND=r8) :: lhet(max(1,gas_pcnst))
            REAL(KIND=r8), dimension(ncol,pver,max(1,clscnt4)) :: ind_prd
            LOGICAL :: convergence
            LOGICAL :: frc_mask
            LOGICAL :: converged(max(1,clscnt4))
            REAL(KIND=r8), dimension(ncol,pver,max(1,clscnt4)) :: prod_out
            REAL(KIND=r8), dimension(ncol,pver,max(1,clscnt4)) :: loss_out
            REAL(KIND=r8), dimension(ncol,pver) :: prod_hydrogen_peroxides_out
            IF (present(o3s_loss)) THEN
                o3s_loss(:,:) = 0._r8
            END IF 
            prod_out(:,:,:) = 0._r8
            loss_out(:,:,:) = 0._r8
            prod_hydrogen_peroxides_out(:,:) = 0._r8
            solution(:) = 0._r8
            !-----------------------------------------------------------------------
            ! ... class independent forcing
            !-----------------------------------------------------------------------
            IF (cls_rxt_cnt(1,4) > 0 .or. extcnt > 0) THEN
                CALL indprd(4, ind_prd, clscnt4, base_sol, extfrc, reaction_rates, ncol)
                ELSE
                DO m = 1,max(1,clscnt4)
                    ind_prd(:,:,m) = 0._r8
                END DO 
            END IF 
            level_loop: DO lev = 1,pver
                column_loop: DO i = 1,ncol
                    IF (lev <= ltrop(i)) CYCLE column_loop
                    !-----------------------------------------------------------------------
                    ! ... transfer from base to local work arrays
                    !-----------------------------------------------------------------------
                    DO m = 1,rxntot
                        lrxt(m) = reaction_rates(i,lev,m)
                    END DO 
                    IF (gas_pcnst > 0) THEN
                        DO m = 1,gas_pcnst
                            lhet(m) = het_rates(i,lev,m)
                        END DO 
                    END IF 
                    !-----------------------------------------------------------------------
                    ! ... time step loop
                    !-----------------------------------------------------------------------
                    dt = delt
                    cut_cnt = 0
                    fail_cnt = 0
                    stp_con_cnt = 0
                    interval_done = 0._r8
                    time_step_loop: DO
                        dti = 1._r8 / dt
                        !-----------------------------------------------------------------------
                        ! ... transfer from base to local work arrays
                        !-----------------------------------------------------------------------
                        DO m = 1,gas_pcnst
                            lsol(m) = base_sol(i,lev,m)
                        END DO 
                        !-----------------------------------------------------------------------
                        ! ... transfer from base to class array
                        !-----------------------------------------------------------------------
                        DO k = 1,clscnt4
                            j = clsmap(k,4)
                            m = permute(k,4)
                            solution(m) = lsol(j)
                        END DO 
                        !-----------------------------------------------------------------------
                        ! ... set the iteration invariant part of the function f(y)
                        !-----------------------------------------------------------------------
                        IF (cls_rxt_cnt(1,4) > 0 .or. extcnt > 0) THEN
                            DO m = 1,clscnt4
                                iter_invariant(m) = dti * solution(m) + ind_prd(i,lev,m)
                            END DO 
                            ELSE
                            DO m = 1,clscnt4
                                iter_invariant(m) = dti * solution(m)
                            END DO 
                        END IF 
                        !-----------------------------------------------------------------------
                        ! ... the linear component
                        !-----------------------------------------------------------------------
                        !if( cls_rxt_cnt(2,4) > 0 ) then
                        CALL linmat(lin_jac, lsol, lrxt, lhet)
                        !end if
                        !=======================================================================
                        ! the newton-raphson iteration for f(y) = 0
                        !=======================================================================
                        iter_loop: DO nr_iter = 1,itermax
                            !-----------------------------------------------------------------------
                            ! ... the non-linear component
                            !-----------------------------------------------------------------------
                            IF (factor(nr_iter)) THEN
                                CALL nlnmat(sys_jac, lsol, lrxt, lin_jac, dti)
                                !-----------------------------------------------------------------------
                                ! ... factor the "system" matrix
                                !-----------------------------------------------------------------------
                                CALL lu_fac(sys_jac)
                            END IF 
                            !-----------------------------------------------------------------------
                            ! ... form f(y)
                            !-----------------------------------------------------------------------
                            CALL imp_prod_loss(prod, loss, lsol, lrxt, lhet)
                            DO m = 1,clscnt4
                                forcing(m) = solution(m)*dti - (iter_invariant(m) + prod(m) - loss(m))
                            END DO 
                            !-----------------------------------------------------------------------
                            ! ... solve for the mixing ratio at t(n+1)
                            !-----------------------------------------------------------------------
                            CALL lu_slv(sys_jac, forcing)
                            DO m = 1,clscnt4
                                solution(m) = solution(m) + forcing(m)
                            END DO 
                            !-----------------------------------------------------------------------
                            ! ... convergence measures
                            !-----------------------------------------------------------------------
                            IF (nr_iter > 1) THEN
                                DO k = 1,clscnt4
                                    m = permute(k,4)
                                    IF (abs(solution(m)) > 1.e-20_r8) THEN
                                        max_delta(k) = abs(forcing(m)/solution(m))
                                        ELSE
                                        max_delta(k) = 0._r8
                                    END IF 
                                END DO 
                            END IF 
                            !-----------------------------------------------------------------------
                            ! ... limit iterate
                            !-----------------------------------------------------------------------
                            WHERE ( solution(:) < 0._r8 )
                                solution(:) = 0._r8
                            END WHERE 
                            !-----------------------------------------------------------------------
                            ! ... transfer latest solution back to work array
                            !-----------------------------------------------------------------------
                            DO k = 1,clscnt4
                                j = clsmap(k,4)
                                m = permute(k,4)
                                lsol(j) = solution(m)
                            END DO 
                            !-----------------------------------------------------------------------
                            ! ... check for convergence
                            !-----------------------------------------------------------------------
                            converged(:) = .true.
                            IF (nr_iter > 1) THEN
                                DO k = 1,clscnt4
                                    m = permute(k,4)
                                    frc_mask = abs(forcing(m)) > small
                                    IF (frc_mask) THEN
                                        converged(k) = abs(forcing(m)) <= epsilon(k)*abs(solution(m))
                                        ELSE
                                        converged(k) = .true.
                                    END IF 
                                END DO 
                                convergence = all(converged(:))
                                IF (convergence) THEN
                                    EXIT
                                END IF 
                            END IF 
                        END DO iter_loop
                        !-----------------------------------------------------------------------
                        ! ... check for newton-raphson convergence
                        !-----------------------------------------------------------------------
                        IF (.not. convergence) THEN
                            !-----------------------------------------------------------------------
                            ! ... non-convergence
                            !-----------------------------------------------------------------------
                            fail_cnt = fail_cnt + 1
                            !kgen_excluded nstep = get_nstep()
                            !kgen_excluded WRITE (iulog, '('' IMP_SOL: TIME STEP '',1P,E21.13,'' FAILED TO CONVERGE @ (LCHNK,LEV,
                            ! COL,NSTEP) = '',4i6)') dt, lchnk, lev, i, nstep
                            stp_con_cnt = 0
                            IF (cut_cnt < cut_limit) THEN
                                cut_cnt = cut_cnt + 1
                                IF (cut_cnt < cut_limit) THEN
                                    dt = .5_r8 * dt
                                    ELSE
                                    dt = .1_r8 * dt
                                END IF 
                                CYCLE time_step_loop
                                ELSE
                                !kgen_excluded WRITE (iulog, '('' IMP_SOL: FAILED TO CONVERGE @ (LCHNK,LEV,COL,NSTEP,DT,TIME) = ''
                                ! ,4i6,1p,2e21.13)') lchnk, lev, i, nstep, dt, interval_done+dt
                                DO m = 1,clscnt4
                                    IF (.not. converged(m)) THEN
                                        !kgen_excluded WRITE (iulog, '(1x,a8,1x,1pe10.3)') solsym(clsmap(m,4)), max_delta(m)
                                    END IF 
                                END DO 
                            END IF 
                        END IF 
                        !-----------------------------------------------------------------------
                        ! ... check for interval done
                        !-----------------------------------------------------------------------
                        interval_done = interval_done + dt
                        IF (abs( delt - interval_done ) <= .0001_r8) THEN
                            IF (fail_cnt > 0) THEN
                                !kgen_excluded WRITE (iulog, *) 'imp_sol : @ (lchnk,lev,col) = ', lchnk, lev, i, ' failed ', fail_cnt, ' times'
                            END IF 
                            EXIT time_step_loop
                            ELSE
                            !-----------------------------------------------------------------------
                            ! ... transfer latest solution back to base array
                            !-----------------------------------------------------------------------
                            IF (convergence) THEN
                                stp_con_cnt = stp_con_cnt + 1
                            END IF 
                            DO m = 1,gas_pcnst
                                base_sol(i,lev,m) = lsol(m)
                            END DO 
                            IF (stp_con_cnt >= 2) THEN
                                dt = 2._r8*dt
                                stp_con_cnt = 0
                            END IF 
                            dt = min(dt,delt-interval_done)
                            ! write(iulog,'('' imp_sol: New time step '',1p,e21.13)') dt
                        END IF 
                    END DO time_step_loop
                    !-----------------------------------------------------------------------
                    ! ... Transfer latest solution back to base array
                    !-----------------------------------------------------------------------
                    cls_loop: DO k = 1,clscnt4
                        j = clsmap(k,4)
                        m = permute(k,4)
                        base_sol(i,lev,j) = solution(m)
                    END DO cls_loop
                    !-----------------------------------------------------------------------
                    ! ... Prod/Loss history buffers...
                    !-----------------------------------------------------------------------
                    cls_loop2: DO k = 1,clscnt4
                        j = clsmap(k,4)
                        m = permute(k,4)
                        has_o3_chem: IF (( full_ozone_chem .or. reduced_ozone_chem .or. middle_atm_chem ) .and.                   &
                                    (j == ox_ndx .or. j == o3a_ndx )) THEN
                            IF (o1d_ndx < 1) THEN
                                loss_out(i,lev,k) = reaction_rates(i,lev,ox_l1_ndx)
                                ELSE
                                IF (j == ox_ndx) loss_out(i,lev,k) = reaction_rates(i,lev,ox_l1_ndx) * base_sol(i,lev,o1d_ndx)    &
                                                                       / base_sol(i,lev,ox_ndx)
                                IF (j == o3a_ndx) loss_out(i,lev,k) = reaction_rates(i,lev,ox_l1_ndx) * base_sol(i,lev,o1da_ndx)  &
                                                                        / base_sol(i,lev,o3a_ndx)
                                IF (h2o_ndx > 0) loss_out(i,lev,k) = loss_out(i,lev,k) * base_sol(i,lev,h2o_ndx)
                            END IF 
                            IF (full_ozone_chem) THEN
                                prod_out(i,lev,k) = reaction_rates(i,lev,ox_p1_ndx) * base_sol(i,lev,ho2_ndx)                     &
                                    + reaction_rates(i,lev,ox_p2_ndx) * base_sol(i,lev,ch3o2_ndx)                         + &
                                reaction_rates(i,lev,ox_p3_ndx) * base_sol(i,lev,po2_ndx)                         + &
                                reaction_rates(i,lev,ox_p4_ndx) * base_sol(i,lev,ch3co3_ndx)                         + &
                                reaction_rates(i,lev,ox_p5_ndx) * base_sol(i,lev,c2h5o2_ndx)                         + .92_r8* &
                                reaction_rates(i,lev,ox_p6_ndx) * base_sol(i,lev,isopo2_ndx)                         + &
                                reaction_rates(i,lev,ox_p7_ndx) * base_sol(i,lev,macro2_ndx)                         + &
                                reaction_rates(i,lev,ox_p8_ndx) * base_sol(i,lev,mco3_ndx)                         + &
                                reaction_rates(i,lev,ox_p9_ndx) * base_sol(i,lev,c3h7o2_ndx)                         + &
                                reaction_rates(i,lev,ox_p10_ndx)* base_sol(i,lev,ro2_ndx)                         + &
                                reaction_rates(i,lev,ox_p11_ndx)* base_sol(i,lev,xo2_ndx)                         + &
                                .9_r8*reaction_rates(i,lev,ox_p12_ndx)*base_sol(i,lev,tolo2_ndx)                         + &
                                reaction_rates(i,lev,ox_p13_ndx)*base_sol(i,lev,terpo2_ndx)                        + &
                                .9_r8*reaction_rates(i,lev,ox_p14_ndx)*base_sol(i,lev,alko2_ndx)                         + &
                                reaction_rates(i,lev,ox_p15_ndx)*base_sol(i,lev,eneo2_ndx)                         + &
                                reaction_rates(i,lev,ox_p16_ndx)*base_sol(i,lev,eo2_ndx)                         + reaction_rates(&
                                i,lev,ox_p17_ndx)*base_sol(i,lev,meko2_ndx)
                                loss_out(i,lev,k) = loss_out(i,lev,k)                         + reaction_rates(i,lev,ox_l2_ndx) * &
                                base_sol(i,lev,oh_ndx)                         + reaction_rates(i,lev,ox_l3_ndx) * base_sol(i,lev,&
                                ho2_ndx)                         + reaction_rates(i,lev,ox_l6_ndx) * base_sol(i,lev,c2h4_ndx)     &
                                                    + reaction_rates(i,lev,ox_l4_ndx) * base_sol(i,lev,c3h6_ndx)                  &
                                       + .9_r8* reaction_rates(i,lev,ox_l5_ndx) * base_sol(i,lev,isop_ndx)                        &
                                 + .8_r8*(reaction_rates(i,lev,ox_l7_ndx) * base_sol(i,lev,mvk_ndx)                         + &
                                reaction_rates(i,lev,ox_l8_ndx) * base_sol(i,lev,macro2_ndx))                         + &
                                .235_r8*reaction_rates(i,lev,ox_l9_ndx) * base_sol(i,lev,c10h16_ndx)
                                ELSE IF ( reduced_ozone_chem ) THEN
                                prod_out(i,lev,k) = reaction_rates(i,lev,ox_p1_ndx) * base_sol(i,lev,ho2_ndx)                     &
                                    + reaction_rates(i,lev,ox_p2_ndx) * base_sol(i,lev,ch3o2_ndx)                         + &
                                reaction_rates(i,lev,ox_p3_ndx) * base_sol(i,lev,c2o3_ndx)                         + &
                                reaction_rates(i,lev,ox_p11_ndx) * base_sol(i,lev,xo2_ndx)
                                loss_out(i,lev,k) = loss_out(i,lev,k)                         + reaction_rates(i,lev,ox_l2_ndx) * &
                                base_sol(i,lev,oh_ndx)                         + reaction_rates(i,lev,ox_l3_ndx) * base_sol(i,lev,&
                                ho2_ndx)                         + .9_r8* reaction_rates(i,lev,ox_l5_ndx) * base_sol(i,lev,&
                                isop_ndx)                         + reaction_rates(i,lev,ox_l6_ndx) * base_sol(i,lev,c2h4_ndx)    &
                                                     + reaction_rates(i,lev,ox_l7_ndx) * base_sol(i,lev,ole_ndx)
                                ELSE IF ( middle_atm_chem ) THEN
                                loss_out(i,lev,k) = loss_out(i,lev,k)                         + reaction_rates(i,lev,ox_l2_ndx) * &
                                base_sol(i,lev,oh_ndx)                         + reaction_rates(i,lev,ox_l3_ndx) * base_sol(i,lev,&
                                ho2_ndx)
                            END IF 
                            IF (j == ox_ndx) THEN
                                IF (.not. middle_atm_chem) THEN
                                    loss_out(i,lev,k) = loss_out(i,lev,k)                            + (reaction_rates(i,lev,&
                                    usr4_ndx) * base_sol(i,lev,no2_ndx) * base_sol(i,lev,oh_ndx)                            + &
                                    3._r8 * reaction_rates(i,lev,usr16_ndx) * base_sol(i,lev,n2o5_ndx)                            &
                                    + 2._r8 * reaction_rates(i,lev,usr17_ndx) * base_sol(i,lev,no3_ndx))                          &
                                      / max(base_sol(i,lev,ox_ndx),1.e-20_r8)
                                END IF 
                                IF (present(o3s_loss)) THEN
                                    o3s_loss(i,lev) = loss_out(i,lev,k)
                                END IF 
                                loss_out(i,lev,k) = loss_out(i,lev,k) * base_sol(i,lev,ox_ndx)
                                prod_out(i,lev,k) = prod_out(i,lev,k) * base_sol(i,lev,no_ndx)
                                ELSE IF (j == o3a_ndx) THEN
                                loss_out(i,lev,k) = loss_out(i,lev,k)                         + (reaction_rates(i,lev,usr4a_ndx) &
                                * base_sol(i,lev,xno2_ndx) * base_sol(i,lev,oh_ndx)                         + 1._r8 * &
                                reaction_rates(i,lev,usr16a_ndx) * base_sol(i,lev,xno2no3_ndx)                         + 2._r8 * &
                                reaction_rates(i,lev,usr16b_ndx) * base_sol(i,lev,no2xno3_ndx)                         + 2._r8 * &
                                reaction_rates(i,lev,usr17b_ndx) * base_sol(i,lev,xno3_ndx))                         / max(&
                                base_sol(i,lev,o3a_ndx),1.e-20_r8)
                                loss_out(i,lev,k) = loss_out(i,lev,k) * base_sol(i,lev,o3a_ndx)
                                prod_out(i,lev,k) = prod_out(i,lev,k) * base_sol(i,lev,xno_ndx)
                            END IF 
                            ELSE
                            prod_out(i,lev,k) = prod(m) + ind_prd(i,lev,m)
                            loss_out(i,lev,k) = loss(m)
                        END IF has_o3_chem
                    END DO cls_loop2
                END DO column_loop
            END DO level_loop
            DO i = 1,clscnt4
                j = clsmap(i,4)
                prod_out(:,:,i) = prod_out(:,:,i)*xhnm
                loss_out(:,:,i) = loss_out(:,:,i)*xhnm
                !kgen_excluded CALL outfld(trim(solsym(j))//'_CHMP', prod_out(:,:,i), ncol, lchnk)
                !kgen_excluded CALL outfld(trim(solsym(j))//'_CHML', loss_out(:,:,i), ncol, lchnk)
                !
                ! added code for ROOH production !PJY not "RO2 production"
                !
                IF (trim(solsym(j)) == 'ALKOOH'         .or.trim(solsym(j)) == 'C2H5OOH'         .or.trim(solsym(j)) == 'CH3OOH'  &
                       .or.trim(solsym(j)) == 'CH3COOH'         .or.trim(solsym(j)) == 'CH3COOOH'         .or.trim(solsym(j)) == &
                'C3H7OOH'         .or.trim(solsym(j)) == 'EOOH'         .or.trim(solsym(j)) == 'ISOPOOH'         .or.trim(solsym(&
                j)) == 'MACROOH'         .or.trim(solsym(j)) == 'MEKOOH'         .or.trim(solsym(j)) == 'POOH'         .or.trim(&
                solsym(j)) == 'ROOH'         .or.trim(solsym(j)) == 'TERPOOH'         .or.trim(solsym(j)) == 'TOLOOH'         &
                &.or.trim(solsym(j)) == 'XOOH') THEN
                    !PJY added this
                    !PJY corrected this (from CH3H7OOH)
                    ! .or.trim(solsym(j)) == 'H2O2' & !PJY removed as H2O2 production asked for separately (as I read 4.2.3, point 7)
                    ! .or.trim(solsym(j)) == 'HCOOH' & !PJY removed this as this is formic acid HC(O)OH - i.e. not H-C-O-O-H
                    !
                    prod_hydrogen_peroxides_out(:,:) = prod_hydrogen_peroxides_out(:,:) + prod_out(:,:,i)
                    !
                END IF 
                !
            END DO 
            !
            !kgen_excluded CALL outfld('H_PEROX_CHMP', prod_hydrogen_peroxides_out(:,:), ncol, lchnk)
            !
        END SUBROUTINE imp_sol
    END MODULE mo_imp_sol
