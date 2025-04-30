
! KGEN-generated Fortran source file
!
! Filename    : prim_advance_mod.F90
! Generated at: 2015-04-12 19:37:49
! KGEN version: 0.4.9



    MODULE prim_advance_mod
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
    USE element_mod, ONLY : kgen_read_mod9 => kgen_read
    USE element_mod, ONLY : kgen_verify_mod9 => kgen_verify
        ! _EXTERNAL
        IMPLICIT NONE
        PRIVATE
        PUBLIC compute_and_apply_rhs
        CONTAINS

        ! write subroutines
        ! No subroutines
        ! No module extern variables









        !
        ! phl notes: output is stored in first argument. Advances from 2nd argument using tendencies evaluated at 3rd rgument:
        ! phl: for offline winds use time at 3rd argument (same as rhs currently)
        !

        SUBROUTINE compute_and_apply_rhs(elem, kgen_unit)
                USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
            ! ===================================
            ! compute the RHS, accumulate into u(np1) and apply DSS
            !
            !           u(np1) = u(nm1) + dt2*DSS[ RHS(u(n0)) ]
            !
            ! This subroutine is normally called to compute a leapfrog timestep
            ! but by adjusting np1,nm1,n0 and dt2, many other timesteps can be
            ! accomodated.  For example, setting nm1=np1=n0 this routine will
            ! take a forward euler step, overwriting the input with the output.
            !
            !    qn0 = timelevel used to access Qdp() in order to compute virtual Temperature
            !          qn0=-1 for the dry case
            !
            ! if  dt2<0, then the DSS'd RHS is returned in timelevel np1
            !
            ! Combining the RHS and DSS pack operation in one routine
            ! allows us to fuse these two loops for more cache reuse
            !
            ! Combining the dt advance and DSS unpack operation in one routine
            ! allows us to fuse these two loops for more cache reuse
            !
            ! note: for prescribed velocity case, velocity will be computed at
            ! "real_time", which should be the time of timelevel n0.
            !
            !
            ! ===================================
            USE kinds, ONLY: real_kind
            USE dimensions_mod, ONLY: np
            USE dimensions_mod, ONLY: nlev
            USE element_mod, ONLY: element_t
            USE prim_si_mod, ONLY: preq_hydrostatic
            IMPLICIT NONE
            integer, intent(in) :: kgen_unit
            INTEGER*8 :: kgen_intvar, start_clock, stop_clock, rate_clock
            TYPE(check_t):: check_status
            REAL(KIND=kgen_dp) :: tolerance
            TYPE(element_t), intent(inout), target :: elem(:)
            ! weighting for eta_dot_dpdn mean flux
            ! local
            ! surface pressure for current tiime level
            REAL(KIND=real_kind), pointer, dimension(:,:,:) :: phi
            REAL(KIND=real_kind), pointer :: ref_phi(:,:,:) => NULL()
            REAL(KIND=real_kind), dimension(np,np,nlev) :: t_v
            ! half level vertical velocity on p-grid
            ! temporary field
            ! generic gradient storage
            ! v.grad(T)
            ! kinetic energy + PHI term
            ! lat-lon coord version
            ! vorticity
            REAL(KIND=real_kind), dimension(np,np,nlev) :: p ! pressure
            REAL(KIND=real_kind), dimension(np,np,nlev) :: dp ! delta pressure
            ! inverse of delta pressure
            ! temperature vertical advection
            ! v.grad(p)
            ! half level pressures on p-grid
            ! velocity vertical advection
            INTEGER :: ie
            !JMD  call t_barrierf('sync_compute_and_apply_rhs', hybrid%par%comm)
                tolerance = 1.E-14
                CALL kgen_init_check(check_status, tolerance)
                CALL kgen_read_real_real_kind_dim3_ptr(phi, kgen_unit)
                READ(UNIT=kgen_unit) t_v
                READ(UNIT=kgen_unit) p
                READ(UNIT=kgen_unit) dp
                READ(UNIT=kgen_unit) ie

                CALL kgen_read_real_real_kind_dim3_ptr(ref_phi, kgen_unit)


                ! call to kernel
                CALL preq_hydrostatic(phi, elem(ie)%state%phis, t_v, p, dp)
                ! kernel verification for output variables
                CALL kgen_verify_real_real_kind_dim3_ptr( "phi", check_status, phi, ref_phi)
                CALL kgen_print_check("preq_hydrostatic", check_status)
                CALL system_clock(start_clock, rate_clock)
                DO kgen_intvar=1,10
                    CALL preq_hydrostatic(phi, elem(ie) % state % phis, t_v, p, dp)
                END DO
                CALL system_clock(stop_clock, rate_clock)
                WRITE(*,*)
                PRINT *, "Elapsed time (sec): ", (stop_clock - start_clock)/REAL(rate_clock*10)
            ! =============================================================
            ! Insert communications here: for shared memory, just a single
            ! sync is required
            ! =============================================================
        CONTAINS

        ! write subroutines
            SUBROUTINE kgen_read_real_real_kind_dim3_ptr(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                real(KIND=real_kind), INTENT(OUT), POINTER, DIMENSION(:,:,:) :: var
                LOGICAL :: is_true
                INTEGER :: idx1,idx2,idx3
                INTEGER, DIMENSION(2,3) :: kgen_bound

                READ(UNIT = kgen_unit) is_true

                IF ( is_true ) THEN
                    READ(UNIT = kgen_unit) kgen_bound(1, 1)
                    READ(UNIT = kgen_unit) kgen_bound(2, 1)
                    READ(UNIT = kgen_unit) kgen_bound(1, 2)
                    READ(UNIT = kgen_unit) kgen_bound(2, 2)
                    READ(UNIT = kgen_unit) kgen_bound(1, 3)
                    READ(UNIT = kgen_unit) kgen_bound(2, 3)
                    ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1, kgen_bound(2, 3) - kgen_bound(1, 3) + 1))
                    READ(UNIT = kgen_unit) var
                    IF ( PRESENT(printvar) ) THEN
                        PRINT *, "** " // printvar // " **", var
                    END IF
                END IF
            END SUBROUTINE kgen_read_real_real_kind_dim3_ptr


        ! verify subroutines
            SUBROUTINE kgen_verify_real_real_kind_dim3_ptr( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                real(KIND=real_kind), intent(in), DIMENSION(:,:,:), POINTER :: var, ref_var
                real(KIND=real_kind) :: nrmsdiff, rmsdiff
                real(KIND=real_kind), allocatable, DIMENSION(:,:,:) :: temp, temp2
                integer :: n
                IF ( ASSOCIATED(var) ) THEN
                check_status%numTotal = check_status%numTotal + 1
                IF ( ALL( var == ref_var ) ) THEN
                
                    check_status%numIdentical = check_status%numIdentical + 1            
                    if(check_status%verboseLevel > 1) then
                        WRITE(*,*)
                        WRITE(*,*) "All elements of ", trim(adjustl(varname)), " are IDENTICAL."
                        !WRITE(*,*) "KERNEL: ", var
                        !WRITE(*,*) "REF.  : ", ref_var
                        IF ( ALL( var == 0 ) ) THEN
                            if(check_status%verboseLevel > 2) then
                                WRITE(*,*) "All values are zero."
                            end if
                        END IF
                    end if
                ELSE
                    allocate(temp(SIZE(var,dim=1),SIZE(var,dim=2),SIZE(var,dim=3)))
                    allocate(temp2(SIZE(var,dim=1),SIZE(var,dim=2),SIZE(var,dim=3)))
                
                    n = count(var/=ref_var)
                    where(abs(ref_var) > check_status%minvalue)
                        temp  = ((var-ref_var)/ref_var)**2
                        temp2 = (var-ref_var)**2
                    elsewhere
                        temp  = (var-ref_var)**2
                        temp2 = temp
                    endwhere
                    nrmsdiff = sqrt(sum(temp)/real(n))
                    rmsdiff = sqrt(sum(temp2)/real(n))
                
                    if(check_status%verboseLevel > 0) then
                        WRITE(*,*)
                        WRITE(*,*) trim(adjustl(varname)), " is NOT IDENTICAL."
                        WRITE(*,*) count( var /= ref_var), " of ", size( var ), " elements are different."
                        if(check_status%verboseLevel > 1) then
                            WRITE(*,*) "Average - kernel ", sum(var)/real(size(var))
                            WRITE(*,*) "Average - reference ", sum(ref_var)/real(size(ref_var))
                        endif
                        WRITE(*,*) "RMS of difference is ",rmsdiff
                        WRITE(*,*) "Normalized RMS of difference is ",nrmsdiff
                    end if
                
                    if (nrmsdiff > check_status%tolerance) then
                        check_status%numFatal = check_status%numFatal+1
                    else
                        check_status%numWarning = check_status%numWarning+1
                    endif
                
                    deallocate(temp,temp2)
                END IF
                END IF
            END SUBROUTINE kgen_verify_real_real_kind_dim3_ptr

        END SUBROUTINE compute_and_apply_rhs
        !TRILINOS


    END MODULE prim_advance_mod
