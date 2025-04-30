
! KGEN-generated Fortran source file
!
! Filename    : prim_advection_mod.F90
! Generated at: 2015-03-03 13:07:29
! KGEN version: 0.4.4











    MODULE prim_advection_mod
        !
        ! two formulations.  both are conservative
        ! u grad Q formulation:
        !
        !    d/dt[ Q] +  U grad Q  +  eta_dot dp/dn dQ/dp  = 0
        !                            ( eta_dot dQ/dn )
        !
        !    d/dt[ dp/dn ] = div( dp/dn U ) + d/dn ( eta_dot dp/dn )
        !
        ! total divergence formulation:
        !    d/dt[dp/dn Q] +  div( U dp/dn Q ) + d/dn ( eta_dot dp/dn Q ) = 0
        !
        ! for convience, rewrite this as dp Q:  (since dn does not depend on time or the horizonal):
        ! equation is now:
        !    d/dt[dp Q] +  div( U dp Q ) + d( eta_dot_dpdn Q ) = 0
        !
        !
        USE kinds, ONLY: real_kind
        ! _EXTERNAL
        IMPLICIT NONE
        PRIVATE
        PUBLIC read_externs_prim_advection_mod
        INTEGER, PARAMETER :: kgen_dp = selected_real_kind(15, 307)
        PUBLIC euler_step
        type, public  ::  check_t
            logical :: Passed
            integer :: numFatal
            integer :: numTotal
            integer :: numIdentical
            integer :: numWarning
            integer :: VerboseLevel
            real(kind=kgen_dp) :: tolerance
        end type check_t
        REAL(KIND=real_kind), allocatable :: qmin(:,:,:)
        REAL(KIND=real_kind), allocatable :: qmax(:,:,:)
        ! derivative struct (nthreads)
        CONTAINS

        ! module extern variables

        SUBROUTINE read_externs_prim_advection_mod(kgen_unit)
        integer, intent(in) :: kgen_unit
        call read_var_real_real_kind_dim3(qmin, kgen_unit)
        call read_var_real_real_kind_dim3(qmax, kgen_unit)
        
        CONTAINS
        subroutine read_var_real_real_kind_dim3(var, kgen_unit)
            integer, intent(in) :: kgen_unit
            real(kind=real_kind), intent(out), dimension(:,:,:), allocatable :: var
            integer, dimension(2,3) :: kgen_bound
            logical is_save
            
            READ(UNIT = kgen_unit) is_save
            if ( is_save ) then
                READ(UNIT = kgen_unit) kgen_bound(1, 1)
                READ(UNIT = kgen_unit) kgen_bound(2, 1)
                READ(UNIT = kgen_unit) kgen_bound(1, 2)
                READ(UNIT = kgen_unit) kgen_bound(2, 2)
                READ(UNIT = kgen_unit) kgen_bound(1, 3)
                READ(UNIT = kgen_unit) kgen_bound(2, 3)
                ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1, kgen_bound(2, 3) - kgen_bound(1, 3) + 1))
                READ(UNIT = kgen_unit) var
            end if
        end subroutine
        END SUBROUTINE read_externs_prim_advection_mod

        subroutine kgen_init_check(check,tolerance)
          type(check_t), intent(inout) :: check
          real(kind=kgen_dp), intent(in), optional :: tolerance
           check%Passed   = .TRUE.
           check%numFatal = 0
           check%numWarning = 0
           check%numTotal = 0
           check%numIdentical = 0
           check%VerboseLevel = 1
           if(present(tolerance)) then
             check%tolerance = tolerance
           else
              check%tolerance = 1.E-14
           endif
        end subroutine kgen_init_check
        subroutine kgen_print_check(kname, check)
           character(len=*) :: kname
           type(check_t), intent(in) ::  check
           write (*,*)
           write (*,*) TRIM(kname),' KGENPrtCheck: Tolerance for normalized RMS: ',check%tolerance
           write (*,*) TRIM(kname),' KGENPrtCheck: Number of variables checked: ',check%numTotal
           write (*,*) TRIM(kname),' KGENPrtCheck: Number of Identical results: ',check%numIdentical
           write (*,*) TRIM(kname),' KGENPrtCheck: Number of warnings detected: ',check%numWarning
           write (*,*) TRIM(kname),' KGENPrtCheck: Number of fatal errors detected: ', check%numFatal
           if (check%numFatal> 0) then
                write(*,*) TRIM(kname),' KGENPrtCheck: verification FAILED'
           else
                write(*,*) TRIM(kname),' KGENPrtCheck: verification PASSED'
           endif
        end subroutine kgen_print_check


        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ! fvm driver
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        !=================================================================================================!





        ! ----------------------------------------------------------------------------------!
        !SUBROUTINE ALE_RKDSS-----------------------------------------------CE-for FVM!
        ! AUTHOR: CHRISTOPH ERATH, MARK TAYLOR, 06. December 2012
        !
        ! DESCRIPTION: ! create a runge kutta taylor serios mixture to calculate the departure grid
        !
        ! CALLS:
        ! INPUT:
        !
        ! OUTPUT:
        !-----------------------------------------------------------------------------------!
        ! this will calculate the velocity at time t+1/2  along the trajectory s(t) given the velocities
        ! at the GLL points at time t and t+1 using a second order time accurate formulation.

        ! ----------------------------------------------------------------------------------!
        !SUBROUTINE FVM_DEP_FROM_GLL----------------------------------------------CE-for FVM!
        ! AUTHOR: CHRISTOPH ERATH, MARK TAYLOR 14. December 2011                            !
        ! DESCRIPTION: calculates the deparute grid for fvm coming from the gll points      !
        !                                                                                   !
        ! CALLS:
        ! INPUT:
        !
        ! OUTPUT:
        !-----------------------------------------------------------------------------------!










        !-----------------------------------------------------------------------------
        !-----------------------------------------------------------------------------
        ! forward-in-time 2 level vertically lagrangian step
        !  this code takes a lagrangian step in the horizontal
        ! (complete with DSS), and then applies a vertical remap
        !
        ! This routine may use dynamics fields at timelevel np1
        ! In addition, other fields are required, which have to be
        ! explicitly saved by the dynamics:  (in elem(ie)%derived struct)
        !
        ! Fields required from dynamics: (in
        !    omega_p   it will be DSS'd here, for later use by CAM physics
        !              we DSS omega here because it can be done for "free"
        !    Consistent mass/tracer-mass advection (used if subcycling turned on)
        !       dp()   dp at timelevel n0
        !       vn0()  mean flux  < U dp > going from n0 to np1
        !
        ! 3 stage
        !    Euler step from t     -> t+.5
        !    Euler step from t+.5  -> t+1.0
        !    Euler step from t+1.0 -> t+1.5
        !    u(t) = u(t)/3 + u(t+2)*2/3
        !
        !-----------------------------------------------------------------------------
        !-----------------------------------------------------------------------------

        !-----------------------------------------------------------------------------
        !-----------------------------------------------------------------------------

        !-----------------------------------------------------------------------------
        !-----------------------------------------------------------------------------

        SUBROUTINE euler_step(kgen_unit)
            ! ===================================
            ! This routine is the basic foward
            ! euler component used to construct RK SSP methods
            !
            !           u(np1) = u(n0) + dt2*DSS[ RHS(u(n0)) ]
            !
            ! n0 can be the same as np1.
            !
            ! DSSopt = DSSeta or DSSomega:   also DSS eta_dot_dpdn or omega
            !
            ! ===================================
            USE kinds, ONLY: real_kind
            USE dimensions_mod, ONLY: np
            USE dimensions_mod, ONLY: nlev
            IMPLICIT NONE
            integer, intent(in) :: kgen_unit

            ! read interface
            !interface kgen_read_var
            !    procedure read_var_real_real_kind_dim3
            !    procedure read_var_real_real_kind_dim2
            !end interface kgen_read_var



            ! verification interface
            interface kgen_verify_var
                procedure verify_var_logical
                procedure verify_var_integer
                procedure verify_var_real
                procedure verify_var_character
                procedure verify_var_real_real_kind_dim3
                procedure verify_var_real_real_kind_dim2
            end interface kgen_verify_var

            INTEGER*8 :: kgen_intvar, start_clock, stop_clock, rate_clock
            TYPE(check_t):: check_status
            REAL(KIND=kgen_dp) :: tolerance
            ! local
            REAL(KIND=real_kind), dimension(np,np  ,nlev) :: qtens
            REAL(KIND=real_kind), allocatable :: ref_qtens(:,:,:)
            REAL(KIND=real_kind), dimension(np,np  ,nlev) :: dp_star
            REAL(KIND=real_kind), dimension(np,np) :: smaug
            INTEGER :: ie
            INTEGER :: ref_ie
            INTEGER :: q
            INTEGER :: ref_q
            ! call t_barrierf('sync_euler_step', hybrid%par%comm)
            !   call t_startf('euler_step')
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            !   compute Q min/max values for lim8
            !   compute biharmonic mixing term f
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            ! compute biharmonic mixing term and qmin/qmax
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            !   2D Advection step
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        tolerance = 1.E-14
                        CALL kgen_init_check(check_status, tolerance)
                        READ(UNIT=kgen_unit) qtens
                        READ(UNIT=kgen_unit) dp_star
                        READ(UNIT=kgen_unit) smaug
                        READ(UNIT=kgen_unit) ie
                        READ(UNIT=kgen_unit) q
                        call read_var_real_real_kind_dim3(ref_qtens, kgen_unit)
                        READ(UNIT=kgen_unit) ref_ie
                        READ(UNIT=kgen_unit) ref_q
                        ! call to kernel
                        CALL limiter_optim_iter_full(qtens(:, :, :), smaug(:, :), qmin(:, q, ie), qmax(:, q, ie), dp_star(:, :, :))
                        ! kernel verification for output variables
                        call kgen_verify_var("qtens", check_status, qtens, ref_qtens)
                        call kgen_verify_var("ie", check_status, ie, ref_ie)
                        call kgen_verify_var("q", check_status, q, ref_q)
                        CALL kgen_print_check("limiter_optim_iter_full", check_status)
                        CALL system_clock(start_clock, rate_clock)
                        DO kgen_intvar=1,10
                            CALL limiter_optim_iter_full(qtens(:, :, :), smaug(:, :), qmin(:, q, ie), qmax(:, q, ie), dp_star(:, :, :))
                        END DO
                        CALL system_clock(stop_clock, rate_clock)
                        WRITE(*,*)
                        PRINT *, "Elapsed time (sec): ", (stop_clock - start_clock)/REAL(rate_clock*10)
            !   call t_stopf('euler_step')
        CONTAINS

        ! read subroutines
        subroutine read_var_real_real_kind_dim3(var, kgen_unit)
            integer, intent(in) :: kgen_unit
            real(kind=real_kind), intent(out), dimension(:,:,:), allocatable :: var
            integer, dimension(2,3) :: kgen_bound
            logical is_save
            
            READ(UNIT = kgen_unit) is_save
            if ( is_save ) then
                READ(UNIT = kgen_unit) kgen_bound(1, 1)
                READ(UNIT = kgen_unit) kgen_bound(2, 1)
                READ(UNIT = kgen_unit) kgen_bound(1, 2)
                READ(UNIT = kgen_unit) kgen_bound(2, 2)
                READ(UNIT = kgen_unit) kgen_bound(1, 3)
                READ(UNIT = kgen_unit) kgen_bound(2, 3)
                ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1, kgen_bound(2, 3) - kgen_bound(1, 3) + 1))
                READ(UNIT = kgen_unit) var
            end if
        end subroutine
        subroutine read_var_real_real_kind_dim2(var, kgen_unit)
            integer, intent(in) :: kgen_unit
            real(kind=real_kind), intent(out), dimension(:,:), allocatable :: var
            integer, dimension(2,2) :: kgen_bound
            logical is_save
            
            READ(UNIT = kgen_unit) is_save
            if ( is_save ) then
                READ(UNIT = kgen_unit) kgen_bound(1, 1)
                READ(UNIT = kgen_unit) kgen_bound(2, 1)
                READ(UNIT = kgen_unit) kgen_bound(1, 2)
                READ(UNIT = kgen_unit) kgen_bound(2, 2)
                ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
                READ(UNIT = kgen_unit) var
            end if
        end subroutine

        subroutine verify_var_logical(varname, check_status, var, ref_var)
            character(*), intent(in) :: varname
            type(check_t), intent(inout) :: check_status
            logical, intent(in) :: var, ref_var
        
            check_status%numTotal = check_status%numTotal + 1
            IF ( var .eqv. ref_var ) THEN
                check_status%numIdentical = check_status%numIdentical + 1
                if(check_status%verboseLevel > 1) then
                    WRITE(*,*)
                    WRITE(*,*) trim(adjustl(varname)), " is IDENTICAL( ", var, " )."
                endif
            ELSE
                if(check_status%verboseLevel > 1) then
                    WRITE(*,*)
                    WRITE(*,*) trim(adjustl(varname)), " is NOT IDENTICAL."
                    if(check_status%verboseLevel > 2) then
                        WRITE(*,*) "KERNEL: ", var
                        WRITE(*,*) "REF.  : ", ref_var
                    endif
                endif
                check_status%numFatal = check_status%numFatal + 1
            END IF
        end subroutine
        
        subroutine verify_var_integer(varname, check_status, var, ref_var)
            character(*), intent(in) :: varname
            type(check_t), intent(inout) :: check_status
            integer, intent(in) :: var, ref_var
        
            check_status%numTotal = check_status%numTotal + 1
            IF ( var == ref_var ) THEN
                check_status%numIdentical = check_status%numIdentical + 1
                if(check_status%verboseLevel > 1) then
                    WRITE(*,*)
                    WRITE(*,*) trim(adjustl(varname)), " is IDENTICAL( ", var, " )."
                endif
            ELSE
                if(check_status%verboseLevel > 0) then
                    WRITE(*,*)
                    WRITE(*,*) trim(adjustl(varname)), " is NOT IDENTICAL."
                    if(check_status%verboseLevel > 2) then
                        WRITE(*,*) "KERNEL: ", var
                        WRITE(*,*) "REF.  : ", ref_var
                    endif
                endif
                check_status%numFatal = check_status%numFatal + 1
            END IF
        end subroutine
        
        subroutine verify_var_real(varname, check_status, var, ref_var)
            character(*), intent(in) :: varname
            type(check_t), intent(inout) :: check_status
            real, intent(in) :: var, ref_var
        
            check_status%numTotal = check_status%numTotal + 1
            IF ( var == ref_var ) THEN
                check_status%numIdentical = check_status%numIdentical + 1
                if(check_status%verboseLevel > 1) then
                    WRITE(*,*)
                    WRITE(*,*) trim(adjustl(varname)), " is IDENTICAL( ", var, " )."
                endif
            ELSE
                if(check_status%verboseLevel > 0) then
                    WRITE(*,*)
                    WRITE(*,*) trim(adjustl(varname)), " is NOT IDENTICAL."
                    if(check_status%verboseLevel > 2) then
                        WRITE(*,*) "KERNEL: ", var
                        WRITE(*,*) "REF.  : ", ref_var
                    endif
                endif
                check_status%numFatal = check_status%numFatal + 1
            END IF
        end subroutine
        
        subroutine verify_var_character(varname, check_status, var, ref_var)
            character(*), intent(in) :: varname
            type(check_t), intent(inout) :: check_status
            character(*), intent(in) :: var, ref_var
        
            check_status%numTotal = check_status%numTotal + 1
            IF ( var == ref_var ) THEN
                check_status%numIdentical = check_status%numIdentical + 1
                if(check_status%verboseLevel > 1) then
                    WRITE(*,*)
                    WRITE(*,*) trim(adjustl(varname)), " is IDENTICAL( ", var, " )."
                endif
            ELSE
                if(check_status%verboseLevel > 0) then
                    WRITE(*,*)
                    WRITE(*,*) trim(adjustl(varname)), " is NOT IDENTICAL."
                    if(check_status%verboseLevel > 2) then
                        WRITE(*,*) "KERNEL: ", var
                        WRITE(*,*) "REF.  : ", ref_var
                    end if
                end if
                check_status%numFatal = check_status%numFatal + 1
            END IF
        end subroutine

        subroutine verify_var_real_real_kind_dim3(varname, check_status, var, ref_var)
            character(*), intent(in) :: varname
            type(check_t), intent(inout) :: check_status
            real(kind=real_kind), intent(in), dimension(:,:,:) :: var
            real(kind=real_kind), intent(in), allocatable, dimension(:,:,:) :: ref_var
            real(kind=real_kind) :: nrmsdiff, rmsdiff
            real(kind=real_kind), allocatable :: temp(:,:,:), temp2(:,:,:)
            integer :: n
        
        
            IF ( ALLOCATED(ref_var) ) THEN
                check_status%numTotal = check_status%numTotal + 1
                allocate(temp(SIZE(var,dim=1),SIZE(var,dim=2),SIZE(var,dim=3)))
                allocate(temp2(SIZE(var,dim=1),SIZE(var,dim=2),SIZE(var,dim=3)))
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
                    n = count(var/=ref_var)
                    where(ref_var .NE. 0)
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
                END IF
                deallocate(temp,temp2)
            END IF
        end subroutine

        subroutine verify_var_real_real_kind_dim2(varname, check_status, var, ref_var)
            character(*), intent(in) :: varname
            type(check_t), intent(inout) :: check_status
            real(kind=real_kind), intent(in), dimension(:,:) :: var
            real(kind=real_kind), intent(in), allocatable, dimension(:,:) :: ref_var
            real(kind=real_kind) :: nrmsdiff, rmsdiff
            real(kind=real_kind), allocatable :: temp(:,:), temp2(:,:)
            integer :: n
        
        
            IF ( ALLOCATED(ref_var) ) THEN
                check_status%numTotal = check_status%numTotal + 1
                allocate(temp(SIZE(var,dim=1),SIZE(var,dim=2)))
                allocate(temp2(SIZE(var,dim=1),SIZE(var,dim=2)))
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
                    n = count(var/=ref_var)
                    where(ref_var .NE. 0)
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
                END IF
                deallocate(temp,temp2)
            END IF
        end subroutine

        END SUBROUTINE euler_step
        !-----------------------------------------------------------------------------
        !-----------------------------------------------------------------------------
        !-----------------------------------------------------------------------------
        !-----------------------------------------------------------------------------
        !-----------------------------------------------------------------------------

        !-----------------------------------------------------------------------------
        !-----------------------------------------------------------------------------
        !-----------------------------------------------------------------------------

        SUBROUTINE limiter_optim_iter_full(ptens, sphweights, minp, maxp, dpmass)
            !THIS IS A NEW VERSION OF LIM8, POTENTIALLY FASTER BECAUSE INCORPORATES KNOWLEDGE FROM
            !PREVIOUS ITERATIONS
            !The idea here is the following: We need to find a grid field which is closest
            !to the initial field (in terms of weighted sum), but satisfies the min/max constraints.
            !So, first we find values which do not satisfy constraints and bring these values
            !to a closest constraint. This way we introduce some mass change (addmass),
            !so, we redistribute addmass in the way that l2 error is smallest.
            !This redistribution might violate constraints thus, we do a few iterations.
            USE kinds, ONLY: real_kind
            USE dimensions_mod, ONLY: np, np, nlev
            REAL(KIND=real_kind), dimension(np*np,nlev), intent(inout) :: ptens
            REAL(KIND=real_kind), dimension(np*np), intent(in   ) :: sphweights
            REAL(KIND=real_kind), dimension(      nlev), intent(inout) :: minp
            REAL(KIND=real_kind), dimension(      nlev), intent(inout) :: maxp
            REAL(KIND=real_kind), dimension(np*np,nlev), intent(in   ), optional :: dpmass
            REAL(KIND=real_kind), dimension(np*np,nlev) :: weights
            INTEGER :: k1, k, i, j, iter, i1, i2
            INTEGER :: whois_neg(np*np), whois_pos(np*np), neg_counter, pos_counter
            REAL(KIND=real_kind) :: addmass, weightssum, mass
            REAL(KIND=real_kind) :: x(np*np), c(np*np)
            REAL(KIND=real_kind) :: al_neg(np*np), al_pos(np*np), howmuch
            REAL(KIND=real_kind) :: tol_limiter = 1e-15
            INTEGER, parameter :: maxiter = 5
            DO k = 1 , nlev
                weights(:,k) = sphweights(:) * dpmass(:,k)
                ptens(:,k) = ptens(:,k) / dpmass(:,k)
            END DO 
            DO k = 1 , nlev
                c = weights(:,k)
                x = ptens(:,k)
                mass = sum(c*x)
                ! relax constraints to ensure limiter has a solution:
                ! This is only needed if runnign with the SSP CFL>1 or
                ! due to roundoff errors
                IF ((mass / sum(c)) < minp(k)) THEN
                    minp(k) = mass / sum(c)
                END IF 
                IF ((mass / sum(c)) > maxp(k)) THEN
                    maxp(k) = mass / sum(c)
                END IF 
                addmass = 0.0d0
                pos_counter = 0
                neg_counter = 0
                ! apply constraints, compute change in mass caused by constraints
                DO k1 = 1 , np*np
                    IF (( x(k1) >= maxp(k) )) THEN
                        addmass = addmass + (x(k1) - maxp(k)) * c(k1)
                        x(k1) = maxp(k)
                        whois_pos(k1) = -1
                        ELSE
                        pos_counter = pos_counter+1
                        whois_pos(pos_counter) = k1
                    END IF 
                    IF (( x(k1) <= minp(k) )) THEN
                        addmass = addmass - (minp(k) - x(k1)) * c(k1)
                        x(k1) = minp(k)
                        whois_neg(k1) = -1
                        ELSE
                        neg_counter = neg_counter+1
                        whois_neg(neg_counter) = k1
                    END IF 
                END DO 
                ! iterate to find field that satifies constraints and is l2-norm closest to original
                weightssum = 0.0d0
                IF (addmass > 0) THEN
                    DO i2 = 1 , maxiter
                        weightssum = 0.0
                        DO k1 = 1 , pos_counter
                            i1 = whois_pos(k1)
                            weightssum = weightssum + c(i1)
                            al_pos(i1) = maxp(k) - x(i1)
                        END DO 
                        IF (( pos_counter > 0 ) .and. ( addmass > tol_limiter * abs(mass) )) THEN
                            DO k1 = 1 , pos_counter
                                i1 = whois_pos(k1)
                                howmuch = addmass / weightssum
                                IF (howmuch > al_pos(i1)) THEN
                                    howmuch = al_pos(i1)
                                    whois_pos(k1) = -1
                                END IF 
                                addmass = addmass - howmuch * c(i1)
                                weightssum = weightssum - c(i1)
                                x(i1) = x(i1) + howmuch
                            END DO 
                            !now sort whois_pos and get a new number for pos_counter
                            !here neg_counter and whois_neg serve as temp vars
                            neg_counter = pos_counter
                            whois_neg = whois_pos
                            whois_pos = -1
                            pos_counter = 0
                            DO k1 = 1 , neg_counter
                                IF (whois_neg(k1) .ne. -1) THEN
                                    pos_counter = pos_counter+1
                                    whois_pos(pos_counter) = whois_neg(k1)
                                END IF 
                            END DO 
                            ELSE
                            EXIT
                        END IF 
                    END DO 
                    ELSE
                    DO i2 = 1 , maxiter
                        weightssum = 0.0
                        DO k1 = 1 , neg_counter
                            i1 = whois_neg(k1)
                            weightssum = weightssum + c(i1)
                            al_neg(i1) = x(i1) - minp(k)
                        END DO 
                        IF (( neg_counter > 0 ) .and. ( (-addmass) > tol_limiter * abs(mass) )) THEN
                            DO k1 = 1 , neg_counter
                                i1 = whois_neg(k1)
                                howmuch = -addmass / weightssum
                                IF (howmuch > al_neg(i1)) THEN
                                    howmuch = al_neg(i1)
                                    whois_neg(k1) = -1
                                END IF 
                                addmass = addmass + howmuch * c(i1)
                                weightssum = weightssum - c(i1)
                                x(i1) = x(i1) - howmuch
                            END DO 
                            !now sort whois_pos and get a new number for pos_counter
                            !here pos_counter and whois_pos serve as temp vars
                            pos_counter = neg_counter
                            whois_pos = whois_neg
                            whois_neg = -1
                            neg_counter = 0
                            DO k1 = 1 , pos_counter
                                IF (whois_pos(k1) .ne. -1) THEN
                                    neg_counter = neg_counter+1
                                    whois_neg(neg_counter) = whois_pos(k1)
                                END IF 
                            END DO 
                            ELSE
                            EXIT
                        END IF 
                    END DO 
                END IF 
                ptens(:,k) = x
            END DO 
            DO k = 1 , nlev
                ptens(:,k) = ptens(:,k) * dpmass(:,k)
            END DO 
        END SUBROUTINE limiter_optim_iter_full
        !-----------------------------------------------------------------------------
        !-----------------------------------------------------------------------------

        !-----------------------------------------------------------------------------
        !-----------------------------------------------------------------------------

        !-----------------------------------------------------------------------------
        !-----------------------------------------------------------------------------


    END MODULE prim_advection_mod
