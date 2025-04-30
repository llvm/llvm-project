
! KGEN-generated Fortran source file
!
! Filename    : mo_imp_sol.F90
! Generated at: 2015-07-15 10:35:30
! KGEN version: 0.4.13



    MODULE mo_imp_sol
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        IMPLICIT NONE
        PRIVATE
        PUBLIC imp_sol
        !-----------------------------------------------------------------------
        ! Newton-Raphson iteration limits
        !-----------------------------------------------------------------------
        ! for xnox ozone chemistry diagnostics
        CONTAINS

        ! write subroutines
        ! No subroutines
        ! No module extern variables


        SUBROUTINE imp_sol(kgen_unit)
                USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
            !-----------------------------------------------------------------------
            ! ... imp_sol advances the volumetric mixing ratio
            ! forward one time step via the fully implicit euler scheme.
            ! this source is meant for small l1 cache machines such as
            ! the intel pentium and itanium cpus
            !-----------------------------------------------------------------------
            USE chem_mods, ONLY: nzcnt
            USE mo_lu_factor, ONLY: lu_fac
            IMPLICIT NONE
            !-----------------------------------------------------------------------
            ! ... dummy args
            !-----------------------------------------------------------------------
            integer, intent(in) :: kgen_unit
            INTEGER*8 :: kgen_intvar, start_clock, stop_clock, rate_clock,maxiter=1000
            TYPE(check_t):: check_status
            REAL(KIND=kgen_dp) :: tolerance
            ! columns in chunck
            ! chunk id
            ! time step (s)
            ! rxt rates (1/cm^3/s)
            ! external in-situ forcing (1/cm^3/s)
            ! washout rates (1/s)
            ! species mixing ratios (vmr)
            ! chemistry troposphere boundary (index)
            !-----------------------------------------------------------------------
            ! ... local variables
            !-----------------------------------------------------------------------
            REAL(KIND=r8) :: sys_jac(max(1,nzcnt))
            REAL(KIND=r8) :: ref_sys_jac(max(1,nzcnt))
            !-----------------------------------------------------------------------
            ! ... class independent forcing
            !-----------------------------------------------------------------------
                                tolerance = 1.E-13
                                CALL kgen_init_check(check_status, tolerance)
                                READ(UNIT=kgen_unit) sys_jac

                                READ(UNIT=kgen_unit) ref_sys_jac


                                ! call to kernel
                   call lu_fac( sys_jac )
                                ! kernel verification for output variables
                                CALL kgen_verify_real_r8_dim1( "sys_jac", check_status, sys_jac, ref_sys_jac)
                                CALL kgen_print_check("lu_fac", check_status)
                                CALL system_clock(start_clock, rate_clock)
                                DO kgen_intvar=1,maxiter
                                    CALL lu_fac(sys_jac)
                                END DO
                                CALL system_clock(stop_clock, rate_clock)
                                WRITE(*,*)
                                PRINT *, "Elapsed time (sec): ", (stop_clock - start_clock)/REAL(rate_clock*10)
                                PRINT *, "Elapsed time per lu_fac call (usec): ", (stop_clock - start_clock)*1e6/REAL(rate_clock*maxiter)
            !
            !
        CONTAINS

        ! write subroutines
            SUBROUTINE kgen_read_real_r8_dim1(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                real(KIND=r8), INTENT(OUT), ALLOCATABLE, DIMENSION(:) :: var
                LOGICAL :: is_true
                INTEGER :: idx1
                INTEGER, DIMENSION(2,1) :: kgen_bound

                READ(UNIT = kgen_unit) is_true

                IF ( is_true ) THEN
                    READ(UNIT = kgen_unit) kgen_bound(1, 1)
                    READ(UNIT = kgen_unit) kgen_bound(2, 1)
                    ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1))
                    READ(UNIT = kgen_unit) var
                    IF ( PRESENT(printvar) ) THEN
                        PRINT *, "** " // printvar // " **", var
                    END IF
                END IF
            END SUBROUTINE kgen_read_real_r8_dim1


        ! verify subroutines
            SUBROUTINE kgen_verify_real_r8_dim1( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                real(KIND=r8), intent(in), DIMENSION(:) :: var, ref_var
                real(KIND=r8) :: nrmsdiff, rmsdiff
                real(KIND=r8), allocatable, DIMENSION(:) :: temp, temp2
                integer :: n
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
                    allocate(temp(SIZE(var,dim=1)))
                    allocate(temp2(SIZE(var,dim=1)))
                
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
            END SUBROUTINE kgen_verify_real_r8_dim1

        END SUBROUTINE imp_sol
    END MODULE mo_imp_sol
