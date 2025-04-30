
! KGEN-generated Fortran source file
!
! Filename    : viscosity_mod.F90
! Generated at: 2015-04-12 19:17:34
! KGEN version: 0.4.9



    MODULE viscosity_mod
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
    USE element_mod, ONLY : kgen_read_mod3 => kgen_read
    USE element_mod, ONLY : kgen_verify_mod3 => kgen_verify
    USE derivative_mod, ONLY : kgen_read_mod2 => kgen_read
    USE derivative_mod, ONLY : kgen_verify_mod2 => kgen_verify
        !
        !  This module should be renamed "global_deriv_mod.F90"
        !
        !  It is a collection of derivative operators that must be applied to the field
        !  over the sphere (as opposed to derivative operators that can be applied element
        !  by element)
        !
        !
        USE kinds, ONLY: real_kind
        USE dimensions_mod, ONLY: np
        USE dimensions_mod, ONLY: nlev
        USE element_mod, ONLY: element_t
        USE derivative_mod, ONLY: vlaplace_sphere_wk
        USE derivative_mod, ONLY: derivative_t
        IMPLICIT NONE
        PUBLIC biharmonic_wk_dp3d
        !
        ! compute vorticity/divergence and then project to make continious
        ! high-level routines uses only for I/O


        ! for older versions of sweq which carry
        ! velocity around in contra-coordinates
        CONTAINS

        ! write subroutines
        ! No subroutines
        ! No module extern variables


        SUBROUTINE biharmonic_wk_dp3d(elem, nt, nets, nete, vtens, deriv, kgen_unit)
                USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            ! compute weak biharmonic operator
            !    input:  h,v (stored in elem()%, in lat-lon coordinates
            !    output: ptens,vtens  overwritten with weak biharmonic of h,v (output in lat-lon coordinates)
            !
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            integer, intent(in) :: kgen_unit
            INTEGER*8 :: kgen_intvar, start_clock, stop_clock, rate_clock
            TYPE(check_t):: check_status
            REAL(KIND=kgen_dp) :: tolerance
            TYPE(element_t), intent(inout), target :: elem(:)
            INTEGER :: nt
            INTEGER :: nets
            INTEGER :: nete
            REAL(KIND=real_kind), dimension(np,np,2,nlev,nets:nete) :: vtens
            REAL(KIND=real_kind) :: ref_vtens(np,np,2,nlev,nets:nete)
            TYPE(derivative_t), intent(in) :: deriv
            ! local
            INTEGER :: ie
            INTEGER :: k
            REAL(KIND=real_kind) :: nu_ratio1
            REAL(KIND=real_kind) :: ref_nu_ratio1
            LOGICAL :: var_coef1
            !if tensor hyperviscosity with tensor V is used, then biharmonic operator is (\grad\cdot V\grad) (\grad \cdot \grad)
            !so tensor is only used on second call to laplace_sphere_wk
            ! note: there is a scaling bug in the treatment of nu_div
            ! nu_ratio is applied twice, once in each laplace operator
            ! so in reality:   nu_div_actual = (nu_div/nu)**2 nu
            ! We should fix this, but it requires adjusting all 1 defaults
                    tolerance = 1.E-14
                    CALL kgen_init_check(check_status, tolerance)
                    READ(UNIT=kgen_unit) ie
                    READ(UNIT=kgen_unit) k
                    READ(UNIT=kgen_unit) nu_ratio1
                    READ(UNIT=kgen_unit) var_coef1

                    READ(UNIT=kgen_unit) ref_vtens
                    READ(UNIT=kgen_unit) ref_nu_ratio1


                    ! call to kernel
                    vtens(:, :, :, k, ie) = vlaplace_sphere_wk(elem(ie) % state % v(:, :, :, k, nt), deriv, elem(ie), var_coef = var_coef1, nu_ratio = nu_ratio1)
                    ! kernel verification for output variables
                    CALL kgen_verify_real_real_kind_dim5( "vtens", check_status, vtens, ref_vtens)
                    CALL kgen_verify_real_real_kind( "nu_ratio1", check_status, nu_ratio1, ref_nu_ratio1)
                    CALL kgen_print_check("vlaplace_sphere_wk", check_status)
                    CALL system_clock(start_clock, rate_clock)
                    DO kgen_intvar=1,10
                        vtens(:, :, :, k, ie) = vlaplace_sphere_wk(elem(ie) % state % v(:, :, :, k, nt), deriv, elem(ie), var_coef = var_coef1, nu_ratio = nu_ratio1)
                    END DO
                    CALL system_clock(stop_clock, rate_clock)
                    WRITE(*,*)
                    PRINT *, "Elapsed time (sec): ", (stop_clock - start_clock)/REAL(rate_clock*10)
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        CONTAINS

        ! write subroutines
            SUBROUTINE kgen_read_real_real_kind_dim5(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                real(KIND=real_kind), INTENT(OUT), ALLOCATABLE, DIMENSION(:,:,:,:,:) :: var
                LOGICAL :: is_true
                INTEGER :: idx1,idx2,idx3,idx4,idx5
                INTEGER, DIMENSION(2,5) :: kgen_bound

                READ(UNIT = kgen_unit) is_true

                IF ( is_true ) THEN
                    READ(UNIT = kgen_unit) kgen_bound(1, 1)
                    READ(UNIT = kgen_unit) kgen_bound(2, 1)
                    READ(UNIT = kgen_unit) kgen_bound(1, 2)
                    READ(UNIT = kgen_unit) kgen_bound(2, 2)
                    READ(UNIT = kgen_unit) kgen_bound(1, 3)
                    READ(UNIT = kgen_unit) kgen_bound(2, 3)
                    READ(UNIT = kgen_unit) kgen_bound(1, 4)
                    READ(UNIT = kgen_unit) kgen_bound(2, 4)
                    READ(UNIT = kgen_unit) kgen_bound(1, 5)
                    READ(UNIT = kgen_unit) kgen_bound(2, 5)
                    ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1, kgen_bound(2, 3) - kgen_bound(1, 3) + 1, kgen_bound(2, 4) - kgen_bound(1, 4) + 1, kgen_bound(2, 5) - kgen_bound(1, 5) + 1))
                    READ(UNIT = kgen_unit) var
                    IF ( PRESENT(printvar) ) THEN
                        PRINT *, "** " // printvar // " **", var
                    END IF
                END IF
            END SUBROUTINE kgen_read_real_real_kind_dim5


        ! verify subroutines
            SUBROUTINE kgen_verify_real_real_kind_dim5( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                real(KIND=real_kind), intent(in), DIMENSION(:,:,:,:,:) :: var, ref_var
                real(KIND=real_kind) :: nrmsdiff, rmsdiff
                real(KIND=real_kind), allocatable, DIMENSION(:,:,:,:,:) :: temp, temp2
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
                    allocate(temp(SIZE(var,dim=1),SIZE(var,dim=2),SIZE(var,dim=3),SIZE(var,dim=4),SIZE(var,dim=5)))
                    allocate(temp2(SIZE(var,dim=1),SIZE(var,dim=2),SIZE(var,dim=3),SIZE(var,dim=4),SIZE(var,dim=5)))
                
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
            END SUBROUTINE kgen_verify_real_real_kind_dim5

            SUBROUTINE kgen_verify_real_real_kind( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                real(KIND=real_kind), intent(in) :: var, ref_var
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
            END SUBROUTINE kgen_verify_real_real_kind

        END SUBROUTINE 













    END MODULE 
