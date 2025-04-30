
! KGEN-generated Fortran source file
!
! Filename    : rrtmg_state.F90
! Generated at: 2015-07-07 00:48:23
! KGEN version: 0.4.13



    MODULE rrtmg_state
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        IMPLICIT NONE
        PRIVATE
        PUBLIC rrtmg_state_t
        PUBLIC num_rrtmg_levs
        TYPE rrtmg_state_t
            REAL(KIND=r8), allocatable :: h2ovmr(:,:) ! h2o volume mixing ratio
            REAL(KIND=r8), allocatable :: o3vmr(:,:) ! o3 volume mixing ratio
            REAL(KIND=r8), allocatable :: co2vmr(:,:) ! co2 volume mixing ratio
            REAL(KIND=r8), allocatable :: ch4vmr(:,:) ! ch4 volume mixing ratio
            REAL(KIND=r8), allocatable :: o2vmr(:,:) ! o2  volume mixing ratio
            REAL(KIND=r8), allocatable :: n2ovmr(:,:) ! n2o volume mixing ratio
            REAL(KIND=r8), allocatable :: cfc11vmr(:,:) ! cfc11 volume mixing ratio
            REAL(KIND=r8), allocatable :: cfc12vmr(:,:) ! cfc12 volume mixing ratio
            REAL(KIND=r8), allocatable :: cfc22vmr(:,:) ! cfc22 volume mixing ratio
            REAL(KIND=r8), allocatable :: ccl4vmr(:,:) ! ccl4 volume mixing ratio
            REAL(KIND=r8), allocatable :: pmidmb(:,:) ! Level pressure (hPa)
            REAL(KIND=r8), allocatable :: pintmb(:,:) ! Model interface pressure (hPa)
            REAL(KIND=r8), allocatable :: tlay(:,:) ! mid point temperature
            REAL(KIND=r8), allocatable :: tlev(:,:) ! interface temperature
        END TYPE rrtmg_state_t
        INTEGER :: num_rrtmg_levs ! number of pressure levels greate than 1.e-4_r8 mbar
        ! Molecular weight of dry air / water vapor
        ! Molecular weight of dry air / carbon dioxide
        ! Molecular weight of dry air / ozone
        ! Molecular weight of dry air / methane
        ! Molecular weight of dry air / nitrous oxide
        ! Molecular weight of dry air / oxygen
        ! Molecular weight of dry air / CFC11
        ! Molecular weight of dry air / CFC12
            PUBLIC kgen_read_externs_rrtmg_state

        ! read interface
        PUBLIC kgen_read
        INTERFACE kgen_read
            MODULE PROCEDURE kgen_read_rrtmg_state_t
        END INTERFACE kgen_read

        PUBLIC kgen_verify
        INTERFACE kgen_verify
            MODULE PROCEDURE kgen_verify_rrtmg_state_t
        END INTERFACE kgen_verify

        CONTAINS

        ! write subroutines
            SUBROUTINE kgen_read_real_r8_dim2_alloc(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                real(KIND=r8), INTENT(OUT), ALLOCATABLE, DIMENSION(:,:) :: var
                LOGICAL :: is_true
                INTEGER :: idx1,idx2
                INTEGER, DIMENSION(2,2) :: kgen_bound

                READ(UNIT = kgen_unit) is_true

                IF ( is_true ) THEN
                    READ(UNIT = kgen_unit) kgen_bound(1, 1)
                    READ(UNIT = kgen_unit) kgen_bound(2, 1)
                    READ(UNIT = kgen_unit) kgen_bound(1, 2)
                    READ(UNIT = kgen_unit) kgen_bound(2, 2)
                    ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
                    READ(UNIT = kgen_unit) var
                    IF ( PRESENT(printvar) ) THEN
                        PRINT *, "** " // printvar // " **", var
                    END IF
                END IF
            END SUBROUTINE kgen_read_real_r8_dim2_alloc


        ! module extern variables

        SUBROUTINE kgen_read_externs_rrtmg_state(kgen_unit)
            INTEGER, INTENT(IN) :: kgen_unit
            READ(UNIT=kgen_unit) num_rrtmg_levs
        END SUBROUTINE kgen_read_externs_rrtmg_state

        SUBROUTINE kgen_read_rrtmg_state_t(var, kgen_unit, printvar)
            INTEGER, INTENT(IN) :: kgen_unit
            CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
            TYPE(rrtmg_state_t), INTENT(out) :: var
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%h2ovmr, kgen_unit, printvar=printvar//"%h2ovmr")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%h2ovmr, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%o3vmr, kgen_unit, printvar=printvar//"%o3vmr")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%o3vmr, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%co2vmr, kgen_unit, printvar=printvar//"%co2vmr")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%co2vmr, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%ch4vmr, kgen_unit, printvar=printvar//"%ch4vmr")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%ch4vmr, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%o2vmr, kgen_unit, printvar=printvar//"%o2vmr")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%o2vmr, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%n2ovmr, kgen_unit, printvar=printvar//"%n2ovmr")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%n2ovmr, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%cfc11vmr, kgen_unit, printvar=printvar//"%cfc11vmr")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%cfc11vmr, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%cfc12vmr, kgen_unit, printvar=printvar//"%cfc12vmr")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%cfc12vmr, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%cfc22vmr, kgen_unit, printvar=printvar//"%cfc22vmr")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%cfc22vmr, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%ccl4vmr, kgen_unit, printvar=printvar//"%ccl4vmr")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%ccl4vmr, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%pmidmb, kgen_unit, printvar=printvar//"%pmidmb")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%pmidmb, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%pintmb, kgen_unit, printvar=printvar//"%pintmb")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%pintmb, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%tlay, kgen_unit, printvar=printvar//"%tlay")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%tlay, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_real_r8_dim2_alloc(var%tlev, kgen_unit, printvar=printvar//"%tlev")
            ELSE
                CALL kgen_read_real_r8_dim2_alloc(var%tlev, kgen_unit)
            END IF
        END SUBROUTINE
        SUBROUTINE kgen_verify_rrtmg_state_t(varname, check_status, var, ref_var)
            CHARACTER(*), INTENT(IN) :: varname
            TYPE(check_t), INTENT(INOUT) :: check_status
            TYPE(check_t) :: dtype_check_status
            TYPE(rrtmg_state_t), INTENT(IN) :: var, ref_var

            check_status%numTotal = check_status%numTotal + 1
            CALL kgen_init_check(dtype_check_status)
            CALL kgen_verify_real_r8_dim2_alloc("h2ovmr", dtype_check_status, var%h2ovmr, ref_var%h2ovmr)
            CALL kgen_verify_real_r8_dim2_alloc("o3vmr", dtype_check_status, var%o3vmr, ref_var%o3vmr)
            CALL kgen_verify_real_r8_dim2_alloc("co2vmr", dtype_check_status, var%co2vmr, ref_var%co2vmr)
            CALL kgen_verify_real_r8_dim2_alloc("ch4vmr", dtype_check_status, var%ch4vmr, ref_var%ch4vmr)
            CALL kgen_verify_real_r8_dim2_alloc("o2vmr", dtype_check_status, var%o2vmr, ref_var%o2vmr)
            CALL kgen_verify_real_r8_dim2_alloc("n2ovmr", dtype_check_status, var%n2ovmr, ref_var%n2ovmr)
            CALL kgen_verify_real_r8_dim2_alloc("cfc11vmr", dtype_check_status, var%cfc11vmr, ref_var%cfc11vmr)
            CALL kgen_verify_real_r8_dim2_alloc("cfc12vmr", dtype_check_status, var%cfc12vmr, ref_var%cfc12vmr)
            CALL kgen_verify_real_r8_dim2_alloc("cfc22vmr", dtype_check_status, var%cfc22vmr, ref_var%cfc22vmr)
            CALL kgen_verify_real_r8_dim2_alloc("ccl4vmr", dtype_check_status, var%ccl4vmr, ref_var%ccl4vmr)
            CALL kgen_verify_real_r8_dim2_alloc("pmidmb", dtype_check_status, var%pmidmb, ref_var%pmidmb)
            CALL kgen_verify_real_r8_dim2_alloc("pintmb", dtype_check_status, var%pintmb, ref_var%pintmb)
            CALL kgen_verify_real_r8_dim2_alloc("tlay", dtype_check_status, var%tlay, ref_var%tlay)
            CALL kgen_verify_real_r8_dim2_alloc("tlev", dtype_check_status, var%tlev, ref_var%tlev)
            IF ( dtype_check_status%numTotal == dtype_check_status%numIdentical ) THEN
                check_status%numIdentical = check_status%numIdentical + 1
            ELSE IF ( dtype_check_status%numFatal > 0 ) THEN
                check_status%numFatal = check_status%numFatal + 1
            ELSE IF ( dtype_check_status%numWarning > 0 ) THEN
                check_status%numWarning = check_status%numWarning + 1
            END IF
        END SUBROUTINE
            SUBROUTINE kgen_verify_real_r8_dim2_alloc( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                real(KIND=r8), intent(in), DIMENSION(:,:), ALLOCATABLE :: var, ref_var
                real(KIND=r8) :: nrmsdiff, rmsdiff
                real(KIND=r8), allocatable, DIMENSION(:,:) :: temp, temp2
                integer :: n
                IF ( ALLOCATED(var) ) THEN
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
                    allocate(temp(SIZE(var,dim=1),SIZE(var,dim=2)))
                    allocate(temp2(SIZE(var,dim=1),SIZE(var,dim=2)))
                
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
            END SUBROUTINE kgen_verify_real_r8_dim2_alloc

        !--------------------------------------------------------------------------------
        ! sets the number of model levels RRTMG operates
        !--------------------------------------------------------------------------------

        !--------------------------------------------------------------------------------
        ! creates (alloacates) an rrtmg_state object
        !--------------------------------------------------------------------------------

        !--------------------------------------------------------------------------------
        ! updates the concentration fields
        !--------------------------------------------------------------------------------

        !--------------------------------------------------------------------------------
        ! de-allocates an rrtmg_state object
        !--------------------------------------------------------------------------------

    END MODULE rrtmg_state
