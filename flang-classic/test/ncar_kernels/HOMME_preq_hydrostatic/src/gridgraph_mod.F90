
! KGEN-generated Fortran source file
!
! Filename    : gridgraph_mod.F90
! Generated at: 2015-04-12 19:37:50
! KGEN version: 0.4.9



    MODULE gridgraph_mod
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        !-------------------------
        !-------------------------------
        !-------------------------
        !-----
        IMPLICIT NONE
        PRIVATE
        INTEGER, public, parameter :: num_neighbors=8 ! for north, south, east, west, neast, nwest, seast, swest
        TYPE, public :: gridvertex_t
            INTEGER, pointer :: nbrs(:) => null() ! The numbers of the neighbor elements
            INTEGER, pointer :: nbrs_face(:) => null() ! The cube face number of the neighbor element (nbrs array)
            INTEGER, pointer :: nbrs_wgt(:) => null() ! The weights for edges defined by nbrs array
            INTEGER, pointer :: nbrs_wgt_ghost(:) => null() ! The weights for edges defined by nbrs array
            INTEGER :: nbrs_ptr(num_neighbors + 1) !index into the nbrs array for each neighbor direction
            INTEGER :: face_number ! which face of the cube this vertex is on
            INTEGER :: number ! element number
            INTEGER :: processor_number ! processor number
            INTEGER :: spacecurve ! index in Space-Filling curve
        END TYPE gridvertex_t
        ! ==========================================
        ! Public Interfaces
        ! ==========================================


        ! read interface
        PUBLIC kgen_read
        INTERFACE kgen_read
            MODULE PROCEDURE kgen_read_gridvertex_t
        END INTERFACE kgen_read

        PUBLIC kgen_verify
        INTERFACE kgen_verify
            MODULE PROCEDURE kgen_verify_gridvertex_t
        END INTERFACE kgen_verify

        CONTAINS

        ! write subroutines
            SUBROUTINE kgen_read_integer_4_dim1_ptr(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                integer(KIND=4), INTENT(OUT), POINTER, DIMENSION(:) :: var
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
            END SUBROUTINE kgen_read_integer_4_dim1_ptr

        ! No module extern variables
        SUBROUTINE kgen_read_gridvertex_t(var, kgen_unit, printvar)
            INTEGER, INTENT(IN) :: kgen_unit
            CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
            TYPE(gridvertex_t), INTENT(out) :: var
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_integer_4_dim1_ptr(var%nbrs, kgen_unit, printvar=printvar//"%nbrs")
            ELSE
                CALL kgen_read_integer_4_dim1_ptr(var%nbrs, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_integer_4_dim1_ptr(var%nbrs_face, kgen_unit, printvar=printvar//"%nbrs_face")
            ELSE
                CALL kgen_read_integer_4_dim1_ptr(var%nbrs_face, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_integer_4_dim1_ptr(var%nbrs_wgt, kgen_unit, printvar=printvar//"%nbrs_wgt")
            ELSE
                CALL kgen_read_integer_4_dim1_ptr(var%nbrs_wgt, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_integer_4_dim1_ptr(var%nbrs_wgt_ghost, kgen_unit, printvar=printvar//"%nbrs_wgt_ghost")
            ELSE
                CALL kgen_read_integer_4_dim1_ptr(var%nbrs_wgt_ghost, kgen_unit)
            END IF
            READ(UNIT=kgen_unit) var%nbrs_ptr
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%nbrs_ptr **", var%nbrs_ptr
            END IF
            READ(UNIT=kgen_unit) var%face_number
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%face_number **", var%face_number
            END IF
            READ(UNIT=kgen_unit) var%number
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%number **", var%number
            END IF
            READ(UNIT=kgen_unit) var%processor_number
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%processor_number **", var%processor_number
            END IF
            READ(UNIT=kgen_unit) var%spacecurve
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%spacecurve **", var%spacecurve
            END IF
        END SUBROUTINE
        SUBROUTINE kgen_verify_gridvertex_t(varname, check_status, var, ref_var)
            CHARACTER(*), INTENT(IN) :: varname
            TYPE(check_t), INTENT(INOUT) :: check_status
            TYPE(check_t) :: dtype_check_status
            TYPE(gridvertex_t), INTENT(IN) :: var, ref_var

            check_status%numTotal = check_status%numTotal + 1
            CALL kgen_init_check(dtype_check_status)
            CALL kgen_verify_integer_4_dim1_ptr("nbrs", dtype_check_status, var%nbrs, ref_var%nbrs)
            CALL kgen_verify_integer_4_dim1_ptr("nbrs_face", dtype_check_status, var%nbrs_face, ref_var%nbrs_face)
            CALL kgen_verify_integer_4_dim1_ptr("nbrs_wgt", dtype_check_status, var%nbrs_wgt, ref_var%nbrs_wgt)
            CALL kgen_verify_integer_4_dim1_ptr("nbrs_wgt_ghost", dtype_check_status, var%nbrs_wgt_ghost, ref_var%nbrs_wgt_ghost)
            CALL kgen_verify_integer_4_dim1("nbrs_ptr", dtype_check_status, var%nbrs_ptr, ref_var%nbrs_ptr)
            CALL kgen_verify_integer("face_number", dtype_check_status, var%face_number, ref_var%face_number)
            CALL kgen_verify_integer("number", dtype_check_status, var%number, ref_var%number)
            CALL kgen_verify_integer("processor_number", dtype_check_status, var%processor_number, ref_var%processor_number)
            CALL kgen_verify_integer("spacecurve", dtype_check_status, var%spacecurve, ref_var%spacecurve)
            IF ( dtype_check_status%numTotal == dtype_check_status%numIdentical ) THEN
                check_status%numIdentical = check_status%numIdentical + 1
            ELSE IF ( dtype_check_status%numFatal > 0 ) THEN
                check_status%numFatal = check_status%numFatal + 1
            ELSE IF ( dtype_check_status%numWarning > 0 ) THEN
                check_status%numWarning = check_status%numWarning + 1
            END IF
        END SUBROUTINE
            SUBROUTINE kgen_verify_integer_4_dim1_ptr( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                integer, intent(in), DIMENSION(:), POINTER :: var, ref_var
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
                    if(check_status%verboseLevel > 0) then
                        WRITE(*,*)
                        WRITE(*,*) trim(adjustl(varname)), " is NOT IDENTICAL."
                        WRITE(*,*) count( var /= ref_var), " of ", size( var ), " elements are different."
                    end if
                
                    check_status%numFatal = check_status%numFatal+1
                END IF
                END IF
            END SUBROUTINE kgen_verify_integer_4_dim1_ptr

            SUBROUTINE kgen_verify_integer_4_dim1( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                integer, intent(in), DIMENSION(:) :: var, ref_var
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
                    if(check_status%verboseLevel > 0) then
                        WRITE(*,*)
                        WRITE(*,*) trim(adjustl(varname)), " is NOT IDENTICAL."
                        WRITE(*,*) count( var /= ref_var), " of ", size( var ), " elements are different."
                    end if
                
                    check_status%numFatal = check_status%numFatal+1
                END IF
            END SUBROUTINE kgen_verify_integer_4_dim1

            SUBROUTINE kgen_verify_integer( varname, check_status, var, ref_var)
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
                        end if
                    end if
                    check_status%numFatal = check_status%numFatal + 1
                END IF
            END SUBROUTINE kgen_verify_integer

        !======================================================================

        !======================================================================

        !======================================================================
        ! =====================================
        ! copy edge:
        ! copy device for overloading = sign.
        ! =====================================

        !======================================================================

        !======================================================================

        !======================================================================

        !======================================================================
        !===========================
        ! search edge list for match
        !===========================

        !======================================================================

        !======================================================================

        !======================================================================

        !======================================================================

        !======================================================================

        !======================================================================
        ! ==========================================
        ! set_GridVertex_neighbors:
        !
        ! Set global element number for element elem
        ! ==========================================

        !======================================================================

        !======================================================================

        !======================================================================

        !======================================================================
    END MODULE gridgraph_mod
