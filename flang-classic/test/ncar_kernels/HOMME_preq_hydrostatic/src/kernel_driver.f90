
! KGEN-generated Fortran source file
!
! Filename    : kernel_driver.f90
! Generated at: 2015-04-12 19:37:49
! KGEN version: 0.4.9


PROGRAM kernel_driver
    USE prim_advance_mod, ONLY : compute_and_apply_rhs
    USE element_mod, ONLY: element_t
    USE physconst, ONLY : kgen_read_externs_physconst
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
    USE element_mod, ONLY : kgen_read_mod9 => kgen_read
    USE element_mod, ONLY : kgen_verify_mod9 => kgen_verify

    IMPLICIT NONE

    INTEGER :: kgen_mpi_rank
    CHARACTER(LEN=16) ::kgen_mpi_rank_conv
    INTEGER, DIMENSION(1), PARAMETER :: kgen_mpi_rank_at = (/ 0 /)
    INTEGER :: kgen_ierr, kgen_unit
    INTEGER :: kgen_repeat_counter
    INTEGER :: kgen_counter
    CHARACTER(LEN=16) :: kgen_counter_conv
    INTEGER, DIMENSION(1), PARAMETER :: kgen_counter_at = (/ 1 /)
    CHARACTER(LEN=1024) :: kgen_filepath
    TYPE(element_t), target, allocatable :: elem(:)

    DO kgen_repeat_counter = 0, 0
        kgen_counter = kgen_counter_at(mod(kgen_repeat_counter, 1)+1)
        WRITE( kgen_counter_conv, * ) kgen_counter
        kgen_mpi_rank = kgen_mpi_rank_at(mod(kgen_repeat_counter, 1)+1)
        WRITE( kgen_mpi_rank_conv, * ) kgen_mpi_rank
        kgen_filepath = "../data/preq_hydrostatic." // trim(adjustl(kgen_counter_conv)) // "." // trim(adjustl(kgen_mpi_rank_conv))
        kgen_unit = kgen_get_newunit()
        OPEN (UNIT=kgen_unit, FILE=kgen_filepath, STATUS="OLD", ACCESS="STREAM", FORM="UNFORMATTED", ACTION="READ", IOSTAT=kgen_ierr, CONVERT="BIG_ENDIAN")
        WRITE (*,*)
        IF ( kgen_ierr /= 0 ) THEN
            CALL kgen_error_stop( "FILE OPEN ERROR: " // trim(adjustl(kgen_filepath)) )
        END IF
        WRITE (*,*)
        WRITE (*,*) "** Verification against '" // trim(adjustl(kgen_filepath)) // "' **"

            CALL kgen_read_externs_physconst(kgen_unit)

            ! driver variables
            CALL kgen_read_element_t_dim1(elem, kgen_unit)

            call compute_and_apply_rhs(elem, kgen_unit)

            CLOSE (UNIT=kgen_unit)
        END DO
    CONTAINS

        ! write subroutines
            SUBROUTINE kgen_read_element_t_dim1(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                TYPE(element_t), INTENT(OUT), ALLOCATABLE, DIMENSION(:) :: var
                LOGICAL :: is_true
                INTEGER :: idx1
                INTEGER, DIMENSION(2,1) :: kgen_bound

                READ(UNIT = kgen_unit) is_true

                IF ( is_true ) THEN
                    READ(UNIT = kgen_unit) kgen_bound(1, 1)
                    READ(UNIT = kgen_unit) kgen_bound(2, 1)
                    ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1))
                    DO idx1=kgen_bound(1,1), kgen_bound(2, 1)
                    IF ( PRESENT(printvar) ) THEN
                            CALL kgen_read_mod9(var(idx1), kgen_unit, printvar=printvar)
                    ELSE
                            CALL kgen_read_mod9(var(idx1), kgen_unit)
                    END IF
                    END DO
                END IF
            END SUBROUTINE kgen_read_element_t_dim1

        FUNCTION kgen_get_newunit() RESULT(new_unit)
           INTEGER, PARAMETER :: UNIT_MIN=100, UNIT_MAX=1000000
           LOGICAL :: is_opened
           INTEGER :: nunit, new_unit, counter
        
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


    END PROGRAM kernel_driver
