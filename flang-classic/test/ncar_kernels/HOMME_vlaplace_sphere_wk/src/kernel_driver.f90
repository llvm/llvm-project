
! KGEN-generated Fortran source file
!
! Filename    : kernel_driver.f90
! Generated at: 2015-04-12 19:17:34
! KGEN version: 0.4.9


PROGRAM kernel_driver
    USE viscosity_mod, ONLY : biharmonic_wk_dp3d
    USE derivative_mod, ONLY: derivative_t
    USE element_mod, ONLY: element_t
    USE dimensions_mod, ONLY: np
    USE kinds, ONLY: real_kind
    USE dimensions_mod, ONLY: nlev
    USE control_mod, ONLY : kgen_read_externs_control_mod
    USE physconst, ONLY : kgen_read_externs_physconst
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
    USE element_mod, ONLY : kgen_read_mod3 => kgen_read
    USE element_mod, ONLY : kgen_verify_mod3 => kgen_verify
    USE derivative_mod, ONLY : kgen_read_mod2 => kgen_read
    USE derivative_mod, ONLY : kgen_verify_mod2 => kgen_verify

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
    INTEGER :: nets
    INTEGER :: nt
    TYPE(derivative_t) :: deriv
    INTEGER :: nete
    TYPE(element_t), target, allocatable :: elem(:)
    REAL(KIND=real_kind), allocatable :: vtens(:,:,:,:,:)

    DO kgen_repeat_counter = 0, 0
        kgen_counter = kgen_counter_at(mod(kgen_repeat_counter, 1)+1)
        WRITE( kgen_counter_conv, * ) kgen_counter
        kgen_mpi_rank = kgen_mpi_rank_at(mod(kgen_repeat_counter, 1)+1)
        WRITE( kgen_mpi_rank_conv, * ) kgen_mpi_rank
        kgen_filepath = "../data/vlaplace_sphere_wk." // trim(adjustl(kgen_counter_conv)) // "." // trim(adjustl(kgen_mpi_rank_conv))
        kgen_unit = kgen_get_newunit()
        OPEN (UNIT=kgen_unit, FILE=kgen_filepath, STATUS="OLD", ACCESS="STREAM", FORM="UNFORMATTED", ACTION="READ", IOSTAT=kgen_ierr, CONVERT="BIG_ENDIAN")
        WRITE (*,*)
        IF ( kgen_ierr /= 0 ) THEN
            CALL kgen_error_stop( "FILE OPEN ERROR: " // trim(adjustl(kgen_filepath)) )
        END IF
        WRITE (*,*)
        WRITE (*,*) "** Verification against '" // trim(adjustl(kgen_filepath)) // "' **"

            CALL kgen_read_externs_control_mod(kgen_unit)
            CALL kgen_read_externs_physconst(kgen_unit)

            ! driver variables
            READ(UNIT=kgen_unit) nt
            READ(UNIT=kgen_unit) nets
            READ(UNIT=kgen_unit) nete
            CALL kgen_read_real_real_kind_dim5(vtens, kgen_unit)
            CALL kgen_read_element_t_dim1(elem, kgen_unit)
            CALL kgen_read_mod2(deriv, kgen_unit)

            call biharmonic_wk_dp3d(elem, nt, nets, nete, vtens, deriv, kgen_unit)

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
                            CALL kgen_read_mod3(var(idx1), kgen_unit, printvar=printvar)
                    ELSE
                            CALL kgen_read_mod3(var(idx1), kgen_unit)
                    END IF
                    END DO
                END IF
            END SUBROUTINE kgen_read_element_t_dim1

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
