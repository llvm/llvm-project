
! KGEN-generated Fortran source file
!
! Filename    : kernel_driver.f90
! Generated at: 2015-07-31 20:52:25
! KGEN version: 0.4.13


PROGRAM kernel_driver
    USE rrtmg_sw_spcvmc, ONLY : spcvmc_sw
    USE shr_kind_mod, ONLY: r8 => shr_kind_r8
    USE rrsw_tbl, ONLY : kgen_read_externs_rrsw_tbl
    USE rrsw_vsn, ONLY : kgen_read_externs_rrsw_vsn
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check

    IMPLICIT NONE

    INTEGER :: kgen_mpi_rank
    CHARACTER(LEN=16) ::kgen_mpi_rank_conv
    INTEGER, DIMENSION(3), PARAMETER :: kgen_mpi_rank_at = (/ 1, 4, 8 /)
    INTEGER :: kgen_ierr, kgen_unit
    INTEGER :: kgen_repeat_counter
    INTEGER :: kgen_counter
    CHARACTER(LEN=16) :: kgen_counter_conv
    INTEGER, DIMENSION(2), PARAMETER :: kgen_counter_at = (/ 1,  5 /)
    CHARACTER(LEN=1024) :: kgen_filepath
    INTEGER :: nlayers
    INTEGER :: ncol
    REAL(KIND=r8), allocatable :: prmu0(:)

    DO kgen_repeat_counter = 0, 5
        kgen_counter = kgen_counter_at(mod(kgen_repeat_counter, 2)+1)
        WRITE( kgen_counter_conv, * ) kgen_counter
        kgen_mpi_rank = kgen_mpi_rank_at(mod(kgen_repeat_counter, 3)+1)
        WRITE( kgen_mpi_rank_conv, * ) kgen_mpi_rank
        kgen_filepath = "../data/reftra_sw." // trim(adjustl(kgen_counter_conv)) // "." // trim(adjustl(kgen_mpi_rank_conv))
        kgen_unit = kgen_get_newunit()
        OPEN (UNIT=kgen_unit, FILE=kgen_filepath, STATUS="OLD", ACCESS="STREAM", FORM="UNFORMATTED", ACTION="READ", IOSTAT=kgen_ierr, CONVERT="BIG_ENDIAN")
        WRITE (*,*)
        IF ( kgen_ierr /= 0 ) THEN
            CALL kgen_error_stop( "FILE OPEN ERROR: " // trim(adjustl(kgen_filepath)) )
        END IF
        WRITE (*,*)
        WRITE (*,*) "** Verification against '" // trim(adjustl(kgen_filepath)) // "' **"

            CALL kgen_read_externs_rrsw_tbl(kgen_unit)
            CALL kgen_read_externs_rrsw_vsn(kgen_unit)

            ! driver variables
            READ(UNIT=kgen_unit) nlayers
            READ(UNIT=kgen_unit) ncol
            CALL kgen_read_real_r8_dim1(prmu0, kgen_unit)

            call spcvmc_sw(nlayers, ncol, prmu0, kgen_unit)

            CLOSE (UNIT=kgen_unit)
        END DO
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
