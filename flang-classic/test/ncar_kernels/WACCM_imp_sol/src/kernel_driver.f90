
! KGEN-generated Fortran source file
!
! Filename    : kernel_driver.f90
! Generated at: 2015-05-13 11:02:21
! KGEN version: 0.4.10


PROGRAM kernel_driver
    USE mo_gas_phase_chemdr, ONLY : gas_phase_chemdr
    USE shr_kind_mod, ONLY: r8 => shr_kind_r8
    USE mo_imp_sol, ONLY : kgen_read_externs_mo_imp_sol
    USE chem_mods, ONLY : kgen_read_externs_chem_mods
    USE mo_tracname, ONLY : kgen_read_externs_mo_tracname
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check

    IMPLICIT NONE

    INTEGER :: kgen_mpi_rank
    CHARACTER(LEN=16) ::kgen_mpi_rank_conv
    INTEGER, DIMENSION(3), PARAMETER :: kgen_mpi_rank_at = (/ 0, 100, 300 /)
    INTEGER :: kgen_ierr, kgen_unit
    INTEGER :: kgen_repeat_counter
    INTEGER :: kgen_counter
    CHARACTER(LEN=16) :: kgen_counter_conv
    INTEGER, DIMENSION(3), PARAMETER :: kgen_counter_at = (/ 10, 100, 50 /)
    CHARACTER(LEN=1024) :: kgen_filepath
    INTEGER :: lchnk
    INTEGER :: ncol
    REAL(KIND=r8) :: delt

    DO kgen_repeat_counter = 0, 8
        kgen_counter = kgen_counter_at(mod(kgen_repeat_counter, 3)+1)
        WRITE( kgen_counter_conv, * ) kgen_counter
        kgen_mpi_rank = kgen_mpi_rank_at(mod(kgen_repeat_counter, 3)+1)
        WRITE( kgen_mpi_rank_conv, * ) kgen_mpi_rank
        kgen_filepath = "../data/imp_sol." // trim(adjustl(kgen_counter_conv)) // "." // trim(adjustl(kgen_mpi_rank_conv))
        kgen_unit = kgen_get_newunit()
        OPEN (UNIT=kgen_unit, FILE=kgen_filepath, STATUS="OLD", ACCESS="STREAM", FORM="UNFORMATTED", ACTION="READ", IOSTAT=kgen_ierr, CONVERT="BIG_ENDIAN")
        WRITE (*,*)
        IF ( kgen_ierr /= 0 ) THEN
            CALL kgen_error_stop( "FILE OPEN ERROR: " // trim(adjustl(kgen_filepath)) )
        END IF
        WRITE (*,*)
        WRITE (*,*) "** Verification against '" // trim(adjustl(kgen_filepath)) // "' **"

            CALL kgen_read_externs_mo_imp_sol(kgen_unit)
            CALL kgen_read_externs_chem_mods(kgen_unit)
            CALL kgen_read_externs_mo_tracname(kgen_unit)

            ! driver variables
            READ(UNIT=kgen_unit) lchnk
            READ(UNIT=kgen_unit) ncol
            READ(UNIT=kgen_unit) delt

            call gas_phase_chemdr(lchnk, ncol, delt, kgen_unit)

            CLOSE (UNIT=kgen_unit)
        END DO
    CONTAINS

        ! write subroutines
        ! No subroutines
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
