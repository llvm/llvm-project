
! KGEN-generated Fortran source file
!
! Filename    : kernel_driver.f90
! Generated at: 2015-07-31 20:35:44
! KGEN version: 0.4.13


PROGRAM kernel_driver
    USE rrtmg_sw_rad, ONLY : rrtmg_sw
    USE rrsw_tbl, ONLY : kgen_read_externs_rrsw_tbl
    USE rrsw_kg19, ONLY : kgen_read_externs_rrsw_kg19
    USE rrsw_kg18, ONLY : kgen_read_externs_rrsw_kg18
    USE rrsw_kg17, ONLY : kgen_read_externs_rrsw_kg17
    USE rrsw_kg16, ONLY : kgen_read_externs_rrsw_kg16
    USE rrsw_wvn, ONLY : kgen_read_externs_rrsw_wvn
    USE rrsw_vsn, ONLY : kgen_read_externs_rrsw_vsn
    USE rrsw_kg24, ONLY : kgen_read_externs_rrsw_kg24
    USE rrsw_kg25, ONLY : kgen_read_externs_rrsw_kg25
    USE rrsw_kg26, ONLY : kgen_read_externs_rrsw_kg26
    USE rrsw_kg27, ONLY : kgen_read_externs_rrsw_kg27
    USE rrsw_kg20, ONLY : kgen_read_externs_rrsw_kg20
    USE rrsw_kg21, ONLY : kgen_read_externs_rrsw_kg21
    USE rrsw_kg22, ONLY : kgen_read_externs_rrsw_kg22
    USE rrsw_kg23, ONLY : kgen_read_externs_rrsw_kg23
    USE rrsw_kg28, ONLY : kgen_read_externs_rrsw_kg28
    USE rrsw_kg29, ONLY : kgen_read_externs_rrsw_kg29
    USE rrsw_con, ONLY : kgen_read_externs_rrsw_con
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check

    IMPLICIT NONE

    INTEGER :: kgen_mpi_rank
    CHARACTER(LEN=16) ::kgen_mpi_rank_conv
    INTEGER, DIMENSION(3), PARAMETER :: kgen_mpi_rank_at = (/ 1, 4, 8 /)
    INTEGER :: kgen_ierr, kgen_unit
    INTEGER :: kgen_repeat_counter
    INTEGER :: kgen_counter
    CHARACTER(LEN=16) :: kgen_counter_conv
    INTEGER, DIMENSION(3), PARAMETER :: kgen_counter_at = (/ 1, 10, 5 /)
    CHARACTER(LEN=1024) :: kgen_filepath
    INTEGER :: lchnk
    INTEGER :: ncol
    INTEGER :: nlay

    DO kgen_repeat_counter = 0, 8
        kgen_counter = kgen_counter_at(mod(kgen_repeat_counter, 3)+1)
        WRITE( kgen_counter_conv, * ) kgen_counter
        kgen_mpi_rank = kgen_mpi_rank_at(mod(kgen_repeat_counter, 3)+1)
        WRITE( kgen_mpi_rank_conv, * ) kgen_mpi_rank
        kgen_filepath = "../data/spcvmc_sw." // trim(adjustl(kgen_counter_conv)) // "." // trim(adjustl(kgen_mpi_rank_conv))
        kgen_unit = kgen_get_newunit()
        OPEN (UNIT=kgen_unit, FILE=kgen_filepath, STATUS="OLD", ACCESS="STREAM", FORM="UNFORMATTED", ACTION="READ", IOSTAT=kgen_ierr, CONVERT="BIG_ENDIAN")
        WRITE (*,*)
        IF ( kgen_ierr /= 0 ) THEN
            CALL kgen_error_stop( "FILE OPEN ERROR: " // trim(adjustl(kgen_filepath)) )
        END IF
        WRITE (*,*)
        WRITE (*,*) "** Verification against '" // trim(adjustl(kgen_filepath)) // "' **"

            CALL kgen_read_externs_rrsw_tbl(kgen_unit)
            CALL kgen_read_externs_rrsw_kg19(kgen_unit)
            CALL kgen_read_externs_rrsw_kg18(kgen_unit)
            CALL kgen_read_externs_rrsw_kg17(kgen_unit)
            CALL kgen_read_externs_rrsw_kg16(kgen_unit)
            CALL kgen_read_externs_rrsw_wvn(kgen_unit)
            CALL kgen_read_externs_rrsw_vsn(kgen_unit)
            CALL kgen_read_externs_rrsw_kg24(kgen_unit)
            CALL kgen_read_externs_rrsw_kg25(kgen_unit)
            CALL kgen_read_externs_rrsw_kg26(kgen_unit)
            CALL kgen_read_externs_rrsw_kg27(kgen_unit)
            CALL kgen_read_externs_rrsw_kg20(kgen_unit)
            CALL kgen_read_externs_rrsw_kg21(kgen_unit)
            CALL kgen_read_externs_rrsw_kg22(kgen_unit)
            CALL kgen_read_externs_rrsw_kg23(kgen_unit)
            CALL kgen_read_externs_rrsw_kg28(kgen_unit)
            CALL kgen_read_externs_rrsw_kg29(kgen_unit)
            CALL kgen_read_externs_rrsw_con(kgen_unit)

            ! driver variables
            READ(UNIT=kgen_unit) lchnk
            READ(UNIT=kgen_unit) ncol
            READ(UNIT=kgen_unit) nlay

            call rrtmg_sw(lchnk, ncol, nlay, kgen_unit)

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
