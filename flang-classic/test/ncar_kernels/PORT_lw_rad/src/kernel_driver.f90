
! KGEN-generated Fortran source file
!
! Filename    : kernel_driver.f90
! Generated at: 2015-07-06 23:28:43
! KGEN version: 0.4.13


PROGRAM kernel_driver
    USE radlw, ONLY : rad_rrtmg_lw
    USE rrtmg_state, ONLY: rrtmg_state_t
    USE rrlw_cld, ONLY : kgen_read_externs_rrlw_cld
    USE rrlw_vsn, ONLY : kgen_read_externs_rrlw_vsn
    USE rrlw_kg13, ONLY : kgen_read_externs_rrlw_kg13
    USE rrlw_kg10, ONLY : kgen_read_externs_rrlw_kg10
    USE rrlw_kg11, ONLY : kgen_read_externs_rrlw_kg11
    USE rrlw_kg16, ONLY : kgen_read_externs_rrlw_kg16
    USE rrlw_kg14, ONLY : kgen_read_externs_rrlw_kg14
    USE rrlw_kg15, ONLY : kgen_read_externs_rrlw_kg15
    USE rrlw_ref, ONLY : kgen_read_externs_rrlw_ref
    USE rrlw_kg12, ONLY : kgen_read_externs_rrlw_kg12
    USE rrlw_wvn, ONLY : kgen_read_externs_rrlw_wvn
    USE rrlw_kg01, ONLY : kgen_read_externs_rrlw_kg01
    USE rrlw_tbl, ONLY : kgen_read_externs_rrlw_tbl
    USE rrlw_kg03, ONLY : kgen_read_externs_rrlw_kg03
    USE rrlw_kg02, ONLY : kgen_read_externs_rrlw_kg02
    USE rrlw_kg05, ONLY : kgen_read_externs_rrlw_kg05
    USE rrlw_kg04, ONLY : kgen_read_externs_rrlw_kg04
    USE rrlw_kg07, ONLY : kgen_read_externs_rrlw_kg07
    USE rrlw_kg06, ONLY : kgen_read_externs_rrlw_kg06
    USE rrlw_kg09, ONLY : kgen_read_externs_rrlw_kg09
    USE rrlw_kg08, ONLY : kgen_read_externs_rrlw_kg08
    USE rrlw_con, ONLY : kgen_read_externs_rrlw_con
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
    USE rrtmg_state, ONLY : kgen_read_mod31 => kgen_read
    USE rrtmg_state, ONLY : kgen_verify_mod31 => kgen_verify

    IMPLICIT NONE

    INTEGER :: kgen_ierr, kgen_unit
    INTEGER :: kgen_repeat_counter
    INTEGER :: kgen_counter
    CHARACTER(LEN=16) :: kgen_counter_conv
    INTEGER, DIMENSION(3), PARAMETER :: kgen_counter_at = (/ 10, 15, 5 /)
    CHARACTER(LEN=1024) :: kgen_filepath
    INTEGER :: lchnk
    INTEGER :: ncol
    TYPE(rrtmg_state_t) :: r_state
    INTEGER :: rrtmg_levs

    DO kgen_repeat_counter = 0, 2
        kgen_counter = kgen_counter_at(mod(kgen_repeat_counter, 3)+1)
        WRITE( kgen_counter_conv, * ) kgen_counter
        kgen_filepath = "../data/rrtmg_lw." // trim(adjustl(kgen_counter_conv))
        kgen_unit = kgen_get_newunit()
        OPEN (UNIT=kgen_unit, FILE=kgen_filepath, STATUS="OLD", ACCESS="STREAM", FORM="UNFORMATTED", ACTION="READ", IOSTAT=kgen_ierr, CONVERT="BIG_ENDIAN")
        WRITE (*,*)
        IF ( kgen_ierr /= 0 ) THEN
            CALL kgen_error_stop( "FILE OPEN ERROR: " // trim(adjustl(kgen_filepath)) )
        END IF
        WRITE (*,*)
        WRITE (*,*) "** Verification against '" // trim(adjustl(kgen_filepath)) // "' **"

        CALL kgen_read_externs_rrlw_cld(kgen_unit)
        CALL kgen_read_externs_rrlw_vsn(kgen_unit)
        CALL kgen_read_externs_rrlw_kg13(kgen_unit)
        CALL kgen_read_externs_rrlw_kg10(kgen_unit)
        CALL kgen_read_externs_rrlw_kg11(kgen_unit)
        CALL kgen_read_externs_rrlw_kg16(kgen_unit)
        CALL kgen_read_externs_rrlw_kg14(kgen_unit)
        CALL kgen_read_externs_rrlw_kg15(kgen_unit)
        CALL kgen_read_externs_rrlw_ref(kgen_unit)
        CALL kgen_read_externs_rrlw_kg12(kgen_unit)
        CALL kgen_read_externs_rrlw_wvn(kgen_unit)
        CALL kgen_read_externs_rrlw_kg01(kgen_unit)
        CALL kgen_read_externs_rrlw_tbl(kgen_unit)
        CALL kgen_read_externs_rrlw_kg03(kgen_unit)
        CALL kgen_read_externs_rrlw_kg02(kgen_unit)
        CALL kgen_read_externs_rrlw_kg05(kgen_unit)
        CALL kgen_read_externs_rrlw_kg04(kgen_unit)
        CALL kgen_read_externs_rrlw_kg07(kgen_unit)
        CALL kgen_read_externs_rrlw_kg06(kgen_unit)
        CALL kgen_read_externs_rrlw_kg09(kgen_unit)
        CALL kgen_read_externs_rrlw_kg08(kgen_unit)
        CALL kgen_read_externs_rrlw_con(kgen_unit)

        ! driver variables
        READ(UNIT=kgen_unit) lchnk
        READ(UNIT=kgen_unit) ncol
        READ(UNIT=kgen_unit) rrtmg_levs
        CALL kgen_read_mod31(r_state, kgen_unit)

        call rad_rrtmg_lw(lchnk, ncol, rrtmg_levs, r_state, kgen_unit)

            CLOSE (UNIT=kgen_unit)
        END DO
    CONTAINS

        ! write subroutines
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
