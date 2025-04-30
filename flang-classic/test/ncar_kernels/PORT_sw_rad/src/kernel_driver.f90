
! KGEN-generated Fortran source file
!
! Filename    : kernel_driver.f90
! Generated at: 2015-07-07 00:48:23
! KGEN version: 0.4.13


PROGRAM kernel_driver
    USE radiation, ONLY : radiation_tend
    USE shr_kind_mod, ONLY: r8 => shr_kind_r8
    USE camsrfexch, ONLY: cam_in_t
    USE physics_types, ONLY: physics_state
    USE camsrfexch, ONLY: cam_out_t
    USE rrsw_ref, ONLY : kgen_read_externs_rrsw_ref
    USE rrsw_tbl, ONLY : kgen_read_externs_rrsw_tbl
    USE rrsw_kg19, ONLY : kgen_read_externs_rrsw_kg19
    USE rrsw_kg18, ONLY : kgen_read_externs_rrsw_kg18
    USE rrsw_kg17, ONLY : kgen_read_externs_rrsw_kg17
    USE rrsw_kg16, ONLY : kgen_read_externs_rrsw_kg16
    USE rrsw_cld, ONLY : kgen_read_externs_rrsw_cld
    USE rrsw_kg29, ONLY : kgen_read_externs_rrsw_kg29
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
    USE scammod, ONLY : kgen_read_externs_scammod
    USE rrsw_kg28, ONLY : kgen_read_externs_rrsw_kg28
    USE radsw, ONLY : kgen_read_externs_radsw
    USE rrtmg_state, ONLY : kgen_read_externs_rrtmg_state
    USE rrsw_con, ONLY : kgen_read_externs_rrsw_con
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
    USE physics_types, ONLY : kgen_read_mod42 => kgen_read
    USE physics_types, ONLY : kgen_verify_mod42 => kgen_verify
    USE camsrfexch, ONLY : kgen_read_mod43 => kgen_read
    USE camsrfexch, ONLY : kgen_verify_mod43 => kgen_verify

    IMPLICIT NONE

    INTEGER :: kgen_ierr, kgen_unit
    INTEGER :: kgen_repeat_counter
    INTEGER :: kgen_counter
    CHARACTER(LEN=16) :: kgen_counter_conv
    INTEGER, DIMENSION(2), PARAMETER :: kgen_counter_at = (/ 1, 2 /)
    CHARACTER(LEN=1024) :: kgen_filepath
    REAL(KIND=r8), allocatable :: fsnt(:)
    TYPE(cam_in_t) :: cam_in
    REAL(KIND=r8), allocatable :: fsns(:)
    TYPE(physics_state), target :: state
    REAL(KIND=r8), allocatable :: fsds(:)
    TYPE(cam_out_t) :: cam_out

    DO kgen_repeat_counter = 0, 1
        kgen_counter = kgen_counter_at(mod(kgen_repeat_counter, 2)+1)
        WRITE( kgen_counter_conv, * ) kgen_counter
        kgen_filepath = "../data/rad_rrtmg_sw." // trim(adjustl(kgen_counter_conv))
        kgen_unit = kgen_get_newunit()
        OPEN (UNIT=kgen_unit, FILE=kgen_filepath, STATUS="OLD", ACCESS="STREAM", FORM="UNFORMATTED", ACTION="READ", IOSTAT=kgen_ierr, CONVERT="BIG_ENDIAN")
        WRITE (*,*)
        IF ( kgen_ierr /= 0 ) THEN
            CALL kgen_error_stop( "FILE OPEN ERROR: " // trim(adjustl(kgen_filepath)) )
        END IF
        WRITE (*,*)
        WRITE (*,*) "** Verification against '" // trim(adjustl(kgen_filepath)) // "' **"

        CALL kgen_read_externs_rrsw_ref(kgen_unit)
        CALL kgen_read_externs_rrsw_tbl(kgen_unit)
        CALL kgen_read_externs_rrsw_kg19(kgen_unit)
        CALL kgen_read_externs_rrsw_kg18(kgen_unit)
        CALL kgen_read_externs_rrsw_kg17(kgen_unit)
        CALL kgen_read_externs_rrsw_kg16(kgen_unit)
        CALL kgen_read_externs_rrsw_cld(kgen_unit)
        CALL kgen_read_externs_rrsw_kg29(kgen_unit)
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
        CALL kgen_read_externs_scammod(kgen_unit)
        CALL kgen_read_externs_rrsw_kg28(kgen_unit)
        CALL kgen_read_externs_radsw(kgen_unit)
        CALL kgen_read_externs_rrtmg_state(kgen_unit)
        CALL kgen_read_externs_rrsw_con(kgen_unit)

        ! driver variables
        CALL kgen_read_real_r8_dim1(fsns, kgen_unit)
        CALL kgen_read_real_r8_dim1(fsnt, kgen_unit)
        CALL kgen_read_real_r8_dim1(fsds, kgen_unit)
        CALL kgen_read_mod42(state, kgen_unit)
        CALL kgen_read_mod43(cam_out, kgen_unit)
        CALL kgen_read_mod43(cam_in, kgen_unit)

        call radiation_tend(fsns, fsnt, fsds, state, cam_out, cam_in, kgen_unit)

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
