
! KGEN-generated Fortran source file
!
! Filename    : kernel_driver.f90
! Generated at: 2015-02-19 15:30:29
! KGEN version: 0.4.4


PROGRAM kernel_driver
    USE mo_psrad_interface, only : psrad_interface
    USE mo_kind, ONLY: wp
    USE mo_psrad_interface, only : read_externs_mo_psrad_interface
    USE mo_radiation_parameters, only : read_externs_mo_radiation_parameters
    USE rrlw_kg12, only : read_externs_rrlw_kg12
    USE rrlw_kg13, only : read_externs_rrlw_kg13
    USE rrlw_planck, only : read_externs_rrlw_planck
    USE rrlw_kg11, only : read_externs_rrlw_kg11
    USE rrlw_kg16, only : read_externs_rrlw_kg16
    USE rrlw_kg14, only : read_externs_rrlw_kg14
    USE rrlw_kg15, only : read_externs_rrlw_kg15
    USE rrlw_kg10, only : read_externs_rrlw_kg10
    USE rrlw_kg01, only : read_externs_rrlw_kg01
    USE rrlw_kg03, only : read_externs_rrlw_kg03
    USE rrlw_kg02, only : read_externs_rrlw_kg02
    USE rrlw_kg05, only : read_externs_rrlw_kg05
    USE rrlw_kg04, only : read_externs_rrlw_kg04
    USE rrlw_kg07, only : read_externs_rrlw_kg07
    USE rrlw_kg06, only : read_externs_rrlw_kg06
    USE rrlw_kg09, only : read_externs_rrlw_kg09
    USE rrlw_kg08, only : read_externs_rrlw_kg08
    USE mo_random_numbers, only : read_externs_mo_random_numbers

    IMPLICIT NONE

    ! read interface
    !interface kgen_read_var
    !    procedure read_var_real_wp_dim1
    !end interface kgen_read_var


    INTEGER :: kgen_ierr, kgen_unit
    INTEGER :: kgen_repeat_counter
    INTEGER :: kgen_counter
    CHARACTER(LEN=16) :: kgen_counter_conv
    INTEGER, DIMENSION(3), PARAMETER :: kgen_counter_at = (/ 1, 10, 50 /)
    CHARACTER(LEN=1024) :: kgen_filepath
    INTEGER :: nb_sw
    INTEGER :: klev
    REAL(KIND=wp), allocatable :: tk_sfc(:)
    INTEGER :: kproma
    INTEGER :: kbdim
    INTEGER :: ktrac

    DO kgen_repeat_counter = 0, 2
        kgen_counter = kgen_counter_at(mod(kgen_repeat_counter, 3)+1)
        WRITE( kgen_counter_conv, * ) kgen_counter
        kgen_filepath = "../data/lrtm." // trim(adjustl(kgen_counter_conv))
        kgen_unit = kgen_get_newunit()
        OPEN (UNIT=kgen_unit, FILE=kgen_filepath, STATUS="OLD", ACCESS="STREAM", FORM="UNFORMATTED", ACTION="READ", IOSTAT=kgen_ierr, CONVERT="BIG_ENDIAN")
        WRITE (*,*)
        IF ( kgen_ierr /= 0 ) THEN
            CALL kgen_error_stop( "FILE OPEN ERROR: " // trim(adjustl(kgen_filepath)) )
        END IF
        WRITE (*,*)
        WRITE (*,*) "************ Verification against '" // trim(adjustl(kgen_filepath)) // "' ************"

        call read_externs_mo_psrad_interface(kgen_unit)
        call read_externs_mo_radiation_parameters(kgen_unit)
        call read_externs_rrlw_kg12(kgen_unit)
        call read_externs_rrlw_kg13(kgen_unit)
        call read_externs_rrlw_planck(kgen_unit)
        call read_externs_rrlw_kg11(kgen_unit)
        call read_externs_rrlw_kg16(kgen_unit)
        call read_externs_rrlw_kg14(kgen_unit)
        call read_externs_rrlw_kg15(kgen_unit)
        call read_externs_rrlw_kg10(kgen_unit)
        call read_externs_rrlw_kg01(kgen_unit)
        call read_externs_rrlw_kg03(kgen_unit)
        call read_externs_rrlw_kg02(kgen_unit)
        call read_externs_rrlw_kg05(kgen_unit)
        call read_externs_rrlw_kg04(kgen_unit)
        call read_externs_rrlw_kg07(kgen_unit)
        call read_externs_rrlw_kg06(kgen_unit)
        call read_externs_rrlw_kg09(kgen_unit)
        call read_externs_rrlw_kg08(kgen_unit)
        call read_externs_mo_random_numbers(kgen_unit)

        ! driver variables
            READ(UNIT=kgen_unit) kbdim
            READ(UNIT=kgen_unit) klev
            READ(UNIT=kgen_unit) nb_sw
            READ(UNIT=kgen_unit) kproma
            READ(UNIT=kgen_unit) ktrac
            !call kgen_read_var(tk_sfc, kgen_unit)
            call read_var_real_wp_dim1(tk_sfc, kgen_unit)
        call psrad_interface(kbdim, klev, nb_sw, kproma, ktrac, tk_sfc, kgen_unit)

            CLOSE (UNIT=kgen_unit)
        END DO
    CONTAINS

        ! read subroutines
        subroutine read_var_real_wp_dim1(var, kgen_unit)
            integer, intent(in) :: kgen_unit
            real(kind=wp), intent(out), dimension(:), allocatable :: var
            integer, dimension(2,1) :: kgen_bound
            logical is_save
            
            READ(UNIT = kgen_unit) is_save
            if ( is_save ) then
                READ(UNIT = kgen_unit) kgen_bound(1, 1)
                READ(UNIT = kgen_unit) kgen_bound(2, 1)
                ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1))
                READ(UNIT = kgen_unit) var
            end if
        end subroutine
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
