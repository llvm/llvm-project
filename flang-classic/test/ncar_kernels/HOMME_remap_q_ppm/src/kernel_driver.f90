
! KGEN-generated Fortran source file
!
! Filename    : kernel_driver.f90
! Generated at: 2015-02-24 15:34:48
! KGEN version: 0.4.4


PROGRAM kernel_driver
    USE vertremap_mod, only : remap1
    USE kinds, ONLY: real_kind
    USE dimensions_mod, ONLY: nlev
    USE perf_mod, only : read_externs_perf_mod
    USE control_mod, only : read_externs_control_mod

    IMPLICIT NONE

    ! read interface
    interface kgen_read_var
        procedure read_var_real_real_kind_dim4
        procedure read_var_real_real_kind_dim3
    end interface kgen_read_var


    INTEGER :: kgen_mpi_rank
    CHARACTER(LEN=16) ::kgen_mpi_rank_conv
    INTEGER, DIMENSION(1), PARAMETER :: kgen_mpi_rank_at = (/ 0 /)
    INTEGER :: kgen_ierr, kgen_unit
    INTEGER :: kgen_repeat_counter
    INTEGER :: kgen_counter
    CHARACTER(LEN=16) :: kgen_counter_conv
    INTEGER, DIMENSION(3), PARAMETER :: kgen_counter_at = (/ 1, 10, 20 /)
    CHARACTER(LEN=1024) :: kgen_filepath
    INTEGER :: nx
    INTEGER :: qsize
    REAL(KIND=real_kind), allocatable :: qdp(:,:,:,:)
    REAL(KIND=real_kind), allocatable :: dp2(:,:,:)
    REAL(KIND=real_kind), allocatable :: dp1(:,:,:)

    DO kgen_repeat_counter = 0, 2
        kgen_counter = kgen_counter_at(mod(kgen_repeat_counter, 3)+1)
        WRITE( kgen_counter_conv, * ) kgen_counter
        kgen_mpi_rank = kgen_mpi_rank_at(mod(kgen_repeat_counter, 1)+1)
        WRITE( kgen_mpi_rank_conv, * ) kgen_mpi_rank
        kgen_filepath = "../data/remap_q_ppm." // trim(adjustl(kgen_counter_conv)) // "." // trim(adjustl(kgen_mpi_rank_conv))
        kgen_unit = kgen_get_newunit()
        OPEN (UNIT=kgen_unit, FILE=kgen_filepath, STATUS="OLD", ACCESS="STREAM", FORM="UNFORMATTED", ACTION="READ", IOSTAT=kgen_ierr, CONVERT="BIG_ENDIAN")
        WRITE (*,*)
        IF ( kgen_ierr /= 0 ) THEN
            CALL kgen_error_stop( "FILE OPEN ERROR: " // trim(adjustl(kgen_filepath)) )
        END IF
        WRITE (*,*)
        WRITE (*,*) "************ Verification against '" // trim(adjustl(kgen_filepath)) // "' ************"

            call read_externs_perf_mod(kgen_unit)
            call read_externs_control_mod(kgen_unit)

            ! driver variables
                READ(UNIT=kgen_unit) nx
                READ(UNIT=kgen_unit) qsize
                call kgen_read_var(qdp, kgen_unit)
                call kgen_read_var(dp1, kgen_unit)
                call kgen_read_var(dp2, kgen_unit)
            call remap1(nx, qsize, qdp, dp1, dp2, kgen_unit)

            CLOSE (UNIT=kgen_unit)
        END DO
    CONTAINS

        ! read subroutines
        subroutine read_var_real_real_kind_dim4(var, kgen_unit)
            integer, intent(in) :: kgen_unit
            real(kind=real_kind), intent(out), dimension(:,:,:,:), allocatable :: var
            integer, dimension(2,4) :: kgen_bound
            logical is_save
            
            READ(UNIT = kgen_unit) is_save
            if ( is_save ) then
                READ(UNIT = kgen_unit) kgen_bound(1, 1)
                READ(UNIT = kgen_unit) kgen_bound(2, 1)
                READ(UNIT = kgen_unit) kgen_bound(1, 2)
                READ(UNIT = kgen_unit) kgen_bound(2, 2)
                READ(UNIT = kgen_unit) kgen_bound(1, 3)
                READ(UNIT = kgen_unit) kgen_bound(2, 3)
                READ(UNIT = kgen_unit) kgen_bound(1, 4)
                READ(UNIT = kgen_unit) kgen_bound(2, 4)
                ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1, kgen_bound(2, 3) - kgen_bound(1, 3) + 1, kgen_bound(2, 4) - kgen_bound(1, 4) + 1))
                READ(UNIT = kgen_unit) var
            end if
        end subroutine
        subroutine read_var_real_real_kind_dim3(var, kgen_unit)
            integer, intent(in) :: kgen_unit
            real(kind=real_kind), intent(out), dimension(:,:,:), allocatable :: var
            integer, dimension(2,3) :: kgen_bound
            logical is_save
            
            READ(UNIT = kgen_unit) is_save
            if ( is_save ) then
                READ(UNIT = kgen_unit) kgen_bound(1, 1)
                READ(UNIT = kgen_unit) kgen_bound(2, 1)
                READ(UNIT = kgen_unit) kgen_bound(1, 2)
                READ(UNIT = kgen_unit) kgen_bound(2, 2)
                READ(UNIT = kgen_unit) kgen_bound(1, 3)
                READ(UNIT = kgen_unit) kgen_bound(2, 3)
                ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1, kgen_bound(2, 3) - kgen_bound(1, 3) + 1))
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
