
! KGEN-generated Fortran source file
!
! Filename    : kernel_driver.f90
! Generated at: 2015-07-26 18:45:57
! KGEN version: 0.4.13


PROGRAM kernel_driver
    USE rrtmg_lw_rad, ONLY : rrtmg_lw
    USE shr_kind_mod, ONLY: r8 => shr_kind_r8
    USE rrlw_wvn, ONLY : kgen_read_externs_rrlw_wvn
    USE rrlw_con, ONLY : kgen_read_externs_rrlw_con
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
    REAL(KIND=r8), allocatable :: tauaer(:,:,:)
    REAL(KIND=r8), allocatable :: play(:,:)
    REAL(KIND=r8), allocatable :: ciwpmcl(:,:,:)
    REAL(KIND=r8), allocatable :: plev(:,:)
    INTEGER :: nlay
    REAL(KIND=r8), allocatable :: tlev(:,:)
    REAL(KIND=r8), allocatable :: tsfc(:)
    REAL(KIND=r8), allocatable :: o3vmr(:,:)
    REAL(KIND=r8), allocatable :: co2vmr(:,:)
    REAL(KIND=r8), allocatable :: ch4vmr(:,:)
    REAL(KIND=r8), allocatable :: o2vmr(:,:)
    REAL(KIND=r8), allocatable :: tlay(:,:)
    INTEGER :: ncol
    REAL(KIND=r8), allocatable :: cfc11vmr(:,:)
    REAL(KIND=r8), allocatable :: cfc12vmr(:,:)
    REAL(KIND=r8), allocatable :: cldfmcl(:,:,:)
    REAL(KIND=r8), allocatable :: n2ovmr(:,:)
    REAL(KIND=r8), allocatable :: cfc22vmr(:,:)
    REAL(KIND=r8), allocatable :: relqmcl(:,:)
    REAL(KIND=r8), allocatable :: ccl4vmr(:,:)
    REAL(KIND=r8), allocatable :: emis(:,:)
    REAL(KIND=r8), allocatable :: h2ovmr(:,:)
    INTEGER :: inflglw
    REAL(KIND=r8), allocatable :: reicmcl(:,:)
    INTEGER :: iceflglw
    INTEGER :: liqflglw
    REAL(KIND=r8), allocatable :: clwpmcl(:,:,:)
    INTEGER :: icld
    REAL(KIND=r8), allocatable :: taucmcl(:,:,:)

    DO kgen_repeat_counter = 0, 8
        kgen_counter = kgen_counter_at(mod(kgen_repeat_counter, 3)+1)
        WRITE( kgen_counter_conv, * ) kgen_counter
        kgen_mpi_rank = kgen_mpi_rank_at(mod(kgen_repeat_counter, 3)+1)
        WRITE( kgen_mpi_rank_conv, * ) kgen_mpi_rank
        kgen_filepath = "../data/inatm." // trim(adjustl(kgen_counter_conv)) // "." // trim(adjustl(kgen_mpi_rank_conv))
        kgen_unit = kgen_get_newunit()
        OPEN (UNIT=kgen_unit, FILE=kgen_filepath, STATUS="OLD", ACCESS="STREAM", FORM="UNFORMATTED", ACTION="READ", IOSTAT=kgen_ierr, CONVERT="BIG_ENDIAN")
        WRITE (*,*)
        IF ( kgen_ierr /= 0 ) THEN
            CALL kgen_error_stop( "FILE OPEN ERROR: " // trim(adjustl(kgen_filepath)) )
        END IF
        WRITE (*,*)
        WRITE (*,*) "** Verification against '" // trim(adjustl(kgen_filepath)) // "' **"

            CALL kgen_read_externs_rrlw_wvn(kgen_unit)
            CALL kgen_read_externs_rrlw_con(kgen_unit)

            ! driver variables
            READ(UNIT=kgen_unit) ncol
            READ(UNIT=kgen_unit) nlay
            READ(UNIT=kgen_unit) icld
            CALL kgen_read_real_r8_dim2(play, kgen_unit)
            CALL kgen_read_real_r8_dim2(plev, kgen_unit)
            CALL kgen_read_real_r8_dim2(tlay, kgen_unit)
            CALL kgen_read_real_r8_dim2(tlev, kgen_unit)
            CALL kgen_read_real_r8_dim1(tsfc, kgen_unit)
            CALL kgen_read_real_r8_dim2(h2ovmr, kgen_unit)
            CALL kgen_read_real_r8_dim2(o3vmr, kgen_unit)
            CALL kgen_read_real_r8_dim2(co2vmr, kgen_unit)
            CALL kgen_read_real_r8_dim2(ch4vmr, kgen_unit)
            CALL kgen_read_real_r8_dim2(o2vmr, kgen_unit)
            CALL kgen_read_real_r8_dim2(n2ovmr, kgen_unit)
            CALL kgen_read_real_r8_dim2(cfc11vmr, kgen_unit)
            CALL kgen_read_real_r8_dim2(cfc12vmr, kgen_unit)
            CALL kgen_read_real_r8_dim2(cfc22vmr, kgen_unit)
            CALL kgen_read_real_r8_dim2(ccl4vmr, kgen_unit)
            CALL kgen_read_real_r8_dim2(emis, kgen_unit)
            READ(UNIT=kgen_unit) inflglw
            READ(UNIT=kgen_unit) iceflglw
            READ(UNIT=kgen_unit) liqflglw
            CALL kgen_read_real_r8_dim3(cldfmcl, kgen_unit)
            CALL kgen_read_real_r8_dim3(ciwpmcl, kgen_unit)
            CALL kgen_read_real_r8_dim3(clwpmcl, kgen_unit)
            CALL kgen_read_real_r8_dim2(reicmcl, kgen_unit)
            CALL kgen_read_real_r8_dim2(relqmcl, kgen_unit)
            CALL kgen_read_real_r8_dim3(taucmcl, kgen_unit)
            CALL kgen_read_real_r8_dim3(tauaer, kgen_unit)

            call rrtmg_lw(ncol, nlay, icld, play, plev, tlay, tlev, tsfc, h2ovmr, o3vmr, co2vmr, ch4vmr, o2vmr, &
n2ovmr, cfc11vmr, cfc12vmr, cfc22vmr, ccl4vmr, emis, inflglw, iceflglw, liqflglw, cldfmcl, ciwpmcl, clwpmcl, &
reicmcl, relqmcl, taucmcl, tauaer, kgen_unit)

            CLOSE (UNIT=kgen_unit)
        END DO
    CONTAINS

        ! write subroutines
            SUBROUTINE kgen_read_real_r8_dim2(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                real(KIND=r8), INTENT(OUT), ALLOCATABLE, DIMENSION(:,:) :: var
                LOGICAL :: is_true
                INTEGER :: idx1,idx2
                INTEGER, DIMENSION(2,2) :: kgen_bound

                READ(UNIT = kgen_unit) is_true

                IF ( is_true ) THEN
                    READ(UNIT = kgen_unit) kgen_bound(1, 1)
                    READ(UNIT = kgen_unit) kgen_bound(2, 1)
                    READ(UNIT = kgen_unit) kgen_bound(1, 2)
                    READ(UNIT = kgen_unit) kgen_bound(2, 2)
                    ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
                    READ(UNIT = kgen_unit) var
                    IF ( PRESENT(printvar) ) THEN
                        PRINT *, "** " // printvar // " **", var
                    END IF
                END IF
            END SUBROUTINE kgen_read_real_r8_dim2

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

            SUBROUTINE kgen_read_real_r8_dim3(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                real(KIND=r8), INTENT(OUT), ALLOCATABLE, DIMENSION(:,:,:) :: var
                LOGICAL :: is_true
                INTEGER :: idx1,idx2,idx3
                INTEGER, DIMENSION(2,3) :: kgen_bound

                READ(UNIT = kgen_unit) is_true

                IF ( is_true ) THEN
                    READ(UNIT = kgen_unit) kgen_bound(1, 1)
                    READ(UNIT = kgen_unit) kgen_bound(2, 1)
                    READ(UNIT = kgen_unit) kgen_bound(1, 2)
                    READ(UNIT = kgen_unit) kgen_bound(2, 2)
                    READ(UNIT = kgen_unit) kgen_bound(1, 3)
                    READ(UNIT = kgen_unit) kgen_bound(2, 3)
                    ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1, kgen_bound(2, 3) - kgen_bound(1, 3) + 1))
                    READ(UNIT = kgen_unit) var
                    IF ( PRESENT(printvar) ) THEN
                        PRINT *, "** " // printvar // " **", var
                    END IF
                END IF
            END SUBROUTINE kgen_read_real_r8_dim3

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
