
! KGEN-generated Fortran source file
!
! Filename    : kernel_driver.f90
! Generated at: 2015-07-31 20:45:42
! KGEN version: 0.4.13


PROGRAM kernel_driver
    USE rrtmg_sw_spcvmc, ONLY : spcvmc_sw
    USE shr_kind_mod, ONLY: r8 => shr_kind_r8
    USE rrsw_vsn, ONLY : kgen_read_externs_rrsw_vsn
    USE rrsw_kg23, ONLY : kgen_read_externs_rrsw_kg23
    USE rrsw_kg28, ONLY : kgen_read_externs_rrsw_kg28
    USE rrsw_con, ONLY : kgen_read_externs_rrsw_con
    USE rrsw_kg24, ONLY : kgen_read_externs_rrsw_kg24
    USE rrsw_kg25, ONLY : kgen_read_externs_rrsw_kg25
    USE rrsw_kg26, ONLY : kgen_read_externs_rrsw_kg26
    USE rrsw_kg27, ONLY : kgen_read_externs_rrsw_kg27
    USE rrsw_kg19, ONLY : kgen_read_externs_rrsw_kg19
    USE rrsw_kg18, ONLY : kgen_read_externs_rrsw_kg18
    USE rrsw_kg22, ONLY : kgen_read_externs_rrsw_kg22
    USE rrsw_wvn, ONLY : kgen_read_externs_rrsw_wvn
    USE rrsw_kg17, ONLY : kgen_read_externs_rrsw_kg17
    USE rrsw_kg16, ONLY : kgen_read_externs_rrsw_kg16
    USE rrsw_kg20, ONLY : kgen_read_externs_rrsw_kg20
    USE rrsw_kg29, ONLY : kgen_read_externs_rrsw_kg29
    USE rrsw_kg21, ONLY : kgen_read_externs_rrsw_kg21
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
    REAL(KIND=r8), allocatable :: selffac(:,:)
    REAL(KIND=r8), allocatable :: selffrac(:,:)
    INTEGER :: ncol
    REAL(KIND=r8), allocatable :: forfac(:,:)
    INTEGER :: nlayers
    REAL(KIND=r8), allocatable :: forfrac(:,:)
    INTEGER, allocatable :: indself(:,:)
    REAL(KIND=r8), allocatable :: colh2o(:,:)
    REAL(KIND=r8), allocatable :: colco2(:,:)
    REAL(KIND=r8), allocatable :: colch4(:,:)
    REAL(KIND=r8), allocatable :: colo3(:,:)
    REAL(KIND=r8), allocatable :: colmol(:,:)
    REAL(KIND=r8), allocatable :: colo2(:,:)
    INTEGER, allocatable :: laytrop(:)
    INTEGER, allocatable :: jp(:,:)
    INTEGER, allocatable :: jt(:,:)
    INTEGER, allocatable :: indfor(:,:)
    INTEGER, allocatable :: jt1(:,:)
    REAL(KIND=r8), allocatable :: fac00(:,:)
    REAL(KIND=r8), allocatable :: fac01(:,:)
    REAL(KIND=r8), allocatable :: fac10(:,:)
    REAL(KIND=r8), allocatable :: fac11(:,:)

    DO kgen_repeat_counter = 0, 8
        kgen_counter = kgen_counter_at(mod(kgen_repeat_counter, 3)+1)
        WRITE( kgen_counter_conv, * ) kgen_counter
        kgen_mpi_rank = kgen_mpi_rank_at(mod(kgen_repeat_counter, 3)+1)
        WRITE( kgen_mpi_rank_conv, * ) kgen_mpi_rank
        kgen_filepath = "../data/taumol_sw." // trim(adjustl(kgen_counter_conv)) // "." // trim(adjustl(kgen_mpi_rank_conv))
        kgen_unit = kgen_get_newunit()
        OPEN (UNIT=kgen_unit, FILE=kgen_filepath, STATUS="OLD", ACCESS="STREAM", FORM="UNFORMATTED", ACTION="READ", IOSTAT=kgen_ierr, CONVERT="BIG_ENDIAN")
        WRITE (*,*)
        IF ( kgen_ierr /= 0 ) THEN
            CALL kgen_error_stop( "FILE OPEN ERROR: " // trim(adjustl(kgen_filepath)) )
        END IF
        WRITE (*,*)
        WRITE (*,*) "** Verification against '" // trim(adjustl(kgen_filepath)) // "' **"

            CALL kgen_read_externs_rrsw_vsn(kgen_unit)
            CALL kgen_read_externs_rrsw_kg23(kgen_unit)
            CALL kgen_read_externs_rrsw_kg28(kgen_unit)
            CALL kgen_read_externs_rrsw_con(kgen_unit)
            CALL kgen_read_externs_rrsw_kg24(kgen_unit)
            CALL kgen_read_externs_rrsw_kg25(kgen_unit)
            CALL kgen_read_externs_rrsw_kg26(kgen_unit)
            CALL kgen_read_externs_rrsw_kg27(kgen_unit)
            CALL kgen_read_externs_rrsw_kg19(kgen_unit)
            CALL kgen_read_externs_rrsw_kg18(kgen_unit)
            CALL kgen_read_externs_rrsw_kg22(kgen_unit)
            CALL kgen_read_externs_rrsw_wvn(kgen_unit)
            CALL kgen_read_externs_rrsw_kg17(kgen_unit)
            CALL kgen_read_externs_rrsw_kg16(kgen_unit)
            CALL kgen_read_externs_rrsw_kg20(kgen_unit)
            CALL kgen_read_externs_rrsw_kg29(kgen_unit)
            CALL kgen_read_externs_rrsw_kg21(kgen_unit)

            ! driver variables
            READ(UNIT=kgen_unit) nlayers
            READ(UNIT=kgen_unit) ncol
            CALL kgen_read_integer_4_dim1(laytrop, kgen_unit)
            CALL kgen_read_integer_4_dim2(indfor, kgen_unit)
            CALL kgen_read_integer_4_dim2(indself, kgen_unit)
            CALL kgen_read_integer_4_dim2(jp, kgen_unit)
            CALL kgen_read_integer_4_dim2(jt, kgen_unit)
            CALL kgen_read_integer_4_dim2(jt1, kgen_unit)
            CALL kgen_read_real_r8_dim2(colmol, kgen_unit)
            CALL kgen_read_real_r8_dim2(colh2o, kgen_unit)
            CALL kgen_read_real_r8_dim2(colco2, kgen_unit)
            CALL kgen_read_real_r8_dim2(colch4, kgen_unit)
            CALL kgen_read_real_r8_dim2(colo3, kgen_unit)
            CALL kgen_read_real_r8_dim2(colo2, kgen_unit)
            CALL kgen_read_real_r8_dim2(forfac, kgen_unit)
            CALL kgen_read_real_r8_dim2(forfrac, kgen_unit)
            CALL kgen_read_real_r8_dim2(selffac, kgen_unit)
            CALL kgen_read_real_r8_dim2(selffrac, kgen_unit)
            CALL kgen_read_real_r8_dim2(fac00, kgen_unit)
            CALL kgen_read_real_r8_dim2(fac01, kgen_unit)
            CALL kgen_read_real_r8_dim2(fac10, kgen_unit)
            CALL kgen_read_real_r8_dim2(fac11, kgen_unit)

            call spcvmc_sw(nlayers, ncol, laytrop, indfor, indself, jp, jt, jt1, colmol, colh2o, colco2, colch4, colo3, colo2, forfac, forfrac, selffac, selffrac, fac00, fac01, fac10, fac11, kgen_unit)

            CLOSE (UNIT=kgen_unit)
        END DO
    CONTAINS

        ! write subroutines
            SUBROUTINE kgen_read_integer_4_dim1(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                integer(KIND=4), INTENT(OUT), ALLOCATABLE, DIMENSION(:) :: var
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
            END SUBROUTINE kgen_read_integer_4_dim1

            SUBROUTINE kgen_read_integer_4_dim2(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                integer(KIND=4), INTENT(OUT), ALLOCATABLE, DIMENSION(:,:) :: var
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
            END SUBROUTINE kgen_read_integer_4_dim2

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
