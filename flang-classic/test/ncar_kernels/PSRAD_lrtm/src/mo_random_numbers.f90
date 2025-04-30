
! KGEN-generated Fortran source file
!
! Filename    : mo_random_numbers.f90
! Generated at: 2015-02-19 15:30:29
! KGEN version: 0.4.4



    MODULE mo_random_numbers
        USE mo_kind, ONLY: dp
        USE mo_kind, ONLY: i8
        IMPLICIT NONE
        LOGICAL, parameter :: big_endian = (transfer(1_i8, 1) == 0)
        INTEGER, parameter :: state_size = 4
        INTEGER :: global_seed(state_size)  =              (/123456789,362436069,21288629,14921776/)
        PRIVATE
        PUBLIC get_random

        INTERFACE get_random
            MODULE PROCEDURE kisssca, kiss_global, kissvec, kissvec_all, kissvec_global
        END INTERFACE get_random
            PUBLIC read_externs_mo_random_numbers

        ! read interface
        PUBLIC kgen_read_var
        interface kgen_read_var
            module procedure read_var_integer_4_dim1
        end interface kgen_read_var

        CONTAINS

        ! module extern variables

        SUBROUTINE read_externs_mo_random_numbers(kgen_unit)
        integer, intent(in) :: kgen_unit
        READ(UNIT=kgen_unit) global_seed
        END SUBROUTINE read_externs_mo_random_numbers


        ! read subroutines
        subroutine read_var_integer_4_dim1(var, kgen_unit)
            integer, intent(in) :: kgen_unit
            integer(kind=4), intent(out), dimension(:), allocatable :: var
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
        ! -----------------------------------------------

        ! -----------------------------------------------

        ! -----------------------------------------------

        SUBROUTINE kissvec_all(kproma, kbdim, seed, harvest)
            INTEGER, intent(in   ) :: kbdim
            INTEGER, intent(in   ) :: kproma
            INTEGER, intent(inout) :: seed(:,:) ! Dimension nproma, seed_size
            REAL(KIND=dp), intent(  out) :: harvest(:) ! Dimension nproma
            LOGICAL :: mask(kbdim)
            mask(:) = .true.
            CALL kissvec(kproma, kbdim, seed, mask, harvest)
        END SUBROUTINE kissvec_all
        ! -----------------------------------------------

        SUBROUTINE kissvec(kproma, kbdim, seed, mask, harvest)
            INTEGER, intent(in   ) :: kbdim
            INTEGER, intent(in   ) :: kproma
            INTEGER, intent(inout) :: seed(:,:) ! Dimension kbdim, seed_size or bigger
            LOGICAL, intent(in   ) :: mask(kbdim)
            REAL(KIND=dp), intent(  out) :: harvest(kbdim)
            INTEGER(KIND=i8) :: kiss(kproma)
            INTEGER :: jk
            DO jk = 1, kproma
                IF (mask(jk)) THEN
                    kiss(jk) = 69069_i8 * seed(jk,1) + 1327217885
                    seed(jk,1) = low_byte(kiss(jk))
                    seed(jk,2) = m (m (m (seed(jk,2), 13), - 17), 5)
                    seed(jk,3) = 18000 * iand (seed(jk,3), 65535) + ishft (seed(jk,3), - 16)
                    seed(jk,4) = 30903 * iand (seed(jk,4), 65535) + ishft (seed(jk,4), - 16)
                    kiss(jk) = int(seed(jk,1), i8) + seed(jk,2) + ishft (seed(jk,3), 16) + seed(jk,4)
                    harvest(jk) = low_byte(kiss(jk))*2.328306e-10_dp + 0.5_dp
                    ELSE
                    harvest(jk) = 0._dp
                END IF 
            END DO 
        END SUBROUTINE kissvec
        ! -----------------------------------------------

        SUBROUTINE kisssca(seed, harvest)
            INTEGER, intent(inout) :: seed(:)
            REAL(KIND=dp), intent(  out) :: harvest
            INTEGER(KIND=i8) :: kiss
            kiss = 69069_i8 * seed(1) + 1327217885
            seed(1) = low_byte(kiss)
            seed(2) = m (m (m (seed(2), 13), - 17), 5)
            seed(3) = 18000 * iand (seed(3), 65535) + ishft (seed(3), - 16)
            seed(4) = 30903 * iand (seed(4), 65535) + ishft (seed(4), - 16)
            kiss = int(seed(1), i8) + seed(2) + ishft (seed(3), 16) + seed(4)
            harvest = low_byte(kiss)*2.328306e-10_dp + 0.5_dp
        END SUBROUTINE kisssca
        ! -----------------------------------------------

        SUBROUTINE kiss_global(harvest)
            REAL(KIND=dp), intent(inout) :: harvest
            CALL kisssca(global_seed, harvest)
        END SUBROUTINE kiss_global
        ! -----------------------------------------------

        SUBROUTINE kissvec_global(harvest)
            REAL(KIND=dp), intent(inout) :: harvest(:)
            INTEGER :: i
            DO i = 1, size(harvest)
                CALL kisssca(global_seed, harvest(i))
            END DO 
        END SUBROUTINE kissvec_global
        ! -----------------------------------------------

        elemental integer FUNCTION m(k, n)
            INTEGER, intent(in) :: k
            INTEGER, intent(in) :: n
            m = ieor (k, ishft (k, n)) ! UNRESOLVED: m
        END FUNCTION m
        ! -----------------------------------------------

        elemental integer FUNCTION low_byte(i)
            INTEGER(KIND=i8), intent(in) :: i
            IF (big_endian) THEN
                low_byte = transfer(ishft(i,bit_size(1)),1) ! UNRESOLVED: low_byte
                ELSE
                low_byte = transfer(i,1) ! UNRESOLVED: low_byte
            END IF 
        END FUNCTION low_byte
    END MODULE mo_random_numbers
