! RUN: %python %S/test_errors.py %s %flang_fc1
! TODO: crashes compiler mysteriously when built with MSVC since patch
! 10b990a03b1ede533b8443edffa3607aae4295c7n or maybe earlier
! UNSUPPORTED: system-windows

!Shape analysis related tests for SELECT RANK Construct(R1148)
program select_rank
   implicit none
   integer, dimension(2,3):: arr_pass
   call check(arr_pass)

contains
    subroutine check(arr)
        implicit none
        integer :: arr(..)
        INTEGER :: j
        select rank (arr)
            rank(2)
                j = INT(0, KIND=MERGE(KIND(0), -1, SIZE(SHAPE(arr)) == 2)) !arr is dummy
        end select
    end subroutine
    subroutine check2(arr)
        implicit none
        integer :: arr(..)
        INTEGER :: j
        integer,dimension(-1:10, 20:30) :: brr

        select rank (arr)
            rank(2)
                j = INT(0, KIND=MERGE(KIND(0), -1, SIZE(SHAPE(brr)) == 2)) !brr is local to subroutine
        end select
    end subroutine
    subroutine checK3(arr)
        implicit none
        integer :: arr(..)
        INTEGER :: j,I,n=5,m=5
        integer,dimension(-1:10, 20:30) :: brr
        integer :: array(2) = [10,20]
        REAL, DIMENSION(5, 5) :: A
        select rank (arr)
            rank(2)
                FORALL (i=1:n,j=1:m,RANK(arr).EQ.SIZE(SHAPE(brr))) &
                    A(i,j) = 1/A(i,j)
        end select
    end subroutine
    subroutine check4(arr)
        implicit none
        integer :: arr(..)
        REAL, DIMENSION(2,3) :: A
        REAL, DIMENSION(0:1,0:2) :: B
        INTEGER :: j
        select rank (arr)
            rank(2)
                A = B   !will assign to only same shape after analysing in any order.
        end select
    end subroutine
    subroutine check5(arr)
        implicit none
        integer :: arr(..)
        INTEGER :: j
        select rank (arr)
            rank(2)
                j = LOC(arr(1,2))
        end select
    end subroutine
end program
