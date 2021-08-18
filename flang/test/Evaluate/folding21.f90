! RUN: %S/test_folding.sh %s %t %flang_fc1
! REQUIRES: shell
! Check array sizes with varying extents, including extents where the upper
! bound is less than the lower bound
module m
 contains
  subroutine s1(a,b)
    real nada1(-2:-1)    ! size =  2
    real nada2(-1:-1)    ! size =  1
    real nada3( 0:-1)    ! size =  0
    real nada4( 1:-1)    ! size =  0
    real nada5( 2:-1)    ! size =  0
    real nada6( 3:-1)    ! size =  0
    real nada7( 5, 3:-1) ! size =  0
    real nada8( -1)      ! size =  0

    integer, parameter :: size1 = size(nada1)
    integer, parameter :: size2 = size(nada2)
    integer, parameter :: size3 = size(nada3)
    integer, parameter :: size4 = size(nada4)
    integer, parameter :: size5 = size(nada5)
    integer, parameter :: size6 = size(nada6)
    integer, parameter :: size7 = size(nada7)
    integer, parameter :: size8 = size(nada8)

    logical, parameter :: test_size_1 = size1 == 2
    logical, parameter :: test_size_2 = size2 == 1
    logical, parameter :: test_size_3 = size3 == 0
    logical, parameter :: test_size_4 = size4 == 0
    logical, parameter :: test_size_5 = size5 == 0
    logical, parameter :: test_size_6 = size6 == 0
    logical, parameter :: test_size_7 = size7 == 0
    logical, parameter :: test_size_8 = size8 == 0
  end subroutine
end module
