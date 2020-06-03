! RUN: %S/test_errors.sh %s %t %f18
!Testing data constraints : C874 - C875, C878 - C881 
module m
    integer, target :: modarray(1)
  contains
    function f(i)
      integer, intent(in) :: i
      integer, pointer :: f
      f => modarray(i)
    end
    subroutine CheckObject 
      type specialNumbers
        integer one
        integer numbers(5)
      end type
      type large
        integer elt(10)
        integer val
        type(specialNumbers) nums
        type(specialNumbers) numsArray(5)
      end type
      type(specialNumbers), parameter ::newNums = &
              specialNumbers(1, (/ 1, 2, 3, 4, 5 /))
      type(specialNumbers), parameter ::newNumsArray(2) = &
              (/ SpecialNumbers(1, (/ 1, 2, 3, 4, 5 /)), &
              SpecialNumbers(1, (/ 1, 2, 3,4, 5 /)) /)
      type(specialNumbers) nums
      type(large) largeArray(5)
      type(large) largeNumber
      real :: a[*]
      real :: b(5)
      integer :: x
      real, parameter:: c(5) = (/ 1, 2, 3, 4, 5 /)
      integer :: d(10, 10)
      character :: name(12)
      integer :: ind = 2
      !C874
      !ERROR: Data object must not be a coindexed variable
      DATA a[1] / 1 /
      !C874
      !ERROR: Data object must not be a coindexed variable
      DATA(a[i], i = 1, 5) / 5 * 1 /
      !C875
      !ERROR: Data object variable must not be a function reference
      DATA f(1) / 1 / 
      !C875
      !ERROR: Data object must have constant subscripts
      DATA b(ind) / 1 /
      !C875
      !ERROR: Data object must have constant subscripts
      DATA name( : ind) / 'Ancd' /
      !C875
      !ERROR: Data object must have constant subscripts
      DATA name(ind:) / 'Ancd' /
      !C878
      !ERROR: Data implied do object must be a variable
      DATA(c(i), i = 1, 5) / 5 * 1 /
      !C878
      !ERROR: Data implied do object must be a variable
      DATA(newNumsArray(i), i = 1, 2) &
              / specialNumbers(1, 2 * (/ 1, 2, 3, 4, 5 /)) /
      !C880
      !ERROR: Data implied do structure component must be subscripted
      DATA(nums % one, i = 1, 5) / 5 * 1 /
      !C879
      !ERROR: Data implied do object must be a variable
      DATA(newNums % numbers(i), i = 1, 5) / 5 * 1 /
      !C879
      !ERROR: Data implied do object must be a variable
      DATA(newNumsArray(i) % one, i = 1, 5) / 5 * 1 /
      !C880
      !OK: Correct use
      DATA(largeArray(j) % nums % one, j = 1, 10) / 10 * 1 /
      !C880
      !OK: Correct use
      DATA(largeNumber % numsArray(j) % one, j = 1, 10) / 10 * 1 /
      !C881
      !ERROR: Data object must have constant subscripts
      DATA(b(x), i = 1, 5) / 5 * 1 /
      !C881 
      !OK: Correct use
      DATA(nums % numbers(i), i = 1, 5) / 5 * 1 /
      !C881
      !OK: Correct use
      DATA((d(i, j), i = 1, 10), j = 1, 10) / 100 * 1 /
      !C881
      !OK: Correct use
      DATA(d(i, 1), i = 1, 10) / 10 * 1 /
    end 
  end
