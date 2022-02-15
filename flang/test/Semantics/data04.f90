! RUN: %python %S/test_errors.py %s %flang_fc1
!Testing data constraints : C876, C877
module m
  integer :: first
  contains
    subroutine h
      integer a,b
      !C876
      !ERROR: Host-associated object 'first' must not be initialized in a DATA statement
      DATA first /1/
    end subroutine

    function g(i)
      integer ::i
      g = i *1024
    end

    function f(i)
      integer ::i
      integer ::result
      integer, allocatable :: a
      integer :: b(i)
      character(len=i), pointer:: charPtr
      character(len=i), allocatable:: charAlloc
      !C876
      !ERROR: Dummy argument 'i' must not be initialized in a DATA statement
      DATA i /1/
      !C876
      !ERROR: Function result 'f' must not be initialized in a DATA statement
      DATA f /1/
      !C876
      !ERROR: Procedure 'g' must not be initialized in a DATA statement
      DATA g /1/
      !C876
      !ERROR: Allocatable 'a' must not be initialized in a DATA statement
      DATA a /1/
      !C876
      !ERROR: Automatic variable 'b' must not be initialized in a DATA statement
      DATA b(0) /1/
      !C876
      !Ok: As charPtr is a pointer, it is not an automatic object
      DATA charPtr / NULL() /
      !C876
      !ERROR: Allocatable 'charalloc' must not be initialized in a DATA statement
      DATA charAlloc / 'abc' /
      f = i *1024
    end

    subroutine CheckObject(i)
      type specialNumbers
        integer one
        integer numbers(5)
        type(specialNumbers), pointer :: headOfTheList
        integer, pointer, dimension(:) :: ptoarray
        character, pointer, dimension(:) :: ptochar
      end type
      type large
        integer, allocatable :: allocVal
        integer, allocatable :: elt(:)
        integer val
        type(specialNumbers) numsArray(10)
      end type
      type(large) largeNumber
      type(large), allocatable :: allocatableLarge
      type(large) :: largeNumberArray(i)
      type(large) :: largeArray(5)
      character :: name(i)
      type small
        real :: x
      end type
      type(small), pointer :: sp
      !This case is ok.
      DATA(largeNumber % numsArray(j) % headOfTheList, j = 1, 10) / 10 * NULL() /
      !C877
      !ERROR: Data object must not contain pointer 'headofthelist' as a non-rightmost part
      DATA(largeNumber % numsArray(j) % headOfTheList % one, j = 1, 10) / 10 * 1 /
      !C877
      !ERROR: Rightmost data object pointer 'ptoarray' must not be subscripted
      DATA(largeNumber % numsArray(j) % ptoarray(1), j = 1, 10) / 10 * 1 /
      !C877
      !ERROR: Rightmost data object pointer 'ptochar' must not be subscripted
      DATA largeNumber % numsArray(1) % ptochar(1:2) / 'ab' /
      !C876
      !ERROR: Allocatable 'elt' must not be initialized in a DATA statement
      DATA(largeNumber % elt(j) , j = 1, 10) / 10 * 1/
      !C876
      !ERROR: Allocatable 'allocval' must not be initialized in a DATA statement
      DATA(largeArray(j) % allocVal , j = 1, 10) / 10 * 1/
      !C876
      !ERROR: Allocatable 'allocatablelarge' must not be initialized in a DATA statement
      DATA allocatableLarge % val / 1 /
      !C876
      !ERROR: Automatic variable 'largenumberarray' must not be initialized in a DATA statement
      DATA(largeNumberArray(j) % val, j = 1, 10) / 10 * NULL() /
      !C876
      !ERROR: Automatic variable 'name' must not be initialized in a DATA statement
      DATA name( : 2) / 'Ancd' /
      !ERROR: Target of pointer 'sp' must not be initialized in a DATA statement
      DATA sp%x / 1.0 /
    end
  end

  block data foo
          integer :: a,b
          common /c/ a,b
          !C876
          !OK: Correct use
          DATA a /1/
  end block data

  module m2
    integer m2_i
    type newType
      integer number
    end type
    type(newType) m2_number1
    contains

    subroutine checkDerivedType(m2_number)
      type(newType) m2_number
      type(newType) m2_number3
      !C876
      !ERROR: Dummy argument 'm2_number' must not be initialized in a DATA statement
      DATA m2_number%number /1/
      !C876
      !ERROR: Host-associated object 'm2_number1' must not be initialized in a DATA statement
      DATA m2_number1%number /1/
      !C876
      !OK: m2_number3 is not associated through use association
      DATA m2_number3%number /1/
    end
  end

  program new
    use m2
    type(newType) m2_number2
    !C876
    !ERROR: USE-associated object 'm2_i' must not be initialized in a DATA statement
    DATA m2_i /1/
    !C876
    !ERROR: USE-associated object 'm2_number1' must not be initialized in a DATA statement
    DATA m2_number1%number /1/
    !C876
    !OK: m2_number2 is not associated through use association
    DATA m2_number2%number /1/
  end program
