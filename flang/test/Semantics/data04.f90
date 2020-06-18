! RUN: %S/test_errors.sh %s %t %f18
!Testing data constraints : C876, C877
module m
  integer :: first
  contains
    subroutine h
      integer a,b
      !C876
      !ERROR: Data object part 'first' must not be accessed by host association
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
      !ERROR: Data object part 'i' must not be a dummy argument
      DATA i /1/
      !C876
      !ERROR: Data object part 'f' must not be a function result
      DATA f /1/
      !C876
      !ERROR: Data object part 'g' must not be a function name
      DATA g /1/
      !C876
      !ERROR: Data object part 'a' must not be an allocatable object
      DATA a /1/
      !C876
      !ERROR: Data object part 'b' must not be an automatic object
      DATA b(0) /1/
      !C876
      !Ok: As charPtr is a pointer, it is not an automatic object
      DATA charPtr / NULL() /
      !C876
      !ERROR: Data object part 'charalloc' must not be an allocatable object
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
        type(specialNumbers) numsArray(5)
      end type
      type(large) largeNumber
      type(large), allocatable :: allocatableLarge
      type(large) :: largeNumberArray(i)
      type(large) :: largeArray(5)
      character :: name(i)
      !C877
      !OK: Correct use
      DATA(largeNumber % numsArray(j) % headOfTheList, j = 1, 10) / 10 * NULL() /
      !C877
      !ERROR: Data object must not contain pointer 'headofthelist' as a non-rightmost part
      DATA(largeNumber % numsArray(j) % headOfTheList % one, j = 1, 10) / 10 * NULL() /
      !C877
      !ERROR: Rightmost data object pointer 'ptoarray' must not be subscripted
      DATA(largeNumber % numsArray(j) % ptoarray(1), j = 1, 10) / 10 * 1 /
      !C877
      !ERROR: Rightmost data object pointer 'ptochar' must not be subscripted
      DATA largeNumber % numsArray(0) % ptochar(1:2) / 'ab' /
      !C876
      !ERROR: Data object part 'elt' must not be an allocatable object
      DATA(largeNumber % elt(j) , j = 1, 10) / 10 * 1/
      !C876
      !ERROR: Data object part 'allocval' must not be an allocatable object
      DATA(largeArray(j) % allocVal , j = 1, 10) / 10 * 1/
      !C876
      !ERROR: Data object part 'allocatablelarge' must not be an allocatable object
      DATA allocatableLarge % val / 1 /
      !C876
      !ERROR: Data object part 'largenumberarray' must not be an automatic object
      DATA(largeNumberArray(j) % val, j = 1, 10) / 10 * NULL() /
      !C876
      !ERROR: Data object part 'name' must not be an automatic object
      DATA name( : 2) / 'Ancd' /
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
      !ERROR: Data object part 'm2_number' must not be a dummy argument
      DATA m2_number%number /1/
      !C876
      !ERROR: Data object part 'm2_number1' must not be accessed by host association
      DATA m2_number1%number /1/
      !C876
      !OK: m2_number3 is not associated through use association
      DATA m2_number3%number /1/
    end
  end

  program new
    use m2
    integer a
    real    b,c
    type seqType
      sequence
      integer number
    end type
    type(SeqType) num
    COMMON b,a,c,num
    type(newType) m2_number2
    !C876
    !ERROR: Data object part 'b' must not be in blank COMMON
    DATA b /1/
    !C876
    !ERROR: Data object part 'm2_i' must not be accessed by use association
    DATA m2_i /1/
    !C876
    !ERROR: Data object part 'm2_number1' must not be accessed by use association
    DATA m2_number1%number /1/
    !C876
    !OK: m2_number2 is not associated through use association
    DATA m2_number2%number /1/
    !C876
    !ERROR: Data object part 'num' must not be in blank COMMON
    DATA num%number /1/
  end program
