!RUN: %python %S/test_errors.py %s %flang_fc1
program test
  real, target :: x
  type t1
    integer :: j/1/
    real, pointer :: ap/x/
  end type
  type, extends(t1) :: t2
    integer :: k/2/
  end type
  type t3(k)
    integer, kind :: k
    !ERROR: Component 'j' in a parameterized data type may not be initialized with a legacy DATA-style value list
    integer :: j/3/
  end type
  type t4
    !ERROR: DATA statement set has more values than objects
    integer j(1) /4, 5/
  end type
  type t5
    integer uninitialized
  end type
  type(t2), parameter :: x2 = t2() !ok
  integer(kind=merge(1,-1,x2%j==1)) tx2j
  integer(kind=merge(2,-1,x2%k==2)) tx2k
  !ERROR: Structure constructor lacks a value for component 'uninitialized'
  type(t5), parameter :: x5 = t5()
end
