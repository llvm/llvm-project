! RUN: %python %S/test_errors.py %s %flang_fc1
! DATA statement errors
subroutine s1
  type :: t1
    integer :: j = 666
  end type t1
  type(t1) :: t1x
  !ERROR: Default-initialized 't1x' must not be initialized in a DATA statement
  data t1x%j / 777 /
  integer :: ja = 888
  !ERROR: Default-initialized 'ja' must not be initialized in a DATA statement
  data ja / 999 /
  integer :: a1(10)
  !ERROR: DATA statement set has more values than objects
  data a1(1:9:2) / 6 * 1 /
  integer :: a2(10)
  !ERROR: DATA statement set has no value for 'a2(2_8)'
  data (a2(k),k=10,1,-2) / 4 * 1 /
  integer :: a3(2)
  !ERROR: DATA statement implied DO loop has a step value of zero
  data (a3(j),j=1,2,0)/2*333/
  integer :: a4(3)
  !ERROR: DATA statement designator 'a4(5_8)' is out of range
  data (a4(j),j=1,5,2) /3*222/
  interface
    real function rfunc(x)
      real, intent(in) :: x
    end function
  end interface
  real, pointer :: rp
  !ERROR: Procedure 'rfunc' may not be used to initialize 'rp', which is not a procedure pointer
  data rp/rfunc/
  procedure(rfunc), pointer :: rpp
  real, target :: rt
  !ERROR: Data object 'rt' may not be used to initialize 'rpp', which is a procedure pointer
  data rpp/rt/
  !ERROR: Initializer for 'rt' must not be a pointer
  data rt/null()/
  !ERROR: Initializer for 'rt' must not be a procedure
  data rt/rfunc/
  integer :: jx, jy
  !WARNING: DATA statement value initializes 'jx' of type 'INTEGER(4)' with CHARACTER
  data jx/'abc'/
  !ERROR: DATA statement value could not be converted to the type 'INTEGER(4)' of the object 'jx'
  data jx/t1()/
  !ERROR: DATA statement value 'jy' for 'jx' is not a constant
  data jx/jy/
end subroutine
