! RUN: %python %S/test_errors.py %s %flang_fc1
! test bind(c) name conflict

module m

  integer :: x, y, z, w, i, j, k

  !ERROR: Two entities have the same BIND(C) name 'aa'
  common /blk1/ x, /blk2/ y
  bind(c, name="aa") :: /blk1/, /blk2/

  integer :: t
  !ERROR: Two entities have the same BIND(C) name 'bb'
  common /blk3/ z
  bind(c, name="bb") :: /blk3/, t

  integer :: t2
  !ERROR: Two entities have the same BIND(C) name 'cc'
  common /blk4/ w
  bind(c, name="cc") :: t2, /blk4/

  !ERROR: The entity 'blk5' has multiple BIND names
  common /blk5/ i
  bind(c, name="dd") :: /blk5/
  bind(c, name="ee") :: /blk5/

  !ERROR: Two entities have the same BIND(C) name 'ff'
  common /blk6/ j, /blk7/ k
  bind(c, name="ff") :: /blk6/
  bind(c, name="ff") :: /blk7/

  !ERROR: The entity 's1' has multiple BIND names
  integer :: s1
  bind(c, name="gg") :: s1
  bind(c, name="hh") :: s1

  !ERROR: Two entities have the same BIND(C) name 'ii'
  integer :: s2, s3
  bind(c, name="ii") :: s2
  bind(c, name="ii") :: s3

  !ERROR: The entity 's4' has multiple BIND names
  integer, bind(c, name="ss1") :: s4
  bind(c, name="jj") :: s4

  !ERROR: The entity 's5' has multiple BIND names
  bind(c, name="kk") :: s5
  integer, bind(c, name="ss2") :: s5

end

subroutine common1()
  real :: x
  common /com/ x
  bind(c, name='xcom') /com/ ! no error
end subroutine

subroutine common2()
  real :: x
  common /com/ x
  bind(c, name='xcom') /com/ ! no error
end subroutine

module a
  integer, bind(c, name="int") :: i
end module

module b
  !ERROR: Two entities have the same BIND(C) name 'int'
  integer, bind(c, name="int") :: i
end module
