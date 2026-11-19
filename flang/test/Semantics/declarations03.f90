! RUN: %python %S/test_errors.py %s %flang_fc1
! test bind(c) name conflict

module m

  integer :: x, y, z, w, i, j, k

  !ERROR: Two entities have the same global name 'aa'
  common /blk1/ x, /blk2/ y
  bind(c, name="aa") :: /blk1/, /blk2/

  integer :: t
  !ERROR: Two entities have the same global name 'bb'
  common /blk3/ z
  bind(c, name="bb") :: /blk3/, t

  integer :: t2
  !ERROR: Two entities have the same global name 'cc'
  common /blk4/ w
  bind(c, name="cc") :: t2, /blk4/

  !ERROR: The entity 'blk5' has multiple BIND names ('dd' and 'ee')
  common /blk5/ i
  bind(c, name="dd") :: /blk5/
  bind(c, name="ee") :: /blk5/

  !ERROR: Two entities have the same global name 'ff'
  common /blk6/ j, /blk7/ k
  bind(c, name="ff") :: /blk6/
  bind(c, name="ff") :: /blk7/

  !ERROR: The entity 's1' has multiple BIND names ('gg' and 'hh')
  integer :: s1
  bind(c, name="gg") :: s1
  !ERROR: BIND_C attribute was already specified on 's1'
  bind(c, name="hh") :: s1

  !ERROR: Two entities have the same global name 'ii'
  integer :: s2, s3
  bind(c, name="ii") :: s2
  bind(c, name="ii") :: s3

  !ERROR: The entity 's4' has multiple BIND names ('ss1' and 'jj')
  integer, bind(c, name="ss1") :: s4
  !ERROR: BIND_C attribute was already specified on 's4'
  bind(c, name="jj") :: s4

  !ERROR: The entity 's5' has multiple BIND names ('kk' and 'ss2')
  bind(c, name="kk") :: s5
  !ERROR: BIND_C attribute was already specified on 's5'
  integer, bind(c, name="ss2") :: s5

  integer, bind(c, name="s6explicit") :: s6
  dimension s6(10) ! caused spurious error

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

subroutine common3()
  real :: y
  common /comA/ y
  bind(c, name='xcom2') /comA/ ! no error
end subroutine

subroutine common4()
  real :: y
  common /comB/ y
  bind(c, name='xcom2') /comB/ ! no error
end subroutine

! A default (implicit) binding label -- bind(c) with no NAME= -- is the
! common block's own name, and still counts as a genuine binding label
! for the merge; this is how most real code spells it.
subroutine common14()
  integer :: m
  common /com5/ m
  bind(c) :: /com5/
end subroutine
subroutine common15()
  integer :: n
  common /x5/ n
  bind(c, name="com5") :: /x5/
end subroutine

! A bound common may not collide with an unbound one: the objects are "com_" and "com".
subroutine common5()
  integer :: i
  common /com/ i
end subroutine
subroutine common6()
  integer :: j
  !ERROR: Two entities have the same global name 'com'
  common /b/ j
  bind(c, name="com") :: /b/
end subroutine

! An empty NAME= leaves the common unbound, so this is the same case.
subroutine common7()
  integer :: i
  common /com2/ i
  bind(c, name="") :: /com2/
end subroutine
subroutine common8()
  integer :: j
  !ERROR: Two entities have the same global name 'com2'
  common /b2/ j
  bind(c, name="com2") :: /b2/
end subroutine

! Two commons in the SAME scoping unit sharing a label is still an error,
! even when another scoping unit already uses that label.
subroutine common9()
  integer :: i
  common /x/ i
  bind(c, name="com3") :: /x/
end subroutine
subroutine common10()
  integer :: j, k
  common /a3/ j
  !ERROR: Two entities have the same global name 'com3'
  common /b3/ k
  bind(c, name="com3") :: /a3/
  bind(c, name="com3") :: /b3/
end subroutine

! Three-way: bound /com4/, unbound /com4/, then a differently named bound
! common -- the unbound one and /other/ do not share storage.
subroutine common11()
  integer :: i
  common /com4/ i
  bind(c, name="com4") :: /com4/
end subroutine
subroutine common12()
  integer :: j
  common /com4/ j
end subroutine
subroutine common13()
  integer :: k
  !ERROR: Two entities have the same global name 'com4'
  common /other/ k
  bind(c, name="com4") :: /other/
end subroutine

module a
  integer, bind(c, name="int") :: i
end module

module b
  !ERROR: Two entities have the same global name 'int'
  integer, bind(c, name="int") :: i
end module

module c
  bind(c, name = "AAA") a
  integer aaa ! ensure no bogus error about multiple binding names
end module
