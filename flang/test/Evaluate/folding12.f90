! RUN: %S/test_folding.sh %s %t %f18
! Test folding of structure constructors
module m1
  type parent_type
    integer :: parent_field
  end type parent_type
  type, extends(parent_type) :: child_type
    integer :: child_field 
  end type child_type
  type parent_array_type
    integer, dimension(2) :: parent_field
  end type parent_array_type
  type, extends(parent_array_type) :: child_array_type
    integer :: child_field
  end type child_array_type

  type(child_type), parameter :: child_const1 = child_type(10, 11)
  logical, parameter :: test_child1 = child_const1%child_field == 11
  logical, parameter :: test_parent = child_const1%parent_field == 10

  type(child_type), parameter :: child_const2 = child_type(12, 13)
  type(child_type), parameter :: array_var(2) = &
    [child_type(14, 15), child_type(16, 17)]
  logical, parameter :: test_array_child = array_var(2)%child_field == 17 
  logical, parameter :: test_array_parent = array_var(2)%parent_field == 16

  type array_type
    real, dimension(3) :: real_field
  end type array_type
  type(array_type), parameter :: array_var2 = &
    array_type([(real(i*i), i = 1,3)])
  logical, parameter :: test_array_var = array_var2%real_field(2) == 4.0

  type(child_type), parameter, dimension(2) :: child_const3 = &
    [child_type(18, 19), child_type(20, 21)]
  integer, dimension(2), parameter :: int_const4 = &
    child_const3(:)%parent_field
  logical, parameter :: test_child2 = int_const4(1) == 18

  type(child_array_type), parameter, dimension(2) :: child_const5 = &
    [child_array_type([22, 23], 24), child_array_type([25, 26], 27)]
  integer, dimension(2), parameter :: int_const6 = child_const5(:)%parent_field(2)
  logical, parameter :: test_child3 = int_const6(1) == 23 

  type(child_type), parameter :: child_const7 =  child_type(28, 29)
  type(parent_type), parameter :: parent_const8 = child_const7%parent_type
  logical, parameter :: test_child4 = parent_const8%parent_field == 28

  type(child_type), parameter :: child_const9 = &
    child_type(parent_type(30), 31)
  integer, parameter :: int_const10 = child_const9%parent_field
  logical, parameter :: test_child5 = int_const10 == 30

end module m1

module m2
  type grandparent_type
    real :: grandparent_field
  end type grandparent_type
  type, extends(grandparent_type) :: parent_type
    integer :: parent_field
  end type parent_type
  type, extends(parent_type) :: child_type
    real :: child_field
  end type child_type

  type(child_type), parameter :: child_const1 = child_type(10.0, 11, 12.0)
  integer, parameter :: int_const2 = &
    child_const1%grandparent_type%grandparent_field
  logical, parameter :: test_child1 = int_const2 == 10.0
  integer, parameter :: int_const3 = &
    child_const1%grandparent_field
  logical, parameter :: test_child2 = int_const3 == 10.0

  type(child_type), parameter :: child_const4 = &
    child_type(parent_type(13.0, 14), 15.0)
  integer, parameter :: int_const5 = &
    child_const4%grandparent_type%grandparent_field
  logical, parameter :: test_child3 = int_const5 == 13.0

  type(child_type), parameter :: child_const6 = &
    child_type(parent_type(grandparent_type(16.0), 17), 18.0)
  integer, parameter :: int_const7 = &
    child_const6%grandparent_type%grandparent_field
  logical, parameter :: test_child4 = int_const7 == 16.0
  integer, parameter :: int_const8 = &
    child_const6%grandparent_field
  logical, parameter :: test_child5 = int_const8 == 16.0
end module m2

module m3
  ! tests that use components with default initializations and with the
  ! components in the structure constructors in a different order from the
  ! declared order
  type parent_type
    integer :: parent_field1
    real :: parent_field2 = 20.0
    logical :: parent_field3
  end type parent_type
  type, extends(parent_type) :: child_type
    real :: child_field1
    logical :: child_field2 = .false.
    integer :: child_field3
  end type child_type

  type(child_type), parameter :: child_const1 = &
    child_type( &
      parent_field2 = 10.0, child_field3 = 11, &
      child_field2 = .true., parent_field3 = .false., &
      parent_field1 = 12, child_field1 = 13.3)
  logical, parameter :: test_child1 = child_const1%child_field1 == 13.3
  logical, parameter :: test_child2 = child_const1%child_field2 .eqv. .true.
  logical, parameter :: test_child3 = child_const1%child_field3 == 11
  logical, parameter :: test_parent1 = child_const1%parent_field1 == 12
  logical, parameter :: test_parent2 = child_const1%parent_field2 == 10.0
  logical, parameter :: test_parent3 = child_const1%parent_field3 .eqv. .false.
  logical, parameter :: test_parent4 = & 
    child_const1%parent_type%parent_field1 == 12
  logical, parameter :: test_parent5 = &
    child_const1%parent_type%parent_field2 == 10.0
  logical, parameter :: test_parent6 = &
    child_const1%parent_type%parent_field3 .eqv. .false.

  type(parent_type), parameter ::parent_const1 = child_const1%parent_type
  logical, parameter :: test_parent7 = parent_const1%parent_field1 == 12
  logical, parameter :: test_parent8 = parent_const1%parent_field2 == 10.0
  logical, parameter :: test_parent9 = &
    parent_const1%parent_field3 .eqv. .false.

  type(child_type), parameter :: child_const2 = &
    child_type( &
      child_field3 = 14, parent_field3 = .true., &
      parent_field1 = 15, child_field1 = 16.6)
  logical, parameter :: test_child4 = child_const2%child_field1 == 16.6
  logical, parameter :: test_child5 = child_const2%child_field2 .eqv. .false.
  logical, parameter :: test_child6 = child_const2%child_field3 == 14
  logical, parameter :: test_parent10 = child_const2%parent_field1 == 15
  logical, parameter :: test_parent11 = child_const2%parent_field2 == 20.0
  logical, parameter :: test_parent12 = child_const2%parent_field3 .eqv. .true.

  type(child_type), parameter :: child_const3 = &
    child_type(parent_type( &
      parent_field2 = 17.7, parent_field3 = .false., parent_field1 = 18), &
        child_field2 = .false., child_field1 = 19.9, child_field3 = 21)
  logical, parameter :: test_child7 = child_const3%parent_field1 == 18
  logical, parameter :: test_child8 = child_const3%parent_field2 == 17.7
  logical, parameter :: test_child9 = child_const3%parent_field3 .eqv. .false.
  logical, parameter :: test_child10 = child_const3%child_field1 == 19.9
  logical, parameter :: test_child11 = child_const3%child_field2 .eqv. .false.
  logical, parameter :: test_child12 = child_const3%child_field3 == 21

  type(child_type), parameter :: child_const4 = &
    child_type(parent_type( &
      parent_field3 = .true., parent_field1 = 22), &
      child_field1 = 23.4, child_field3 = 24)
  logical, parameter :: test_child13 = child_const4%parent_field1 == 22
  logical, parameter :: test_child14 = child_const4%parent_field2 == 20.0
  logical, parameter :: test_child15 = child_const4%parent_field3 .eqv. .true.
  logical, parameter :: test_child16 = child_const4%child_field1 == 23.4
  logical, parameter :: test_child17 = child_const4%child_field2 .eqv. .false.
  logical, parameter :: test_child18 = child_const4%child_field3 == 24

end module m3
