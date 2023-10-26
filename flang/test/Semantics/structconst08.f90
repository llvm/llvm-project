! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
!
! Error tests for structure constructors of derived types with allocatable components

module m
  type parent1
    integer, allocatable :: pa
  end type parent1
  type parent2
    real, allocatable :: pa(:)
  end type parent2
  type child
    integer :: i
    type(parent2) :: ca
  end type

contains
  subroutine test1()
    integer :: j
    real :: arr(5)
!ERROR: Must be a constant value
    type(parent1) :: tp1 = parent1(3)
!ERROR: Must be a constant value
    type(parent1) :: tp2 = parent1(j)
    type(parent1) :: tp3 = parent1(null())

!ERROR: Must be a constant value
    type(parent2) :: tp4 = parent2([1.1,2.1,3.1])
!ERROR: Must be a constant value
    type(parent2) :: tp5 = parent2(arr)
    type(parent2) :: tp6 = parent2(null())
  end subroutine test1

  subroutine test2()
    integer :: j
    real :: arr(5)
    type(parent1) :: tp1
    type(parent2) :: tp2
    tp1 = parent1(3)
    tp1 = parent1(j)
    tp1 = parent1(null())

    tp2 = parent2([1.1,2.1,3.1])
    tp2 = parent2(arr)
    tp2 = parent2(null())
  end subroutine test2

  subroutine test3()
    type(child) :: tc1 = child(5, parent2(null()))
!ERROR: Must be a constant value
    type(child) :: tc2 = child(5, parent2([1.1,1.2]))
    type(child) :: tc3

    tc3 = child(5, parent2(null()))
    tc3 = child(5, parent2([1.1,1.2]))
  end subroutine test3
end module m
