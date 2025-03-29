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
    integer, pointer :: ipp
    real, pointer :: rpp(:)
!ERROR: Must be a constant value
    type(parent1) :: tp1 = parent1(3)
!ERROR: Must be a constant value
    type(parent1) :: tp2 = parent1(j)
    type(parent1) :: tp3 = parent1(null())
!PORTABILITY: NULL() with arguments is not standard conforming as the value for allocatable component 'pa'
    type(parent1) :: tp4 = parent1(null(ipp))

!ERROR: Must be a constant value
    type(parent2) :: tp5 = parent2([1.1,2.1,3.1])
!ERROR: Must be a constant value
    type(parent2) :: tp6 = parent2(arr)
    type(parent2) :: tp7 = parent2(null())
!PORTABILITY: NULL() with arguments is not standard conforming as the value for allocatable component 'pa'
    type(parent2) :: tp8 = parent2(null(rpp))
  end subroutine test1

  subroutine test2()
    integer :: j
    real :: arr(5)
    integer, pointer :: ipp
    real, pointer :: rpp(:)
    type(parent1) :: tp1
    type(parent2) :: tp2
    tp1 = parent1(3)
    tp1 = parent1(j)
    tp1 = parent1(null())
!PORTABILITY: NULL() with arguments is not standard conforming as the value for allocatable component 'pa'
    tp1 = parent1(null(ipp))

    tp2 = parent2([1.1,2.1,3.1])
    tp2 = parent2(arr)
    tp2 = parent2(null())
!PORTABILITY: NULL() with arguments is not standard conforming as the value for allocatable component 'pa'
    tp2 = parent2(null(rpp))
  end subroutine test2

  subroutine test3()
    real, pointer :: pp(:)
    type(child) :: tc1 = child(5, parent2(null()))
!PORTABILITY: NULL() with arguments is not standard conforming as the value for allocatable component 'pa'
    type(child) :: tc10 = child(5, parent2(null(pp)))
!ERROR: Must be a constant value
    type(child) :: tc3 = child(5, parent2([1.1,1.2]))
    type(child) :: tc4

    tc4 = child(5, parent2(null()))
    tc4 = child(5, parent2([1.1,1.2]))
  end subroutine test3
end module m
