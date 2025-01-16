! RUN: %python %S/test_errors.py %s %flang_fc1
!C1118

subroutine test1
  critical
    !ERROR: RETURN statement is not allowed in a CRITICAL construct
    return
  end critical
end subroutine test1

subroutine test2()
  implicit none
  critical
    !ERROR: An image control statement is not allowed in a CRITICAL construct
    SYNC ALL
  end critical
end subroutine test2

subroutine test3()
  use iso_fortran_env, only: team_type
  implicit none
  type(team_type) :: j
  critical
    !ERROR: An image control statement is not allowed in a CRITICAL construct
    sync team (j)
  end critical
end subroutine test3

subroutine test4()
  integer, allocatable, codimension[:] :: ca

  critical
    !ERROR: An image control statement is not allowed in a CRITICAL construct
    allocate(ca[*])
  end critical

  critical
    !ERROR: An image control statement is not allowed in a CRITICAL construct
    deallocate(ca)
  end critical
end subroutine test4

subroutine test5()
  use iso_fortran_env, only: team_type
  implicit none
  type(team_type) :: j
  critical
    change team (j)
    !ERROR: An image control statement is not allowed in a CRITICAL construct
    end team
  end critical
end subroutine test5

subroutine test6
  critical
    critical
    !ERROR: An image control statement is not allowed in a CRITICAL construct
    end critical
  end critical
end subroutine test6

subroutine test7()
  use iso_fortran_env
  type(event_type) :: x[*], y[*]
  critical
    !ERROR: An image control statement is not allowed in a CRITICAL construct
    event post (x)
    !ERROR: An image control statement is not allowed in a CRITICAL construct
    event wait (y)
  end critical
end subroutine test7

subroutine test8()
  use iso_fortran_env
  type(team_type) :: t

  critical
    !ERROR: An image control statement is not allowed in a CRITICAL construct
    form team(1, t)
  end critical
end subroutine test8

subroutine test9()
  use iso_fortran_env
  type(lock_type), save :: l[*]

  critical
    !ERROR: An image control statement is not allowed in a CRITICAL construct
    lock(l)
    !ERROR: An image control statement is not allowed in a CRITICAL construct
    unlock(l)
  end critical
end subroutine test9

subroutine test10()
  use iso_fortran_env
  integer, allocatable, codimension[:] :: ca
  allocate(ca[*])

  critical
    block
      integer, allocatable, codimension[:] :: cb
      cb = ca
    !TODO: Deallocation of this coarray is not currently caught
    end block
  end critical
end subroutine test10

subroutine test11()
  integer, allocatable, codimension[:] :: ca, cb
  critical
    !ERROR: An image control statement is not allowed in a CRITICAL construct
    call move_alloc(cb, ca)
  end critical
end subroutine test11

subroutine test12()
  critical
    !ERROR: An image control statement is not allowed in a CRITICAL construct
    stop
  end critical
end subroutine test12
