! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for C1553 and 18.3.4(1)

function func1() result(res) bind(c)
  ! ERROR: BIND(C) function result cannot have ALLOCATABLE or POINTER attribute
  integer, pointer :: res
end

function func2() result(res) bind(c)
  ! ERROR: BIND(C) function result cannot have ALLOCATABLE or POINTER attribute
  integer, allocatable :: res
end

function func3() result(res) bind(c)
  ! ERROR: BIND(C) function result must be scalar
  integer :: res(2)
end

function func4() result(res) bind(c)
  ! ERROR: BIND(C) character function result must have length one
  character(*) :: res
end

function func5(n) result(res) bind(c)
  integer :: n
  ! ERROR: BIND(C) character function result must have length one
  character(n) :: res
end

function func6() result(res) bind(c)
  ! ERROR: BIND(C) character function result must have length one
  character(2) :: res
end

function func7() result(res) bind(c)
  integer, parameter :: n = 1
  character(n) :: res ! OK
end

function func8() result(res) bind(c)
  ! ERROR: BIND(C) function result cannot have ALLOCATABLE or POINTER attribute
  ! ERROR: BIND(C) character function result must have length one
  character(:), pointer :: res
end

function func9() result(res) bind(c)
  ! ERROR: BIND(C) function result cannot be a coarray
  integer :: res[10, *]
end
