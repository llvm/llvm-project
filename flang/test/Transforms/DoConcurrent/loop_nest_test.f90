! Tests loop-nest detection algorithm for do-concurrent mapping.

! REQUIRES: asserts

! RUN: %flang_fc1 -emit-hlfir  -fopenmp -fdo-concurrent-to-openmp=host \
! RUN:   -mmlir -debug -mmlir -mlir-disable-threading %s -o - 2> %t.log || true

! RUN: FileCheck %s < %t.log

program main
  implicit none

contains

subroutine foo(n)
  implicit none
  integer :: n, m
  integer :: i, j, k
  integer :: x
  integer, dimension(n) :: a
  integer, dimension(n, n, n) :: b

  ! CHECK: Loop pair starting at location
  ! CHECK: loc("{{.*}}":[[# @LINE + 1]]:{{.*}}) is perfectly nested
  do concurrent(i=1:n, j=1:bar(n*m, n/m))
    a(i) = n
  end do

  ! CHECK: Loop pair starting at location
  ! CHECK: loc("{{.*}}":[[# @LINE + 1]]:{{.*}}) is perfectly nested
  do concurrent(i=bar(n, x):n, j=1:bar(n*m, n/m))
    a(i) = n
  end do

  ! CHECK: Loop pair starting at location
  ! CHECK: loc("{{.*}}":[[# @LINE + 1]]:{{.*}}) is not perfectly nested
  do concurrent(i=bar(n, x):n)
    do concurrent(j=1:bar(n*m, n/m))
      a(i) = n
    end do
  end do

  ! CHECK: Loop pair starting at location
  ! CHECK: loc("{{.*}}":[[# @LINE + 1]]:{{.*}}) is not perfectly nested
  do concurrent(i=1:n)
    x = 10
    do concurrent(j=1:m)
      b(i,j,k) = i * j + k
    end do
  end do

  ! CHECK: Loop pair starting at location
  ! CHECK: loc("{{.*}}":[[# @LINE + 1]]:{{.*}}) is not perfectly nested
  do concurrent(i=1:n)
    do concurrent(j=1:m)
      b(i,j,k) = i * j + k
    end do
    x = 10
  end do

  ! CHECK: Loop pair starting at location
  ! CHECK: loc("{{.*}}":[[# @LINE + 1]]:{{.*}}) is not perfectly nested
  do concurrent(i=1:n)
    do concurrent(j=1:m)
      b(i,j,k) = i * j + k
      x = 10
    end do
  end do

  ! Verify the (i,j) and (j,k) pairs of loops are detected as perfectly nested.
  !
  ! CHECK: Loop pair starting at location
  ! CHECK: loc("{{.*}}":[[# @LINE + 3]]:{{.*}}) is perfectly nested
  ! CHECK: Loop pair starting at location
  ! CHECK: loc("{{.*}}":[[# @LINE + 1]]:{{.*}}) is perfectly nested
  do concurrent(i=bar(n, x):n, j=1:bar(n*m, n/m), k=1:bar(n*m, bar(n*m, n/m)))
    a(i) = n
  end do
end subroutine

pure function bar(n, m)
    implicit none
    integer, intent(in) :: n, m
    integer :: bar

    bar = n + m
end function

end program main
