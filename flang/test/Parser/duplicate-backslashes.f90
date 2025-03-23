! RUN: %flang -cpp -E %s -o %t.f90
! RUN: %flang %t.f90 -o %t
! RUN: %t | FileCheck %s

program main
  implicit none

  ! Test single backslash
  write(*, '(A)') "\"   ! Expected single backslash in output
  ! CHECK: \
  ! CHECK-NOT: \\

  ! Test double backslash
  write(*, '(A)') "\\"   ! Expected double backslashes in output
  ! CHECK: \\
  ! CHECK-NOT: \\\\

  ! Test quadruple backslash
  write(*, '(A)') "\\\\"   ! Expected quadruple backslashes in output
  ! CHECK: \\\\
  ! CHECK-NOT: \\\\\\\\

end program main
