! RUN: %flang -fsyntax-only -pedantic %s 2>&1 | FileCheck %s

program stat_test
  use iso_c_binding
  use iso_fortran_env
  integer :: coarray[*]
  type(event_type) :: e[*]
  integer(C_INT8_T) :: myint8
  integer(C_INT16_T) :: myint16
 
  ! CHECK: warning:  The stat-variable must have an integer kind of at least 16-bit
  coarray[1, stat=myint8] = 0 ! violates F2023 9.7.4
  
  ! CHECK: warning: The stat-variable must have an integer kind of at least 16-bit
  sync all(stat=myint8) ! violates F2023 9.7.4
  
  ! CHECK: warning: The stat-variable must have an integer kind of at least 16-bit
  event wait(e, stat=myint8) ! violates F2023 9.7.4
  
  ! CHECK-NOT: warning: The stat-variable must have an integer kind of at least 16-bit
  coarray[1, stat=myint16] = 0 ! ok
  
  sync all(stat=myint16) ! ok
  
  event wait(e, stat=myint16) ! ok

end program
