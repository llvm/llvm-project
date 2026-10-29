!RUN: %python %S/test_errors.py %s %flang_fc1 -fopenmp -Werror -Wstat-kind

program stat_test
  use iso_c_binding
  use iso_fortran_env
  integer :: i, coarray[*]
  integer, allocatable :: a(:)
  type(event_type) :: e[*]
  type(team_type) :: t
  integer(C_INT8_T) :: myint8
  integer(C_INT16_T) :: myint16
 
  !WARNING: The stat-variable should have a decimal exponent range of at least four [-Wstat-kind]
  coarray[1, stat=myint8] = 0
  
  !WARNING: The stat-variable should have a decimal exponent range of at least four [-Wstat-kind]
  sync all(stat=myint8)
  
  !WARNING: The stat-variable should have a decimal exponent range of at least four [-Wstat-kind]
  event wait(e, stat=myint8)
  
  !WARNING: The stat-variable should have a decimal exponent range of at least four [-Wstat-kind]
  allocate(a(3), stat=myint8)
  
  !WARNING: The stat-variable should have a decimal exponent range of at least four [-Wstat-kind]
  deallocate(a, stat=myint8)
  
  !WARNING: The stat-variable should have a decimal exponent range of at least four [-Wstat-kind]
  form team(1, t, stat=myint8)

  !WARNING: The stat-variable should have a decimal exponent range of at least four [-Wstat-kind]
  !$omp parallel do if(coarray[1, stat=myint8] > 0)
  do i = 1, 2
  end do
  !$omp end parallel do
  
  coarray[1, stat=myint16] = 0 ! ok
  sync all(stat=myint16) ! ok
  event wait(e, stat=myint16) ! ok
  allocate(a(3), stat=myint16) ! ok
  deallocate(a, stat=myint16) ! ok
  form team(1, t, stat=myint16) ! ok

end program
