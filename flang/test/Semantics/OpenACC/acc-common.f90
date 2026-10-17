! RUN: %python %S/../test_errors.py %s %flang -fopenacc
module acc_common_decl
  implicit none
  integer a
  common /a_common/ a
!$acc declare create (/a_common/)
  data a/42/
end module acc_common_decl

module acc_common_another
  implicit none
  integer c, d
  common /a_common/ c, d
!$acc declare create (/a_common/)
end module acc_common_another

module acc_common_intermediate
  use acc_common_decl
  implicit none
  integer b
  common /b_common/ b
!$acc declare create (/b_common/)
end module acc_common_intermediate

program acc_decl_test
  use acc_common_intermediate
  use acc_common_another
  implicit none

  a = 1
  b = 10
!$acc update device (/a_common/)
  a = 2
!$acc update device (/b_common/)
  b = 20
!$acc update device (/a_common/)
  c = 3
  d = 30
!ERROR: Could not find COMMON block 'a_common_bad' used in OpenACC directive
!$acc update device (/a_common_bad/)
end program

subroutine common_dsa_conflict()
  integer :: a, b
  common /dsa_common/ a, b

  !ERROR: 'dsa_common' appears in more than one data-sharing clause on the same OpenACC directive
  !$acc parallel private(/dsa_common/) firstprivate(/dsa_common/)
  a = b
  !$acc end parallel
end subroutine
