! RUN: %python %S/../test_errors.py %s %flang -fopenacc
module acc_common_decl
  implicit none
  integer a
  common /a_common/ a
!$acc declare create (/a_common/)
  data a/42/
end module acc_common_decl

program acc_decl_test
  use acc_common_decl
  implicit none

  a = 1
!$acc update device (/a_common/)
  a = 2
!ERROR: Could not find COMMON block 'a_common_bad' used in OpenACC directive
!$acc update device (/a_common_bad/)
end program
