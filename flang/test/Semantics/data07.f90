! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
module m
 contains
  subroutine s1
    !ERROR: DATA statement initializations affect 'jb(5_8)' more than once
    integer :: ja(10), jb(10)
    data (ja(k),k=1,9,2) / 5*1 / ! ok
    data (ja(k),k=10,2,-2) / 5*2 / ! ok
    data (jb(k),k=1,9,2) / 5*1 / ! ok
    data (jb(k),k=2,10,3) / 3*2 / ! conflict at 5
  end subroutine
end module
