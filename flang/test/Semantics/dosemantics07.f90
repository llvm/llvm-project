! RUN: %S/test_errors.sh %s %t %f18
!C1132
! If the do-stmt is a nonlabel-do-stmt, the corresponding end-do shall be an
! end-do-stmt.
subroutine s1()
  do while (.true.)
    print *, "Hello"
  continue
!ERROR: expected 'END DO'
end subroutine s1
