! RUN: not %f18 -funparse-with-symbols %s 2>&1 | FileCheck %s
! CHECK: expected end of statement

subroutine s
  do 10 i = 1, 10
5  end do
10 end do
end subroutine s
