! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
program main
  interface generic
    subroutine sub1(j, k)
      integer(1) j
      integer k
      !dir$ ignore_tkr(kr) k
    end
    subroutine sub2(j, k)
      integer(2) j
      integer k
      !dir$ ignore_tkr(kr) k
    end
    subroutine sub4(j, k)
      integer(4) j
      integer k
      !dir$ ignore_tkr(kr) k
    end
  end interface
!CHECK: CALL sub1(1_1,1_1)
  call generic(1_1,1_1)
!CHECK: CALL sub1(1_1,1_2)
  call generic(1_1,1_2)
!CHECK: CALL sub1(1_1,[INTEGER(1)::1_1])
  call generic(1_1,[1_1])
!CHECK: CALL sub2(1_2,1_1)
  call generic(1_2,1_1)
!CHECK: CALL sub2(1_2,1_2)
  call generic(1_2,1_2)
!CHECK: CALL sub2(1_2,[INTEGER(1)::1_1])
  call generic(1_2,[1_1])
!CHECK: CALL sub4(1_4,1_1)
  call generic(1_4,1_1)
!CHECK: CALL sub4(1_4,1_2)
  call generic(1_4,1_2)
!CHECK: CALL sub4(1_4,[INTEGER(1)::1_1])
  call generic(1_4,[1_1])
end
