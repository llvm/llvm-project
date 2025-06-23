!RUN: %flang_fc1 -E %s | FileCheck --check-prefix=PREPRO %s
!RUN: %flang_fc1 -fdebug-unparse %s | FileCheck --check-prefix=NORMAL %s
!RUN: %flang_fc1 -fopenmp -fdebug-unparse %s | FileCheck --check-prefix=OMP %s

c$   !

C$   
              continue

      k=0                                                               w
      k=0
c$   0                                                                  x
c$   1                                                                  y
c$   2 k=                                                                 z
c$   !                                                                  A
c$   !1                                                                 B
      print *,k
*$1   continue
      end

!PREPRO:!$   &
!PREPRO:              continue
!PREPRO:      k=0
!PREPRO:      k=0
!PREPRO:!$   
!PREPRO:!$   &
!PREPRO:!$   &k=
!PREPRO:!$   &
!PREPRO:!$   &1
!PREPRO:      print *,k
!PREPRO:!$ 1  continue
!PREPRO:      end

!NORMAL:  k=0_4
!NORMAL:  k=0_4
!NORMAL: PRINT *, k
!NORMAL:END PROGRAM

!OMP:  CONTINUE
!OMP:  k=0_4
!OMP:  k=0_4
!OMP:  k=1_4
!OMP: PRINT *, k
!OMP: 1 CONTINUE
!OMP:END PROGRAM
