! Error test -- DO loop uses obsolete loop termination statement
! See R1131 and C1131

! RUN: %f18 -funparse-with-symbols -Mstandard %s 2>&1 | FileCheck %s
! CHECK: A DO loop should terminate with an END DO or CONTINUE

program endDo
  do 10 i = 1, 5
10  print *, "in loop"
end program endDo
