!RUN: %flang -c -fhermetic-module-files -DWHICH=1 %s && %flang -c -fhermetic-module-files -DWHICH=2 %s && %flang -c -fhermetic-module-files %s && cat modfile78c.mod | FileCheck %s

#if WHICH == 1
module modfile78a
  integer :: global_variable = 0
end
#elif WHICH == 2
module modfile78b
  use modfile78a
 contains
  subroutine test
  end
end
#else
module modfile78c
  use modfile78a
  use modfile78b
end
#endif

!CHECK: module modfile78c
!CHECK: use modfile78a,only:global_variable
!CHECK: use modfile78b,only:test
!CHECK: end
!CHECK: module modfile78a
!CHECK: integer(4)::global_variable
!CHECK: end
!CHECK: module modfile78b
!CHECK: use modfile78a,only:global_variable
!CHECK: contains
!CHECK: subroutine test()
!CHECK: end
!CHECK: end
