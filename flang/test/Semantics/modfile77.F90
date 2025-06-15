!RUN: %flang -c -fhermetic-module-files -DWHICH=1 %s && %flang -c -fhermetic-module-files -DWHICH=2 %s && %flang -c -fhermetic-module-files %s && cat modfile77c.mod | FileCheck %s

#if WHICH == 1
module modfile77a
  interface gen
    procedure proc
  end interface
 contains
  subroutine proc
    print *, 'ok'
  end
end
#elif WHICH == 2
module modfile77b
  use modfile77a
end
#else
module modfile77c
  use modfile77a
  use modfile77b
end
#endif

!CHECK: module modfile77c
!CHECK: use modfile77a,only:proc
!CHECK: use modfile77a,only:gen
!CHECK: interface gen
!CHECK: end interface
!CHECK: end
!CHECK: module modfile77a
!CHECK: interface gen
!CHECK: procedure::proc
!CHECK: end interface
!CHECK: contains
!CHECK: subroutine proc()
!CHECK: end
!CHECK: end
