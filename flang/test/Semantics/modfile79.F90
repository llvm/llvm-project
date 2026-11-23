!RUN: rm -rf %t && mkdir -p %t
!RUN: %flang -c -DWHICH=1 -J%t %s && FileCheck %s <%t/modfile79a.mod && %flang -c -fhermetic-module-files -DWHICH=2 -J%t %s && %flang -c -J%t %s && FileCheck %s <%t/modfile79a.mod

!Ensure that writing modfile79c.mod doesn't cause a spurious
!regeneration of modfile79a.mod from its copy in the hermetic
!module file modfile79b.mod.
!CHECK: !mod$ v1 sum:93ec75fe672c5b6c
!CHECK-NEXT: module modfile79a

#if WHICH == 1
module modfile79a
  interface foo
    module procedure foo
  end interface
 contains
  subroutine foo
  end
end
#elif WHICH == 2
module modfile79b
  use modfile79a
  interface bar
    procedure foo
  end interface
end
#else
module modfile79c
  use modfile79b
 contains
  subroutine test
    call bar
  end
end
#endif
