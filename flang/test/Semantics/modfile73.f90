! This test verifies that both invocations produce a consistent order in the
! generated `.mod` file. Previous versions of Flang exhibited non-deterministic
! behavior due to iterating over a set ordered by heap pointers. This issue was
! particularly noticeable when using Flang as a library.

! RUN: rm -rf %t && mkdir -p %t
! RUN: %flang_fc1 \
! RUN:   -fsyntax-only \
! RUN:   -J%t \
! RUN:   %S/Inputs/modfile73-a.f90 \
! RUN:   %S/Inputs/modfile73-b.f90 \
! RUN:   %S/Inputs/modfile73-c.f90
! RUN: %flang_fc1 -fsyntax-only -J%t %s
! RUN: cat %t/modfile73.mod | FileCheck %s

! RUN: rm -rf %t && mkdir -p %t
! RUN: %flang_fc1 \
! RUN:   -fsyntax-only \
! RUN:   -J%t \
! RUN:   %S/Inputs/modfile73-a.f90 \
! RUN:   %S/Inputs/modfile73-b.f90 \
! RUN:   %S/Inputs/modfile73-c.f90 \
! RUN:   %s
! RUN: cat %t/modfile73.mod | FileCheck %s

  use modfile73ba
end  
module modfile73
  use modfile73bb
  use modfile73c 
  CONTAINS
   subroutine init_
  end  
  subroutine delete_
  end  
  subroutine assign_
  end  
  function initialized_ 
  end  
  function same_ 
  end  
  function refcount_ 
  end  
  function id_ 
  end  
  function name_ 
  end  
  subroutine tag_new_object
   end  
end

!      CHECK: !need$ {{.*}} n modfile73bb
! CHECK-NEXT: !need$ {{.*}} n modfile73c
