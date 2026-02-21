!RUN: rm -rf %t && mkdir -p %t
!RUN: %flang_fc1 -fsyntax-only -J%t %s
!RUN: cat %t/modfile82a.mod | FileCheck --check-prefix=CHECK-A %s
!RUN: cat %t/modfile82b.mod | FileCheck --check-prefix=CHECK-B %s
!RUN: cat %t/modfile82c.mod | FileCheck --check-prefix=CHECK-C %s

!CHECK-A: module modfile82a
module modfile82a
  integer a
end

!CHECK-B: !need$ {{[0-9a-f]*}} n modfile82a
!CHECK-B: module modfile82b
module modfile82b
  use modfile82a
  integer b
end

!CHECK-C: !need$ {{[0-9a-f]*}} n modfile82a
!CHECK-C: !need$ {{[0-9a-f]*}} n modfile82b
!CHECK-C: module modfile82c
module modfile82c
  use modfile82a
 contains
   subroutine foo
     use modfile82b, only: b
     print *, a, b
   end
end
