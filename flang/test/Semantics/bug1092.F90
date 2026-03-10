!RUN: rm -rf %t && mkdir -p %t
!RUN: %flang_fc1 -DWHICH=1 -fsyntax-only -J%t %s
!RUN: %flang_fc1 -DWHICH=2 -fsyntax-only -fhermetic-module-files -I%t -J%t %s
!RUN: %flang_fc1 -fsyntax-only -I%t %s 2>&1 | FileCheck --allow-empty %s
!CHECK-NOT: error:

#if WHICH == 1
module bug1092a
  type t
  end type
 contains
  subroutine subr(x)
    type(t) x
  end
end
#elif WHICH == 2
module bug1092b
  use bug1092a, only: subr
end
#else
use bug1092a, only: t
use bug1092b, only: subr
type(t) x
call subr(x)
end
#endif
