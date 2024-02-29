! RUN: %flang_fc1 -fsyntax-only -pedantic %s  2>&1 | FileCheck %s --allow-empty
! Regression test for bogus use-association name conflict
!   error: Cannot use-associate 's2'; it is already declared in this scope
! CHECK-NOT: error:
module m1
 contains
  subroutine s1
  end
  subroutine s2
  end
end

module m2
  use m1, s1a => s1
  use m1, s2a => s2
 contains
  subroutine s1
  end
  subroutine s2
  end
end
