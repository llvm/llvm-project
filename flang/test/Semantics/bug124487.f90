!RUN: %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck --allow-empty %s
!CHECK-NOT: error:
module m
  interface
    module subroutine smp(x)
      character, external :: x
    end
  end interface
end
submodule (m) sm
 contains
  module procedure smp ! crashes here
  end
end
