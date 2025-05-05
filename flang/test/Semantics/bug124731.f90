!RUN: %flang_fc1 -fsyntax-only %s  2>&1 | FileCheck %s --allow-empty
!CHECK-NOT: error:
module m1
  interface
    module subroutine foo
    end
  end interface
  real x
end
module m2
  use m1
end
submodule(m1) sm1
  use m2 ! ok
 contains
  module procedure foo
  end
end
submodule(m1) sm2
 contains
  subroutine bar
    use m2 ! ok
  end
end
