!RUN: %flang -fsyntax-only %s 2>&1 | FileCheck --allow-empty %s
!Ensure no bogus error message about incompatible character length
!CHECK-NOT: error

module m1
  integer :: n = 1
end

module m2
  interface
    module subroutine s(a,b)
      use m1
      character(n) :: a
      character(n) :: b
    end
  end interface
end

submodule(m2) m2s1
 contains
  module procedure s
  end
end
