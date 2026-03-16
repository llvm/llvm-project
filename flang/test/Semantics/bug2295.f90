!RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
module m
  type t
   contains
    procedure :: s1, s2
    generic :: g => s1, s2
  end type
 contains
  subroutine s1(this, x, j)
    class(t) this
  end
  subroutine s2(this, z, y)
    class(t) this
  end
  subroutine test(that)
    class(t) that
!CHECK: error: No specific subroutine of generic 'g' matches the actual arguments
!CHECK: Specific procedure 's1' does not match the actual arguments because
!CHECK: Argument keyword 'z=' is not recognized for this procedure reference
!CHECK: Specific procedure 's2' does not match the actual arguments because
!CHECK: Keyword argument 'z=' has already been specified positionally (#2) in this procedure reference
    call that%g(1., z=2.)
  end
end
