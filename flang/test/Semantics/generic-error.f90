! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
module m
  interface generic
    procedure :: sub1, sub2
  end interface
 contains
  subroutine sub1(x)
  end
  subroutine sub2(j)
  end
end

program test
  use m
!CHECK: error: No specific subroutine of generic 'generic' matches the actual arguments
!CHECK: Specific procedure 'sub1' does not match the actual arguments
!CHECK: Actual argument type 'REAL(8)' is not compatible with dummy argument type 'REAL(4)'
!CHECK: Specific procedure 'sub2' does not match the actual arguments
!CHECK: Actual argument type 'REAL(8)' is not compatible with dummy argument type 'INTEGER(4)'
  call generic(1.d0)
end
