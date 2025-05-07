! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
program p
 contains
! CHECK: error: expected 'END'
! CHECK: in the context: END PROGRAM statement
  continue
end

subroutine s
 contains
! CHECK: error: expected 'END'
! CHECK: in the context: SUBROUTINE subprogram
  continue
end

function f()
 contains
! CHECK: error: expected 'END'
! CHECK: in the context: FUNCTION subprogram
  continue
end

module m
  interface
    module subroutine ms
    end
  end interface
 contains
! CHECK: error: expected 'END'
! CHECK: in the context: END MODULE statement
  continue
end

module m2
 contains
  subroutine m2s
   contains
! CHECK: error: expected 'END'
! CHECK: in the context: SUBROUTINE subprogram
    continue
  end
end

submodule(m) s1
 contains
! CHECK: error: expected 'END'
! CHECK: in the context: END SUBMODULE statement
  continue
end

submodule(m) s2
 contains
  module procedure ms
   contains
! CHECK: error: expected 'END'
! CHECK: in the context: END PROCEDURE statement
    continue
  end
end

! Ensure no error cascade
! CHECK-NOT: error:
