! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s

simple impure subroutine bug()
end subroutine

! CHECK: may not have both the SIMPLE and IMPURE attributes
