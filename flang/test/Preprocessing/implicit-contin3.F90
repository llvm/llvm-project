! RUN: not %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
! Test implicit continuation for possible function-like macro calls only
#define flm(x) x
call notamacro(3
)
!CHECK: error: Unmatched '('
!CHECK: error: Unmatched ')'
