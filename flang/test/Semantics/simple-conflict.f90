! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
!
! Verify that PURE and SIMPLE prefix-specs are mutually exclusive

pure simple subroutine ps()
end
! CHECK: error: Attributes 'PURE' and 'SIMPLE' conflict

simple pure function sp()
end
! CHECK: error: Attributes 'PURE' and 'SIMPLE' conflict

