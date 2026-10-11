! RUN: %flang_fc1 -fdebug-unparse-no-sema %s | FileCheck %s
! RUN: %flang_fc1 -ffixed-form -fdebug-unparse-no-sema %s | FileCheck %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree %s | FileCheck %s --check-prefix=DUMP

      module protected_target_syntax
        integer, pointer, protected_target :: p
        integer, pointer :: q, r
        protected_target q
        PROTECTED_TARGET :: r
        integer, pointer, protected :: ordinary
        type t
          integer, protected_target, pointer :: c
        end type
      end module

! CHECK: MODULE protected_target_syntax
! CHECK: INTEGER, POINTER, PROTECTED_TARGET :: p
! CHECK: PROTECTED_TARGET :: q
! CHECK-NEXT: PROTECTED_TARGET :: r
! CHECK-NEXT: INTEGER, POINTER, PROTECTED :: ordinary
! CHECK: INTEGER, PROTECTED_TARGET, POINTER :: c
! DUMP: ProtectedTarget
! DUMP: ProtectedTargetStmt
! DUMP: ProtectedTargetStmt
! DUMP: Protected{{$}}
! DUMP: ProtectedTarget
