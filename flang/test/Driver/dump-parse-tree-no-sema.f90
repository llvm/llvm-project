!----------
! RUN lines
!----------
! RUN: %flang_fc1 -fdebug-dump-parse-tree %s 2>&1 | FileCheck %s --check-prefix=SEMA_ON
! RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 | FileCheck %s --check-prefix=SEMA_OFF

! SEMA_ON: | | | NamedConstant -> Name = 'i'
! SEMA_ON-NEXT: | | | Constant -> Expr = '1_4'
! SEMA_ON-NEXT: | | | | LiteralConstant -> IntLiteralConstant = '1'

! SEMA_OFF: | | | NamedConstant -> Name = 'i'
! SEMA_OFF-NEXT: | | | Constant -> Expr -> LiteralConstant -> IntLiteralConstant = '1'

parameter(i=1)
integer :: j
end program
