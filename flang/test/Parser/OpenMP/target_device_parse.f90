! REQUIRES: openmp_runtime

! RUN: %flang_fc1 -fdebug-unparse-no-sema %openmp_flags %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree %openmp_flags %s | FileCheck --check-prefix="PARSE-TREE" %s
! Checks the parsing of Openmp 5.0 Target Device constructs
!
PROGRAM main
  USE OMP_LIB
  IMPLICIT NONE
  INTEGER :: X, Y
  INTEGER :: M = 1


!------------------------------------------------------
! Check Device clause with a constant argument
!------------------------------------------------------
!CHECK: !$OMP TARGET DEVICE(1)
!$OMP TARGET DEVICE(1)
  M = M + 1
!CHECK: !$OMP END TARGET
!$OMP END TARGET

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: OmpClauseList -> OmpClause -> Device -> OmpDeviceClause
!PARSE-TREE: Scalar -> Integer -> Expr = '1_4'
!PARSE-TREE: LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: OmpEndBlockDirective
!PARSE-TREE: OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: OmpClauseList ->

!------------------------------------------------------
! Check Device clause with a constant integer expression argument
!------------------------------------------------------
!CHECK: !$OMP TARGET DEVICE(2-1)
!$OMP TARGET DEVICE(2-1)
  M = M + 1
!CHECK: !$OMP END TARGET
!$OMP END TARGET

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: OmpClauseList -> OmpClause -> Device -> OmpDeviceClause
!PARSE-TREE: Scalar -> Integer -> Expr = '1_4'
!PARSE-TREE: Subtract
!PARSE-TREE: Expr = '2_4'
!PARSE-TREE: LiteralConstant -> IntLiteralConstant = '2'
!PARSE-TREE: Expr = '1_4'
!PARSE-TREE: LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: OmpEndBlockDirective
!PARSE-TREE: OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: OmpClauseList ->


!------------------------------------------------------
! Check Device clause with a variable argument
!------------------------------------------------------
!CHECK: !$OMP TARGET DEVICE(X)
!$OMP TARGET DEVICE(X)
  M = M + 1
!CHECK: !$OMP END TARGET
!$OMP END TARGET

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: OmpClauseList -> OmpClause -> Device -> OmpDeviceClause
!PARSE-TREE: Scalar -> Integer -> Expr = 'x'
!PARSE-TREE: Designator -> DataRef -> Name = 'x'
!PARSE-TREE: OmpEndBlockDirective
!PARSE-TREE: OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: OmpClauseList ->


!------------------------------------------------------
! Check Device clause with an variable integer expression
!------------------------------------------------------
!CHECK: !$OMP TARGET DEVICE(X+Y)
!$OMP TARGET DEVICE(X+Y)
  M = M + 1
!CHECK: !$OMP END TARGET
!$OMP END TARGET

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: OmpClauseList -> OmpClause -> Device -> OmpDeviceClause
!PARSE-TREE: Scalar -> Integer -> Expr = 'x+y'
!PARSE-TREE: Add
!PARSE-TREE: Expr = 'x'
!PARSE-TREE: Designator -> DataRef -> Name = 'x'
!PARSE-TREE: Expr = 'y'
!PARSE-TREE: Designator -> DataRef -> Name = 'y'
!PARSE-TREE: OmpEndBlockDirective
!PARSE-TREE: OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: OmpClauseList ->

!------------------------------------------------------
! Check Device Ancestor clause with a constant argument
!------------------------------------------------------
!CHECK: !$OMP TARGET DEVICE(ANCESTOR:1)
!$OMP TARGET DEVICE(ANCESTOR: 1)
  M = M + 1
!CHECK: !$OMP END TARGET
!$OMP END TARGET

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: OmpClauseList -> OmpClause -> Device -> OmpDeviceClause
!PARSE-TREE: DeviceModifier = Ancestor
!PARSE-TREE: Scalar -> Integer -> Expr = '1_4'
!PARSE-TREE: LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: OmpEndBlockDirective
!PARSE-TREE: OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: OmpClauseList ->


!--------------------------------------------------------
! Check Device Devive-Num clause with a constant argument
!--------------------------------------------------------
!CHECK: !$OMP TARGET DEVICE(DEVICE_NUM:2)
!$OMP TARGET DEVICE(DEVICE_NUM: 2)
  M = M + 1
!CHECK: !$OMP END TARGET
!$OMP END TARGET

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: OmpClauseList -> OmpClause -> Device -> OmpDeviceClause
!PARSE-TREE: DeviceModifier = Device_Num
!PARSE-TREE: Scalar -> Integer -> Expr = '2_4'
!PARSE-TREE: LiteralConstant -> IntLiteralConstant = '2'
!PARSE-TREE: OmpEndBlockDirective
!PARSE-TREE: OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: OmpClauseList ->


!-------------------------------------------------------------------
! Check Device Ancestor clause with a variable expression argument
!-------------------------------------------------------------------
!CHECK: !$OMP TARGET DEVICE(ANCESTOR:X+Y)
!$OMP TARGET DEVICE(ANCESTOR: X + Y)
  M = M + 1
!CHECK: !$OMP END TARGET
!$OMP END TARGET

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: OmpClauseList -> OmpClause -> Device -> OmpDeviceClause
!PARSE-TREE: DeviceModifier = Ancestor
!PARSE-TREE: Scalar -> Integer -> Expr = 'x+y'
!PARSE-TREE: Add
!PARSE-TREE: Expr = 'x'
!PARSE-TREE: Designator -> DataRef -> Name = 'x'
!PARSE-TREE: Expr = 'y'
!PARSE-TREE: Designator -> DataRef -> Name = 'y'
!PARSE-TREE: OmpEndBlockDirective
!PARSE-TREE: OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: OmpClauseList ->


!-------------------------------------------------------------------
! Check Device Devive-Num clause with a variable expression argument
!-------------------------------------------------------------------
!CHECK: !$OMP TARGET DEVICE(DEVICE_NUM:X-Y)
!$OMP TARGET DEVICE(DEVICE_NUM: X - Y)
  M = M + 1
!CHECK: !$OMP END TARGET
!$OMP END TARGET

!PARSE-TREE: OmpBeginBlockDirective
!PARSE-TREE: OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE: OmpClauseList -> OmpClause -> Device -> OmpDeviceClause
!PARSE-TREE: DeviceModifier = Device_Num
!PARSE-TREE: Scalar -> Integer -> Expr = 'x-y'
!PARSE-TREE: Subtract
!PARSE-TREE: Expr = 'x'
!PARSE-TREE: Designator -> DataRef -> Name = 'x'
!PARSE-TREE: Expr = 'y'
!PARSE-TREE: Designator -> DataRef -> Name = 'y'
!PARSE-TREE: OmpEndBlockDirective
!PARSE-TREE: OmpBlockDirective -> llvm::omp::Directive = target
END PROGRAM
