! RUN: %flang_fc1 -fopenmp -fdebug-dump-parse-tree %s | FileCheck %s --check-prefix=PARSE-TREE
! RUN: %flang_fc1 -fopenmp -fdebug-unparse %s | FileCheck %s --check-prefix="UNPARSE"

integer function func(a, b, c)
  integer  :: a, b, c
  func = a + b + c
end function func

subroutine sub(x)
  use iso_c_binding
  integer :: func
  integer :: r
  type(c_ptr) :: x
  integer :: a = 14, b = 7, c = 21

!UNPARSE: !$OMP DISPATCH DEVICE(3_4) NOWAIT NOCONTEXT(.false._4) NOVARIANTS(.true._4)
!UNPARSE:   r=func(a,b,c)
!UNPARSE: !$OMP END DISPATCH

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPDispatchConstruct
!PARSE-TREE: | OmpDirectiveSpecification
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = dispatch
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Device -> OmpDeviceClause
!PARSE-TREE: | | | Scalar -> Integer -> Expr = '3_4'
!PARSE-TREE: | | | | LiteralConstant -> IntLiteralConstant = '3'
!PARSE-TREE: | | OmpClause -> Nowait
!PARSE-TREE: | | OmpClause -> Nocontext -> Scalar -> Logical -> Expr = '.false._4'
!PARSE-TREE: | | | LiteralConstant -> LogicalLiteralConstant
!PARSE-TREE: | | | | bool = 'false'
!PARSE-TREE: | | OmpClause -> Novariants -> Scalar -> Logical -> Expr = '.true._4'
!PARSE-TREE: | | | EQ
!PARSE-TREE: | | | | Expr = '1_4'
!PARSE-TREE: | | | | | LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | | | | Expr = '1_4'
!PARSE-TREE: | | | | | LiteralConstant -> IntLiteralConstant = '1'
!PARSE-TREE: | | Flags = None
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt
![...]
!PARSE-TREE: | OmpDirectiveSpecification
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = dispatch
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | | Flags = None

  !$omp dispatch device(3) nowait nocontext(.false.) novariants(1.eq.1)
  r = func(a, b, c)
  !$omp end dispatch

!! Test the "no end dispatch" option.
!UNPARSE: !$OMP DISPATCH DEVICE(3_4) IS_DEVICE_PTR(x)
!UNPARSE:   r=func(a+1_4,b+2_4,c+3_4)

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPDispatchConstruct
!PARSE-TREE: | OmpDirectiveSpecification
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = dispatch
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Device -> OmpDeviceClause
!PARSE-TREE: | | | Scalar -> Integer -> Expr = '3_4'
!PARSE-TREE: | | | | LiteralConstant -> IntLiteralConstant = '3'
!PARSE-TREE: | | OmpClause -> IsDevicePtr -> OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | Flags = None
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt
!PARSE-TREE-NOT: OmpDirectiveSpecification

  !$omp dispatch device(3) is_device_ptr(x)
  r = func(a+1, b+2, c+3)

end subroutine sub
