!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=60 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=60 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine f00(a, b)
  integer :: a, b
  integer :: x
  !$omp atomic update compare
  if (x < a) x = b
end

!UNPARSE: SUBROUTINE f00 (a, b)
!UNPARSE:  INTEGER a, b
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP ATOMIC UPDATE COMPARE
!UNPARSE:  IF (x<a)  x=b
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPAtomicConstruct
!PARSE-TREE: | OmpDirectiveSpecification
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = atomic
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Update ->
!PARSE-TREE: | | OmpClause -> Compare
!PARSE-TREE: | | Flags = None
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> IfStmt
!PARSE-TREE: | | | Scalar -> Logical -> Expr = 'x<a'
!PARSE-TREE: | | | | LT
!PARSE-TREE: | | | | | Expr = 'x'
!PARSE-TREE: | | | | | | Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | | | Expr = 'a'
!PARSE-TREE: | | | | | | Designator -> DataRef -> Name = 'a'
!PARSE-TREE: | | | ActionStmt -> AssignmentStmt = 'x=b'
!PARSE-TREE: | | | | Variable = 'x'
!PARSE-TREE: | | | | | Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | | Expr = 'b'
!PARSE-TREE: | | | | | Designator -> DataRef -> Name = 'b'

subroutine f01(a, b)
  integer :: a, b
  integer :: x
  !$omp atomic update compare
  if (x < a) then
    x = b
  endif
end

!UNPARSE: SUBROUTINE f01 (a, b)
!UNPARSE:  INTEGER a, b
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP ATOMIC UPDATE COMPARE
!UNPARSE:  IF (x<a) THEN
!UNPARSE:    x=b
!UNPARSE:  END IF
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPAtomicConstruct
!PARSE-TREE: | OmpDirectiveSpecification
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = atomic
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Update ->
!PARSE-TREE: | | OmpClause -> Compare
!PARSE-TREE: | | Flags = None
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> IfConstruct
!PARSE-TREE: | | | IfThenStmt
!PARSE-TREE: | | | | Scalar -> Logical -> Expr = 'x<a'
!PARSE-TREE: | | | | | LT
!PARSE-TREE: | | | | | | Expr = 'x'
!PARSE-TREE: | | | | | | | Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | | | | Expr = 'a'
!PARSE-TREE: | | | | | | | Designator -> DataRef -> Name = 'a'
!PARSE-TREE: | | | Block
!PARSE-TREE: | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'x=b'
!PARSE-TREE: | | | | | Variable = 'x'
!PARSE-TREE: | | | | | | Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | | | Expr = 'b'
!PARSE-TREE: | | | | | | Designator -> DataRef -> Name = 'b'
!PARSE-TREE: | | | EndIfStmt ->

subroutine f02(a, b)
  integer :: a, b
  integer :: x
  logical :: c
  c = x < a
  !$omp atomic update compare
  if (c) then
    x = b
  endif
end

!UNPARSE: SUBROUTINE f02 (a, b)
!UNPARSE:  INTEGER a, b
!UNPARSE:  INTEGER x
!UNPARSE:  LOGICAL c
!UNPARSE:   c=x<a
!UNPARSE: !$OMP ATOMIC UPDATE COMPARE
!UNPARSE:  IF (c) THEN
!UNPARSE:    x=b
!UNPARSE:  END IF
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'c=x<a'
!PARSE-TREE: | Variable = 'c'
!PARSE-TREE: | | Designator -> DataRef -> Name = 'c'
!PARSE-TREE: | Expr = 'x<a'
!PARSE-TREE: | | LT
!PARSE-TREE: | | | Expr = 'x'
!PARSE-TREE: | | | | Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | Expr = 'a'
!PARSE-TREE: | | | | Designator -> DataRef -> Name = 'a'
!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPAtomicConstruct
!PARSE-TREE: | OmpDirectiveSpecification
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = atomic
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Update ->
!PARSE-TREE: | | OmpClause -> Compare
!PARSE-TREE: | | Flags = None
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> IfConstruct
!PARSE-TREE: | | | IfThenStmt
!PARSE-TREE: | | | | Scalar -> Logical -> Expr = 'c'
!PARSE-TREE: | | | | | Designator -> DataRef -> Name = 'c'
!PARSE-TREE: | | | Block
!PARSE-TREE: | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'x=b'
!PARSE-TREE: | | | | | Variable = 'x'
!PARSE-TREE: | | | | | | Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | | | Expr = 'b'
!PARSE-TREE: | | | | | | Designator -> DataRef -> Name = 'b'
!PARSE-TREE: | | | EndIfStmt ->

subroutine g00(a, b)
  integer :: a, b
  integer :: x, v
  !$omp atomic update capture compare
  v = x
  if (x < a) x = b
  !$omp end atomic
end

!UNPARSE: SUBROUTINE g00 (a, b)
!UNPARSE:  INTEGER a, b
!UNPARSE:  INTEGER x, v
!UNPARSE: !$OMP ATOMIC UPDATE CAPTURE COMPARE
!UNPARSE:   v=x
!UNPARSE:  IF (x<a)  x=b
!UNPARSE: !$OMP END ATOMIC
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPAtomicConstruct
!PARSE-TREE: | OmpDirectiveSpecification
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = atomic
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Update ->
!PARSE-TREE: | | OmpClause -> Capture
!PARSE-TREE: | | OmpClause -> Compare
!PARSE-TREE: | | Flags = None
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'v=x'
!PARSE-TREE: | | | Variable = 'v'
!PARSE-TREE: | | | | Designator -> DataRef -> Name = 'v'
!PARSE-TREE: | | | Expr = 'x'
!PARSE-TREE: | | | | Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> IfStmt
!PARSE-TREE: | | | Scalar -> Logical -> Expr = 'x<a'
!PARSE-TREE: | | | | LT
!PARSE-TREE: | | | | | Expr = 'x'
!PARSE-TREE: | | | | | | Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | | | Expr = 'a'
!PARSE-TREE: | | | | | | Designator -> DataRef -> Name = 'a'
!PARSE-TREE: | | | ActionStmt -> AssignmentStmt = 'x=b'
!PARSE-TREE: | | | | Variable = 'x'
!PARSE-TREE: | | | | | Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | | Expr = 'b'
!PARSE-TREE: | | | | | Designator -> DataRef -> Name = 'b'
!PARSE-TREE: | OmpDirectiveSpecification
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = atomic
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | | Flags = None

subroutine g01(a, b)
  integer :: a, b
  integer :: x, v
  !$omp atomic update capture compare
  v = x
  if (x < a) then
    x = b
  endif
  !$omp end atomic
end

!UNPARSE: SUBROUTINE g01 (a, b)
!UNPARSE:  INTEGER a, b
!UNPARSE:  INTEGER x, v
!UNPARSE: !$OMP ATOMIC UPDATE CAPTURE COMPARE
!UNPARSE:   v=x
!UNPARSE:  IF (x<a) THEN
!UNPARSE:    x=b
!UNPARSE:  END IF
!UNPARSE: !$OMP END ATOMIC
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPAtomicConstruct
!PARSE-TREE: | OmpDirectiveSpecification
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = atomic
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Update ->
!PARSE-TREE: | | OmpClause -> Capture
!PARSE-TREE: | | OmpClause -> Compare
!PARSE-TREE: | | Flags = None
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'v=x'
!PARSE-TREE: | | | Variable = 'v'
!PARSE-TREE: | | | | Designator -> DataRef -> Name = 'v'
!PARSE-TREE: | | | Expr = 'x'
!PARSE-TREE: | | | | Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> IfConstruct
!PARSE-TREE: | | | IfThenStmt
!PARSE-TREE: | | | | Scalar -> Logical -> Expr = 'x<a'
!PARSE-TREE: | | | | | LT
!PARSE-TREE: | | | | | | Expr = 'x'
!PARSE-TREE: | | | | | | | Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | | | | Expr = 'a'
!PARSE-TREE: | | | | | | | Designator -> DataRef -> Name = 'a'
!PARSE-TREE: | | | Block
!PARSE-TREE: | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'x=b'
!PARSE-TREE: | | | | | Variable = 'x'
!PARSE-TREE: | | | | | | Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | | | Expr = 'b'
!PARSE-TREE: | | | | | | Designator -> DataRef -> Name = 'b'
!PARSE-TREE: | | | EndIfStmt ->
!PARSE-TREE: | OmpDirectiveSpecification
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = atomic
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | | Flags = None

subroutine g02(a, b)
  integer :: a, b
  integer :: x, v
  !$omp atomic update capture compare
  if (x < a) then
    x = b
  else
    v = x
  endif
  !$omp end atomic
end

!UNPARSE: SUBROUTINE g02 (a, b)
!UNPARSE:  INTEGER a, b
!UNPARSE:  INTEGER x, v
!UNPARSE: !$OMP ATOMIC UPDATE CAPTURE COMPARE
!UNPARSE:  IF (x<a) THEN
!UNPARSE:    x=b
!UNPARSE:  ELSE
!UNPARSE:    v=x
!UNPARSE:  END IF
!UNPARSE: !$OMP END ATOMIC
!UNPARSE: END SUBROUTINE

!PARSE-TREE: ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPAtomicConstruct
!PARSE-TREE: | OmpDirectiveSpecification
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = atomic
!PARSE-TREE: | | OmpClauseList -> OmpClause -> Update ->
!PARSE-TREE: | | OmpClause -> Capture
!PARSE-TREE: | | OmpClause -> Compare
!PARSE-TREE: | | Flags = None
!PARSE-TREE: | Block
!PARSE-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> IfConstruct
!PARSE-TREE: | | | IfThenStmt
!PARSE-TREE: | | | | Scalar -> Logical -> Expr = 'x<a'
!PARSE-TREE: | | | | | LT
!PARSE-TREE: | | | | | | Expr = 'x'
!PARSE-TREE: | | | | | | | Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | | | | Expr = 'a'
!PARSE-TREE: | | | | | | | Designator -> DataRef -> Name = 'a'
!PARSE-TREE: | | | Block
!PARSE-TREE: | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'x=b'
!PARSE-TREE: | | | | | Variable = 'x'
!PARSE-TREE: | | | | | | Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | | | Expr = 'b'
!PARSE-TREE: | | | | | | Designator -> DataRef -> Name = 'b'
!PARSE-TREE: | | | ElseBlock
!PARSE-TREE: | | | | ElseStmt ->
!PARSE-TREE: | | | | Block
!PARSE-TREE: | | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'v=x'
!PARSE-TREE: | | | | | | Variable = 'v'
!PARSE-TREE: | | | | | | | Designator -> DataRef -> Name = 'v'
!PARSE-TREE: | | | | | | Expr = 'x'
!PARSE-TREE: | | | | | | | Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | | | EndIfStmt ->
!PARSE-TREE: | OmpDirectiveSpecification
!PARSE-TREE: | | OmpDirectiveName -> llvm::omp::Directive = atomic
!PARSE-TREE: | | OmpClauseList ->
!PARSE-TREE: | | Flags = None
