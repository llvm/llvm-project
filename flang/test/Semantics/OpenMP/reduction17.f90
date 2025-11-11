! This test is targeting the RewriteArrayElements function within rewrite-parse-tree.cpp. Its important that this behaviour is working as otherwise the OpenMP Lowering of ArrayElements in Reduction Clauses will not function correctly.
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp %s | FileCheck %s --check-prefix=CHECK-TREE
! RUN: %flang_fc1 -fdebug-unparse -fopenmp %s | FileCheck %s --check-prefix=CHECK-UNPARSE

program test
  integer a(200)
  integer b(200)
  integer c(200)
  integer z(10)
  integer :: k = 10
  integer :: j

!! When a scalar array element is used, the array element is replaced with a temprorary so it is correctly lowered as an Integer
!$omp do reduction (+: a(2))
  do i = 1,2
    a(2) = a(2) + i
  end do
!$omp end do
! CHECK-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'reduction_temp_a(2)=a(2_8)'
! CHECK-TREE-NEXT: | | | Variable = 'reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> Name = 'reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | Expr = 'a(2_8)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> ArrayElement
! CHECK-TREE-NEXT: | | | | | DataRef -> Name = 'a'
! CHECK-TREE-NEXT: | | | | | SectionSubscript -> Integer -> Expr = '2_4'
! CHECK-TREE-NEXT: | | | | | | LiteralConstant -> IntLiteralConstant = '2'
! CHECK-TREE: | | | | OmpClauseList -> OmpClause -> Reduction -> OmpReductionClause
! CHECK-TREE-NEXT: | | | | | Modifier -> OmpReductionIdentifier -> DefinedOperator -> IntrinsicOperator = Add
! CHECK-TREE-NEXT: | | | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'reduction_temp_a(2)'
! CHECK-TREE: | | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'reduction_temp_a(2)=reduction_temp_a(2)+i'
! CHECK-TREE-NEXT: | | | | | | Variable = 'reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | | | | | Designator -> DataRef -> Name = 'reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | | | | Expr = 'reduction_temp_a(2)+i'
! CHECK-TREE-NEXT: | | | | | | | Add
! CHECK-TREE-NEXT: | | | | | | | | Expr = 'reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | | | | | | | Designator -> DataRef -> Name = 'reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | | | | | | Expr = 'i'
! CHECK-TREE-NEXT: | | | | | | | | | Designator -> DataRef -> Name = 'i'
! CHECK-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'a(2_8)=reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | Variable = 'a(2_8)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> ArrayElement
! CHECK-TREE-NEXT: | | | | | DataRef -> Name = 'a'
! CHECK-TREE-NEXT: | | | | | SectionSubscript -> Integer -> Expr = '2_4'
! CHECK-TREE-NEXT: | | | | | | LiteralConstant -> IntLiteralConstant = '2'
! CHECK-TREE-NEXT: | | | Expr = 'reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> Name = 'reduction_temp_a(2)'

!! Ensure that consective reduction clauses can be correctly processed in the same block
!$omp do reduction (+: b(2))
  do i = 1,3
    b(2) = b(2) + i
  end do
!$omp end do
! CHECK-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'reduction_temp_b(2)=b(2_8)'
! CHECK-TREE-NEXT: | | | Variable = 'reduction_temp_b(2)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> Name = 'reduction_temp_b(2)'
! CHECK-TREE-NEXT: | | | Expr = 'b(2_8)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> ArrayElement
! CHECK-TREE-NEXT: | | | | | DataRef -> Name = 'b'
! CHECK-TREE-NEXT: | | | | | SectionSubscript -> Integer -> Expr = '2_4'
! CHECK-TREE-NEXT: | | | | | | LiteralConstant -> IntLiteralConstant = '2'
! CHECK-TREE: | | | | OmpClauseList -> OmpClause -> Reduction -> OmpReductionClause
! CHECK-TREE-NEXT: | | | | | Modifier -> OmpReductionIdentifier -> DefinedOperator -> IntrinsicOperator = Add
! CHECK-TREE-NEXT: | | | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'reduction_temp_b(2)'
! CHECK-TREE: | | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'reduction_temp_b(2)=reduction_temp_b(2)+i'
! CHECK-TREE-NEXT: | | | | | | Variable = 'reduction_temp_b(2)'
! CHECK-TREE-NEXT: | | | | | | | Designator -> DataRef -> Name = 'reduction_temp_b(2)'
! CHECK-TREE-NEXT: | | | | | | Expr = 'reduction_temp_b(2)+i'
! CHECK-TREE-NEXT: | | | | | | | Add
! CHECK-TREE-NEXT: | | | | | | | | Expr = 'reduction_temp_b(2)'
! CHECK-TREE-NEXT: | | | | | | | | | Designator -> DataRef -> Name = 'reduction_temp_b(2)'
! CHECK-TREE-NEXT: | | | | | | | | Expr = 'i'
! CHECK-TREE-NEXT: | | | | | | | | | Designator -> DataRef -> Name = 'i'
! CHECK-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'b(2_8)=reduction_temp_b(2)'
! CHECK-TREE-NEXT: | | | Variable = 'b(2_8)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> ArrayElement
! CHECK-TREE-NEXT: | | | | | DataRef -> Name = 'b'
! CHECK-TREE-NEXT: | | | | | SectionSubscript -> Integer -> Expr = '2_4'
! CHECK-TREE-NEXT: | | | | | | LiteralConstant -> IntLiteralConstant = '2'
! CHECK-TREE-NEXT: | | | Expr = 'reduction_temp_b(2)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> Name = 'reduction_temp_b(2)'

!! Ensure that we can reuse the same array element later on. This will use the same symbol as the previous use of a(2) for the temporary value
!$omp do reduction (+: a(2))
  do i = 1,4
    a(2) = a(2) + i
    a(1) = a(2)
  end do
!$omp end do
! CHECK-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'reduction_temp_a(2)=a(2_8)'
! CHECK-TREE-NEXT: | | | Variable = 'reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> Name = 'reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | Expr = 'a(2_8)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> ArrayElement
! CHECK-TREE-NEXT: | | | | | DataRef -> Name = 'a'
! CHECK-TREE-NEXT: | | | | | SectionSubscript -> Integer -> Expr = '2_4'
! CHECK-TREE-NEXT: | | | | | | LiteralConstant -> IntLiteralConstant = '2'
! CHECK-TREE: | | | | OmpClauseList -> OmpClause -> Reduction -> OmpReductionClause
! CHECK-TREE-NEXT: | | | | | Modifier -> OmpReductionIdentifier -> DefinedOperator -> IntrinsicOperator = Add
! CHECK-TREE-NEXT: | | | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'reduction_temp_a(2)'
! CHECK-TREE: | | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'reduction_temp_a(2)=reduction_temp_a(2)+i'
! CHECK-TREE-NEXT: | | | | | | Variable = 'reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | | | | | Designator -> DataRef -> Name = 'reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | | | | Expr = 'reduction_temp_a(2)+i'
! CHECK-TREE-NEXT: | | | | | | | Add
! CHECK-TREE-NEXT: | | | | | | | | Expr = 'reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | | | | | | | Designator -> DataRef -> Name = 'reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | | | | | | Expr = 'i'
! CHECK-TREE-NEXT: | | | | | | | | | Designator -> DataRef -> Name = 'i'
! CHECK-TREE: | | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'a(1_8)=reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | | | | Variable = 'a(1_8)'
! CHECK-TREE-NEXT: | | | | | | | Designator -> DataRef -> ArrayElement
! CHECK-TREE-NEXT: | | | | | | | | DataRef -> Name = 'a'
! CHECK-TREE-NEXT: | | | | | | | | SectionSubscript -> Integer -> Expr = '1_4'
! CHECK-TREE-NEXT: | | | | | | | | | LiteralConstant -> IntLiteralConstant = '1'
! CHECK-TREE-NEXT: | | | | | | Expr = 'reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | | | | | Designator -> DataRef -> Name = 'reduction_temp_a(2)'
! CHECK-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'a(2_8)=reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | Variable = 'a(2_8)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> ArrayElement
! CHECK-TREE-NEXT: | | | | | DataRef -> Name = 'a'
! CHECK-TREE-NEXT: | | | | | SectionSubscript -> Integer -> Expr = '2_4'
! CHECK-TREE-NEXT: | | | | | | LiteralConstant -> IntLiteralConstant = '2'
! CHECK-TREE-NEXT: | | | Expr = 'reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> Name = 'reduction_temp_a(2)'

!! Check that multiple reductions work correctly
!$omp parallel do reduction (+:b(2), c(2))
  do i=1,10
    b(2) = b(2) + i
    c(2) = c(2) + i
  end do
!$omp end parallel do
! CHECK-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'reduction_temp_b(2)=b(2_8)'
! CHECK-TREE-NEXT: | | | Variable = 'reduction_temp_b(2)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> Name = 'reduction_temp_b(2)'
! CHECK-TREE-NEXT: | | | Expr = 'b(2_8)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> ArrayElement
! CHECK-TREE-NEXT: | | | | | DataRef -> Name = 'b'
! CHECK-TREE-NEXT: | | | | | SectionSubscript -> Integer -> Expr = '2_4'
! CHECK-TREE-NEXT: | | | | | | LiteralConstant -> IntLiteralConstant = '2'
! CHECK-TREE-NEXT: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'reduction_temp_c(2)=c(2_8)'
! CHECK-TREE-NEXT: | | | Variable = 'reduction_temp_c(2)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> Name = 'reduction_temp_c(2)'
! CHECK-TREE-NEXT: | | | Expr = 'c(2_8)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> ArrayElement
! CHECK-TREE-NEXT: | | | | | DataRef -> Name = 'c'
! CHECK-TREE-NEXT: | | | | | SectionSubscript -> Integer -> Expr = '2_4'
! CHECK-TREE-NEXT: | | | | | | LiteralConstant -> IntLiteralConstant = '2'
! CHECK-TREE: | | | | OmpClauseList -> OmpClause -> Reduction -> OmpReductionClause
! CHECK-TREE-NEXT: | | | | | Modifier -> OmpReductionIdentifier -> DefinedOperator -> IntrinsicOperator = Add
! CHECK-TREE-NEXT: | | | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'reduction_temp_b(2)'
! CHECK-TREE-NEXT: | | | | | OmpObject -> Designator -> DataRef -> Name = 'reduction_temp_c(2)'
! CHECK-TREE: ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'reduction_temp_b(2)=reduction_temp_b(2)+i'
! CHECK-TREE-NEXT: | | | | | | Variable = 'reduction_temp_b(2)'
! CHECK-TREE-NEXT: | | | | | | | Designator -> DataRef -> Name = 'reduction_temp_b(2)'
! CHECK-TREE-NEXT: | | | | | | Expr = 'reduction_temp_b(2)+i'
! CHECK-TREE-NEXT: | | | | | | | Add
! CHECK-TREE-NEXT: | | | | | | | | Expr = 'reduction_temp_b(2)'
! CHECK-TREE-NEXT: | | | | | | | | | Designator -> DataRef -> Name = 'reduction_temp_b(2)'
! CHECK-TREE-NEXT: | | | | | | | | Expr = 'i'
! CHECK-TREE-NEXT: | | | | | | | | | Designator -> DataRef -> Name = 'i'
! CHECK-TREE-NEXT: | | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'reduction_temp_c(2)=reduction_temp_c(2)+i'
! CHECK-TREE-NEXT: | | | | | | Variable = 'reduction_temp_c(2)'
! CHECK-TREE-NEXT: | | | | | | | Designator -> DataRef -> Name = 'reduction_temp_c(2)'
! CHECK-TREE-NEXT: | | | | | | Expr = 'reduction_temp_c(2)+i'
! CHECK-TREE-NEXT: | | | | | | | Add
! CHECK-TREE-NEXT: | | | | | | | | Expr = 'reduction_temp_c(2)'
! CHECK-TREE-NEXT: | | | | | | | | | Designator -> DataRef -> Name = 'reduction_temp_c(2)'
! CHECK-TREE-NEXT: | | | | | | | | Expr = 'i'
! CHECK-TREE-NEXT: | | | | | | | | | Designator -> DataRef -> Name = 'i'
! CHECK-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'c(2_8)=reduction_temp_c(2)'
! CHECK-TREE-NEXT: | | | Variable = 'c(2_8)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> ArrayElement
! CHECK-TREE-NEXT: | | | | | DataRef -> Name = 'c'
! CHECK-TREE-NEXT: | | | | | SectionSubscript -> Integer -> Expr = '2_4'
! CHECK-TREE-NEXT: | | | | | | LiteralConstant -> IntLiteralConstant = '2'
! CHECK-TREE-NEXT: | | | Expr = 'reduction_temp_c(2)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> Name = 'reduction_temp_c(2)'
! CHECK-TREE-NEXT: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'b(2_8)=reduction_temp_b(2)'
! CHECK-TREE-NEXT: | | | Variable = 'b(2_8)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> ArrayElement
! CHECK-TREE-NEXT: | | | | | DataRef -> Name = 'b'
! CHECK-TREE-NEXT: | | | | | SectionSubscript -> Integer -> Expr = '2_4'
! CHECK-TREE-NEXT: | | | | | | LiteralConstant -> IntLiteralConstant = '2'
! CHECK-TREE-NEXT: | | | Expr = 'reduction_temp_b(2)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> Name = 'reduction_temp_b(2)'

!! Check that when the identifier for the element comes from a variable, this get replaced
!$omp parallel do reduction (+: c(j))
  do i=1,10
    c(j) = c(j) + i
    c(j) = c(j) - c(k)
  end do
!$omp end parallel do
! CHECK-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'reduction_temp_c(j)=c(int(j,kind=8))'
! CHECK-TREE-NEXT: | | | Variable = 'reduction_temp_c(j)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> Name = 'reduction_temp_c(j)'
! CHECK-TREE-NEXT: | | | Expr = 'c(int(j,kind=8))'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> ArrayElement
! CHECK-TREE-NEXT: | | | | | DataRef -> Name = 'c'
! CHECK-TREE-NEXT: | | | | | SectionSubscript -> Integer -> Expr = 'j'
! CHECK-TREE-NEXT: | | | | | | Designator -> DataRef -> Name = 'j'
! CHECK-TREE: | | | | OmpClauseList -> OmpClause -> Reduction -> OmpReductionClause
! CHECK-TREE-NEXT: | | | | | Modifier -> OmpReductionIdentifier -> DefinedOperator -> IntrinsicOperator = Add
! CHECK-TREE-NEXT: | | | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'reduction_temp_c(j)'
! CHECK-TREE: | | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'reduction_temp_c(j)=reduction_temp_c(j)+i'
! CHECK-TREE-NEXT: | | | | | | Variable = 'reduction_temp_c(j)'
! CHECK-TREE-NEXT: | | | | | | | Designator -> DataRef -> Name = 'reduction_temp_c(j)'
! CHECK-TREE-NEXT: | | | | | | Expr = 'reduction_temp_c(j)+i'
! CHECK-TREE-NEXT: | | | | | | | Add
! CHECK-TREE-NEXT: | | | | | | | | Expr = 'reduction_temp_c(j)'
! CHECK-TREE-NEXT: | | | | | | | | | Designator -> DataRef -> Name = 'reduction_temp_c(j)'
! CHECK-TREE-NEXT: | | | | | | | | Expr = 'i'
! CHECK-TREE-NEXT: | | | | | | | | | Designator -> DataRef -> Name = 'i'
! CHECK-TREE-NEXT: | | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'reduction_temp_c(j)=reduction_temp_c(j)-c(int(k,kind=8))'
! CHECK-TREE-NEXT: | | | | | | Variable = 'reduction_temp_c(j)'
! CHECK-TREE-NEXT: | | | | | | | Designator -> DataRef -> Name = 'reduction_temp_c(j)'
! CHECK-TREE-NEXT: | | | | | | Expr = 'reduction_temp_c(j)-c(int(k,kind=8))'
! CHECK-TREE-NEXT: | | | | | | | Subtract
! CHECK-TREE-NEXT: | | | | | | | | Expr = 'reduction_temp_c(j)'
! CHECK-TREE-NEXT: | | | | | | | | | Designator -> DataRef -> Name = 'reduction_temp_c(j)'
! CHECK-TREE-NEXT: | | | | | | | | Expr = 'c(int(k,kind=8))'
! CHECK-TREE-NEXT: | | | | | | | | | Designator -> DataRef -> ArrayElement
! CHECK-TREE-NEXT: | | | | | | | | | | DataRef -> Name = 'c'
! CHECK-TREE-NEXT: | | | | | | | | | | SectionSubscript -> Integer -> Expr = 'k'
! CHECK-TREE-NEXT: | | | | | | | | | | | Designator -> DataRef -> Name = 'k'
! CHECK-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'c(int(j,kind=8))=reduction_temp_c(j)'
! CHECK-TREE-NEXT: | | | Variable = 'c(int(j,kind=8))'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> ArrayElement
! CHECK-TREE-NEXT: | | | | | DataRef -> Name = 'c'
! CHECK-TREE-NEXT: | | | | | SectionSubscript -> Integer -> Expr = 'j'
! CHECK-TREE-NEXT: | | | | | | Designator -> DataRef -> Name = 'j'
! CHECK-TREE-NEXT: | | | Expr = 'reduction_temp_c(j)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> Name = 'reduction_temp_c(j)'

!! Array Sections will not get changed
  !$omp parallel do reduction(+:z(1:10:1))
  do i=1,10
  end do
  !$omp end parallel do
! CHECK-TREE: | | | | OmpClauseList -> OmpClause -> Reduction -> OmpReductionClause
! CHECK-TREE-NEXT: | | | | | Modifier -> OmpReductionIdentifier -> DefinedOperator -> IntrinsicOperator = Add
! CHECK-TREE-NEXT: | | | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> ArrayElement
! CHECK-TREE-NEXT: | | | | | | DataRef -> Name = 'z'
! CHECK-TREE-NEXT: | | | | | | SectionSubscript -> SubscriptTriplet
! CHECK-TREE-NEXT: | | | | | | | Scalar -> Integer -> Expr = '1_4'
! CHECK-TREE-NEXT: | | | | | | | | LiteralConstant -> IntLiteralConstant = '1'
! CHECK-TREE-NEXT: | | | | | | | Scalar -> Integer -> Expr = '10_4'
! CHECK-TREE-NEXT: | | | | | | | | LiteralConstant -> IntLiteralConstant = '10'
! CHECK-TREE-NEXT: | | | | | | | Scalar -> Integer -> Expr = '1_4'
! CHECK-TREE-NEXT: | | | | | | | | LiteralConstant -> IntLiteralConstant = '1'
end program test

! CHECK-UNPARSE: PROGRAM TEST
! CHECK-UNPARSE-NEXT:  INTEGER a(200_4)
! CHECK-UNPARSE-NEXT:  INTEGER b(200_4)
! CHECK-UNPARSE-NEXT:  INTEGER c(200_4)
! CHECK-UNPARSE-NEXT:  INTEGER z(10_4)
! CHECK-UNPARSE-NEXT:  INTEGER :: k = 10_4
! CHECK-UNPARSE-NEXT:  INTEGER j
! CHECK-UNPARSE-NEXT:   reduction_temp_a(2)=a(2_8)
! CHECK-UNPARSE-NEXT: !$OMP DO REDUCTION(+: reduction_temp_a(2))
! CHECK-UNPARSE-NEXT:  DO i=1_4,2_4
! CHECK-UNPARSE-NEXT:    reduction_temp_a(2)=reduction_temp_a(2)+i
! CHECK-UNPARSE-NEXT:  END DO
! CHECK-UNPARSE-NEXT: !$OMP END DO
! CHECK-UNPARSE-NEXT:   a(2_8)=reduction_temp_a(2)
! CHECK-UNPARSE-NEXT:   reduction_temp_b(2)=b(2_8)
! CHECK-UNPARSE-NEXT: !$OMP DO REDUCTION(+: reduction_temp_b(2))
! CHECK-UNPARSE-NEXT:  DO i=1_4,3_4
! CHECK-UNPARSE-NEXT:    reduction_temp_b(2)=reduction_temp_b(2)+i
! CHECK-UNPARSE-NEXT:  END DO
! CHECK-UNPARSE-NEXT: !$OMP END DO
! CHECK-UNPARSE-NEXT:   b(2_8)=reduction_temp_b(2)
! CHECK-UNPARSE-NEXT:   reduction_temp_a(2)=a(2_8)
! CHECK-UNPARSE-NEXT: !$OMP DO REDUCTION(+: reduction_temp_a(2))
! CHECK-UNPARSE-NEXT:  DO i=1_4,4_4
! CHECK-UNPARSE-NEXT:    reduction_temp_a(2)=reduction_temp_a(2)+i
! CHECK-UNPARSE-NEXT:    a(1_8)=reduction_temp_a(2)
! CHECK-UNPARSE-NEXT:  END DO
! CHECK-UNPARSE-NEXT: !$OMP END DO
! CHECK-UNPARSE-NEXT:   a(2_8)=reduction_temp_a(2)
! CHECK-UNPARSE-NEXT:   reduction_temp_b(2)=b(2_8)
! CHECK-UNPARSE-NEXT:   reduction_temp_c(2)=c(2_8)
! CHECK-UNPARSE-NEXT: !$OMP PARALLEL DO REDUCTION(+: reduction_temp_b(2),reduction_temp_c(2))
! CHECK-UNPARSE-NEXT:  DO i=1_4,10_4
! CHECK-UNPARSE-NEXT:    reduction_temp_b(2)=reduction_temp_b(2)+i
! CHECK-UNPARSE-NEXT:    reduction_temp_c(2)=reduction_temp_c(2)+i
! CHECK-UNPARSE-NEXT:  END DO
! CHECK-UNPARSE-NEXT: !$OMP END PARALLEL DO
! CHECK-UNPARSE-NEXT:   c(2_8)=reduction_temp_c(2)
! CHECK-UNPARSE-NEXT:   b(2_8)=reduction_temp_b(2)
! CHECK-UNPARSE-NEXT:   reduction_temp_c(j)=c(int(j,kind=8))
! CHECK-UNPARSE-NEXT: !$OMP PARALLEL DO REDUCTION(+: reduction_temp_c(j))
! CHECK-UNPARSE-NEXT:  DO i=1_4,10_4
! CHECK-UNPARSE-NEXT:    reduction_temp_c(j)=reduction_temp_c(j)+i
! CHECK-UNPARSE-NEXT:    reduction_temp_c(j)=reduction_temp_c(j)-c(int(k,kind=8))
! CHECK-UNPARSE-NEXT:  END DO
! CHECK-UNPARSE-NEXT: !$OMP END PARALLEL DO
! CHECK-UNPARSE-NEXT:   c(int(j,kind=8))=reduction_temp_c(j)
! CHECK-UNPARSE-NEXT: !$OMP PARALLEL DO REDUCTION(+: z(1_4:10_4:1_4))
! CHECK-UNPARSE-NEXT:  DO i=1_4,10_4
! CHECK-UNPARSE-NEXT:  END DO
! CHECK-UNPARSE-NEXT: !$OMP END PARALLEL DO
! CHECK-UNPARSE-NEXT: END PROGRAM TEST
