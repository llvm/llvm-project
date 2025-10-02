! This test is targeting the RewriteArrayElements function within rewrite-parse-tree.cpp. Its important that this behaviour is working as otherwise the OpenMP Lowering of ArrayElements in Reduction Clauses will not function correctly.
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp %s | FileCheck %s --check-prefix=CHECK-TREE
! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s --check-prefix=CHECK-HLFIR
! RUN: %flang_fc1 -fdebug-unparse -fopenmp %s | FileCheck %s --check-prefix=CHECK-UNPARSE

program test
  integer a(2)
  integer b(2)
  integer c(2)
  integer z(10)
  integer :: k = 10

!! When a scalar array element is used, the array element is replaced with a temprorary so it is correctly lowered as an Integer
! CHECK-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'reduction_temp_a(2)=a(2_8)'
! CHECK-TREE-NEXT: | | | Variable = 'reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> Name = 'reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | Expr = 'a(2_8)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> ArrayElement
! CHECK-TREE-NEXT: | | | | | DataRef -> Name = 'a'
! CHECK-TREE-NEXT: | | | | | SectionSubscript -> Integer -> Expr = '2_4'
! CHECK-TREE-NEXT: | | | | | | LiteralConstant -> IntLiteralConstant = '2'
!$omp do reduction (+: a(2))
! CHECK-TREE: | | | | OmpClauseList -> OmpClause -> Reduction -> OmpReductionClause
! CHECK-TREE-NEXT: | | | | | Modifier -> OmpReductionIdentifier -> DefinedOperator -> IntrinsicOperator = Add
! CHECK-TREE-NEXT: | | | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'reduction_temp_a(2)'
! CHECK-HLFIR: omp.wsloop private(@_QFEi_private_i32 %11#0 -> %arg0 : !fir.ref<i32>) reduction(@add_reduction_i32 %15#0 -> %arg1 : !fir.ref<i32>) { 
  do i = 1,2
    a(2) = a(2) + i
! CHECK-TREE: | | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'reduction_temp_a(2)=reduction_temp_a(2)+i'
! CHECK-TREE-NEXT: | | | | | | Variable = 'reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | | | | | Designator -> DataRef -> Name = 'reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | | | | Expr = 'reduction_temp_a(2)+i'
! CHECK-TREE-NEXT: | | | | | | | Add
! CHECK-TREE-NEXT: | | | | | | | | Expr = 'reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | | | | | | | Designator -> DataRef -> Name = 'reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | | | | | | Expr = 'i'
! CHECK-TREE-NEXT: | | | | | | | | | Designator -> DataRef -> Name = 'i'
! CHECK-HLFIR: hlfir.declare %arg0 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-HLFIR-NEXT: hlfir.declare %arg1 {uniq_name = "_QFEreduction_temp_a(2)"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-HLFIR-NEXT: hlfir.assign %arg2 to %33#0 : i32, !fir.ref<i32>
! CHECK-HLFIR-NEXT: fir.load %34#0 : !fir.ref<i32>
! CHECK-HLFIR-NEXT: fir.load %33#0 : !fir.ref<i32>
! CHECK-HLFIR-NEXT: arith.addi %35, %36 : i32
! CHECK-HLFIR-NEXT: hlfir.assign %37 to %34#0 : i32, !fir.ref<i32>
  end do
!$omp end do
! CHECK-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'a(2_8)=reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | Variable = 'a(2_8)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> ArrayElement
! CHECK-TREE-NEXT: | | | | | DataRef -> Name = 'a'
! CHECK-TREE-NEXT: | | | | | SectionSubscript -> Integer -> Expr = '2_4'
! CHECK-TREE-NEXT: | | | | | | LiteralConstant -> IntLiteralConstant = '2'
! CHECK-TREE-NEXT: | | | Expr = 'reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> Name = 'reduction_temp_a(2)'

!! Ensure that consective reduction clauses can be correctly processed in the same block
! CHECK-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'reduction_temp_b(2)=b(2_8)'
! CHECK-TREE-NEXT: | | | Variable = 'reduction_temp_b(2)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> Name = 'reduction_temp_b(2)'
! CHECK-TREE-NEXT: | | | Expr = 'b(2_8)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> ArrayElement
! CHECK-TREE-NEXT: | | | | | DataRef -> Name = 'b'
! CHECK-TREE-NEXT: | | | | | SectionSubscript -> Integer -> Expr = '2_4'
! CHECK-TREE-NEXT: | | | | | | LiteralConstant -> IntLiteralConstant = '2'
!$omp do reduction (+: b(2))
! CHECK-TREE: | | | | OmpClauseList -> OmpClause -> Reduction -> OmpReductionClause
! CHECK-TREE-NEXT: | | | | | Modifier -> OmpReductionIdentifier -> DefinedOperator -> IntrinsicOperator = Add
! CHECK-TREE-NEXT: | | | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'reduction_temp_b(2)'
! CHECK-HLFIR: omp.wsloop private(@_QFEi_private_i32 %11#0 -> %arg0 : !fir.ref<i32>) reduction(@add_reduction_i32 %17#0 -> %arg1 : !fir.ref<i32>) {
  do i = 1,3
    b(2) = b(2) + i
! CHECK-TREE: | | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'reduction_temp_b(2)=reduction_temp_b(2)+i'
! CHECK-TREE-NEXT: | | | | | | Variable = 'reduction_temp_b(2)'
! CHECK-TREE-NEXT: | | | | | | | Designator -> DataRef -> Name = 'reduction_temp_b(2)'
! CHECK-TREE-NEXT: | | | | | | Expr = 'reduction_temp_b(2)+i'
! CHECK-TREE-NEXT: | | | | | | | Add
! CHECK-TREE-NEXT: | | | | | | | | Expr = 'reduction_temp_b(2)'
! CHECK-TREE-NEXT: | | | | | | | | | Designator -> DataRef -> Name = 'reduction_temp_b(2)'
! CHECK-TREE-NEXT: | | | | | | | | Expr = 'i'
! CHECK-TREE-NEXT: | | | | | | | | | Designator -> DataRef -> Name = 'i'
! CHECK-HLFIR: hlfir.declare %arg0 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-HLFIR-NEXT: hlfir.declare %arg1 {uniq_name = "_QFEreduction_temp_b(2)"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-HLFIR-NEXT: hlfir.assign %arg2 to %33#0 : i32, !fir.ref<i32>
! CHECK-HLFIR-NEXT: fir.load %34#0 : !fir.ref<i32>
! CHECK-HLFIR-NEXT: fir.load %33#0 : !fir.ref<i32>
! CHECK-HLFIR-NEXT: arith.addi %35, %36 : i32
! CHECK-HLFIR-NEXT: hlfir.assign %37 to %34#0 : i32, !fir.ref<i32>
  end do
!$omp end do
! CHECK-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'b(2_8)=reduction_temp_b(2)'
! CHECK-TREE-NEXT: | | | Variable = 'b(2_8)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> ArrayElement
! CHECK-TREE-NEXT: | | | | | DataRef -> Name = 'b'
! CHECK-TREE-NEXT: | | | | | SectionSubscript -> Integer -> Expr = '2_4'
! CHECK-TREE-NEXT: | | | | | | LiteralConstant -> IntLiteralConstant = '2'
! CHECK-TREE-NEXT: | | | Expr = 'reduction_temp_b(2)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> Name = 'reduction_temp_b(2)'

!! Ensure that we can reuse the same array element later on. This will use the same symbol as the previous use of a(2) for the temporary value
! CHECK-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'reduction_temp_a(2)=a(2_8)'
! CHECK-TREE-NEXT: | | | Variable = 'reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> Name = 'reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | Expr = 'a(2_8)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> ArrayElement
! CHECK-TREE-NEXT: | | | | | DataRef -> Name = 'a'
! CHECK-TREE-NEXT: | | | | | SectionSubscript -> Integer -> Expr = '2_4'
! CHECK-TREE-NEXT: | | | | | | LiteralConstant -> IntLiteralConstant = '2'
!$omp do reduction (+: a(2))
! CHECK-TREE: | | | | OmpClauseList -> OmpClause -> Reduction -> OmpReductionClause
! CHECK-TREE-NEXT: | | | | | Modifier -> OmpReductionIdentifier -> DefinedOperator -> IntrinsicOperator = Add
! CHECK-TREE-NEXT: | | | | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'reduction_temp_a(2)'
! CHECK-HLFIR: omp.wsloop private(@_QFEi_private_i32 %11#0 -> %arg0 : !fir.ref<i32>) reduction(@add_reduction_i32 %15#0 -> %arg1 : !fir.ref<i32>) {
  do i = 1,4
    a(2) = a(2) + i
! CHECK-TREE: | | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'reduction_temp_a(2)=reduction_temp_a(2)+i'
! CHECK-TREE-NEXT: | | | | | | Variable = 'reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | | | | | Designator -> DataRef -> Name = 'reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | | | | Expr = 'reduction_temp_a(2)+i'
! CHECK-TREE-NEXT: | | | | | | | Add
! CHECK-TREE-NEXT: | | | | | | | | Expr = 'reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | | | | | | | Designator -> DataRef -> Name = 'reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | | | | | | Expr = 'i'
! CHECK-TREE-NEXT: | | | | | | | | | Designator -> DataRef -> Name = 'i'
! CHECK-HLFIR: hlfir.declare %arg0 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-HLFIR-NEXT: hlfir.declare %arg1 {uniq_name = "_QFEreduction_temp_a(2)"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-HLFIR-NEXT: hlfir.assign %arg2 to %33#0 : i32, !fir.ref<i32>
! CHECK-HLFIR-NEXT: fir.load %34#0 : !fir.ref<i32>
! CHECK-HLFIR-NEXT: fir.load %33#0 : !fir.ref<i32>
! CHECK-HLFIR-NEXT: arith.addi %35, %36 : i32
! CHECK-HLFIR-NEXT: hlfir.assign %37 to %34#0 : i32, !fir.ref<i32>
    !! We need to make sure that for the array element that has not been reduced, this does not get replaced with a temp
    a(1) = a(2)
! CHECK-TREE: | | | | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'a(1_8)=reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | | | | Variable = 'a(1_8)'
! CHECK-TREE-NEXT: | | | | | | | Designator -> DataRef -> ArrayElement
! CHECK-TREE-NEXT: | | | | | | | | DataRef -> Name = 'a'
! CHECK-TREE-NEXT: | | | | | | | | SectionSubscript -> Integer -> Expr = '1_4'
! CHECK-TREE-NEXT: | | | | | | | | | LiteralConstant -> IntLiteralConstant = '1'
! CHECK-TREE-NEXT: | | | | | | Expr = 'reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | | | | | Designator -> DataRef -> Name = 'reduction_temp_a(2)'
! CHECK-HLFIR: arith.constant 1 : index
! CHECK-HLFIR-NEXT: hlfir.designate %3#0 (%c1)  : (!fir.ref<!fir.array<2xi32>>, index) -> !fir.ref<i32>
! CHECK-HLFIR-NEXT: hlfir.assign %38 to %39 : i32, !fir.ref<i32>
  end do
!$omp end do
! CHECK-TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> ActionStmt -> AssignmentStmt = 'a(2_8)=reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | Variable = 'a(2_8)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> ArrayElement
! CHECK-TREE-NEXT: | | | | | DataRef -> Name = 'a'
! CHECK-TREE-NEXT: | | | | | SectionSubscript -> Integer -> Expr = '2_4'
! CHECK-TREE-NEXT: | | | | | | LiteralConstant -> IntLiteralConstant = '2'
! CHECK-TREE-NEXT: | | | Expr = 'reduction_temp_a(2)'
! CHECK-TREE-NEXT: | | | | Designator -> DataRef -> Name = 'reduction_temp_a(2)'

!! Array Sections will not get changed
  !$omp parallel do reduction(+:z(1:10:1))
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
! CHECK-HLFIR: omp.wsloop private(@_QFEi_private_i32 %11#0 -> %arg0 : !fir.ref<i32>) reduction(byref @add_reduction_byref_box_10xi32 %34 -> %arg1 : !fir.ref<!fir.box<!fir.array<10xi32>>>) {
  do i=1,10
    k = k + 1
  end do
  !$omp end parallel do

end program test

! CHECK-UNPARSE: PROGRAM TEST
! CHECK-UNPARSE-NEXT:  INTEGER a(2_4)
! CHECK-UNPARSE-NEXT:  INTEGER b(2_4)
! CHECK-UNPARSE-NEXT:  INTEGER c(2_4)
! CHECK-UNPARSE-NEXT:  INTEGER z(10_4)
! CHECK-UNPARSE-NEXT:  INTEGER :: k = 10_4
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
! CHECK-UNPARSE-NEXT: !$OMP PARALLEL DO REDUCTION(+: z(1_4:10_4:1_4))
! CHECK-UNPARSE-NEXT:  DO i=1_4,10_4
! CHECK-UNPARSE-NEXT:    k=k+1_4
! CHECK-UNPARSE-NEXT:  END DO
! CHECK-UNPARSE-NEXT: !$OMP END PARALLEL DO
! CHECK-UNPARSE-NEXT: END PROGRAM TEST
