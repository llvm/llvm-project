!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=60 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=60 %s | FileCheck --check-prefix="PARSE-TREE" %s

module m
implicit none
integer :: a, b
common /blk/ a

!$omp threadprivate(/blk/, b)

end module

!UNPARSE: MODULE m
!UNPARSE:  IMPLICIT NONE
!UNPARSE:  INTEGER a, b
!UNPARSE:  COMMON /blk/a
!UNPARSE: !$OMP THREADPRIVATE(/blk/, b)
!UNPARSE: END MODULE

!PARSE-TREE: DeclarationConstruct -> SpecificationConstruct -> OpenMPDeclarativeConstruct -> OpenMPThreadprivate -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = threadprivate
!PARSE-TREE: | OmpArgumentList -> OmpArgument -> OmpLocator -> OmpObject -> Name = 'blk'
!PARSE-TREE: | OmpArgument -> OmpLocator -> OmpObject -> Designator -> DataRef -> Name = 'b'
!PARSE-TREE: | OmpClauseList ->
!PARSE-TREE: | Flags = None
