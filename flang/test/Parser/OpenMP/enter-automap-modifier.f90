!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=60 %s | FileCheck %s --check-prefix=UNPARSE
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=60 %s | FileCheck %s --check-prefix=PARSE-TREE

program automap
   integer :: x
   !$omp declare target enter(automap: x)
end program

!UNPARSE: PROGRAM AUTOMAP
!UNPARSE:  INTEGER x
!UNPARSE: !$OMP DECLARE_TARGET ENTER(AUTOMAP: x)
!UNPARSE: END PROGRAM

!PARSE-TREE: DeclarationConstruct -> SpecificationConstruct -> OpenMPDeclarativeConstruct -> OpenMPDeclareTargetConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = declare target
!PARSE-TREE: | OmpClauseList -> OmpClause -> Enter -> OmpEnterClause
!PARSE-TREE: | | Modifier -> OmpAutomapModifier -> Value = Automap
!PARSE-TREE: | | OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | Flags = None
