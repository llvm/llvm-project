!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=60 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=60 %s | FileCheck --check-prefix="PARSE-TREE" %s

module m
implicit none

integer :: x, y(10), z
!$omp groupprivate(x, y) device_type(nohost)
!$omp groupprivate(z)

end module

!UNPARSE: MODULE m
!UNPARSE:  IMPLICIT NONE
!UNPARSE:  INTEGER x, y(10_4), z
!UNPARSE: !$OMP GROUPPRIVATE(x, y) DEVICE_TYPE(NOHOST)
!UNPARSE: !$OMP GROUPPRIVATE(z)
!UNPARSE: END MODULE

!PARSE-TREE: DeclarationConstruct -> SpecificationConstruct -> OpenMPDeclarativeConstruct -> OpenMPGroupprivate -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = groupprivate
!PARSE-TREE: | OmpArgumentList -> OmpArgument -> OmpLocator -> OmpObject -> Designator -> DataRef -> Name = 'x'
!PARSE-TREE: | OmpArgument -> OmpLocator -> OmpObject -> Designator -> DataRef -> Name = 'y'
!PARSE-TREE: | OmpClauseList -> OmpClause -> DeviceType -> OmpDeviceTypeClause -> DeviceTypeDescription = Nohost
!PARSE-TREE: | Flags = None
!PARSE-TREE: DeclarationConstruct -> SpecificationConstruct -> OpenMPDeclarativeConstruct -> OpenMPGroupprivate -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = groupprivate
!PARSE-TREE: | OmpArgumentList -> OmpArgument -> OmpLocator -> OmpObject -> Designator -> DataRef -> Name = 'z'
!PARSE-TREE: | OmpClauseList ->
!PARSE-TREE: | Flags = None
