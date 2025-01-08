! RUN: %flang_fc1 -fdebug-unparse-no-sema -fopenmp %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema -fopenmp %s | FileCheck --check-prefix="PARSE-TREE" %s
program main
!CHECK-LABEL: program main
  implicit none

  type ty
     integer :: x
  end type ty
  

!CHECK: !$OMP DECLARE MAPPER (mymapper:ty::mapped) MAP(mapped,mapped%x)
  !$omp declare mapper(mymapper : ty :: mapped) map(mapped, mapped%x)

!PARSE-TREE:      OpenMPDeclareMapperConstruct
!PARSE-TREE:        OmpDeclareMapperSpecifier
!PARSE-TREE:         Name = 'mymapper'
!PARSE-TREE:         TypeSpec -> DerivedTypeSpec
!PARSE-TREE:           Name = 'ty'
!PARSE-TREE:         Name = 'mapped'    
!PARSE-TREE:        OmpMapClause
!PARSE-TREE:          OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'mapped'
!PARSE-TREE:          OmpObject -> Designator -> DataRef -> StructureComponent
!PARSE-TREE:           DataRef -> Name = 'mapped'
!PARSE-TREE:           Name = 'x'  

!CHECK: !$OMP DECLARE MAPPER (ty::mapped) MAP(mapped,mapped%x)
  !$omp declare mapper(ty :: mapped) map(mapped, mapped%x)
  
!PARSE-TREE:      OpenMPDeclareMapperConstruct
!PARSE-TREE:        OmpDeclareMapperSpecifier
!PARSE-TREE:         TypeSpec -> DerivedTypeSpec
!PARSE-TREE:           Name = 'ty'
!PARSE-TREE:         Name = 'mapped'    
!PARSE-TREE:        OmpMapClause
!PARSE-TREE:          OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'mapped'
!PARSE-TREE:          OmpObject -> Designator -> DataRef -> StructureComponent
!PARSE-TREE:           DataRef -> Name = 'mapped'
!PARSE-TREE:           Name = 'x'
  
end program main
!CHECK-LABEL: end program main
