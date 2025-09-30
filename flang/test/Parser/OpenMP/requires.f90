!RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=50 %s | FileCheck --ignore-case --check-prefix="UNPARSE" %s
!RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=50 %s | FileCheck --check-prefix="PARSE-TREE" %s

!$omp requires atomic_default_mem_order(seq_cst)

!UNPARSE: !$OMP REQUIRES ATOMIC_DEFAULT_MEM_ORDER(SEQ_CST)

!PARSE-TREE: OpenMPDeclarativeConstruct -> OpenMPRequiresConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = requires
!PARSE-TREE: | OmpClauseList -> OmpClause -> AtomicDefaultMemOrder -> OmpAtomicDefaultMemOrderClause -> OmpMemoryOrderType = Seq_Cst
!PARSE-TREE: | Flags = None

!$omp requires unified_shared_memory unified_address

!UNPARSE: !$OMP REQUIRES UNIFIED_SHARED_MEMORY UNIFIED_ADDRESS

!PARSE-TREE: OpenMPDeclarativeConstruct -> OpenMPRequiresConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = requires
!PARSE-TREE: | OmpClauseList -> OmpClause -> UnifiedSharedMemory
!PARSE-TREE: | OmpClause -> UnifiedAddress
!PARSE-TREE: | Flags = None

!$omp requires dynamic_allocators reverse_offload

!UNPARSE: !$OMP REQUIRES DYNAMIC_ALLOCATORS REVERSE_OFFLOAD

!PARSE-TREE: OpenMPDeclarativeConstruct -> OpenMPRequiresConstruct -> OmpDirectiveSpecification
!PARSE-TREE: | OmpDirectiveName -> llvm::omp::Directive = requires
!PARSE-TREE: | OmpClauseList -> OmpClause -> DynamicAllocators
!PARSE-TREE: | OmpClause -> ReverseOffload
!PARSE-TREE: | Flags = None

end
