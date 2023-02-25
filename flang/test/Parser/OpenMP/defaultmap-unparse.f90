! RUN: %flang_fc1 -fdebug-unparse-no-sema -fopenmp %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema -fopenmp %s | FileCheck --check-prefix="PARSE-TREE" %s
program main
!CHECK-LABEL: program main
  implicit none

  integer, parameter :: N = 256

  type data01
   integer :: a
   integer :: arr(N)
  end type

  real :: arrA(N), arrB(N)
  integer, target :: arrC(N)
  type(data01) :: data01_a  
  integer, allocatable :: alloc_arr(:)
  integer, pointer :: ptrArr(:)  

  arrA = 1.414
  arrB = 3.14
  arrC = -1
  data01_a%a = -1
  data01_arr = -1
  allocate(alloc_arr(N))
  alloc_arr = -1


!CHECK: !$omp target defaultmap(tofrom:scalar)  
  !$omp target defaultmap(tofrom:scalar) 
  do i = 1, N
   a = 3.14
  enddo
!CHECK: !$omp end target
  !$omp end target

!PARSE-TREE:      OmpBeginBlockDirective
!PARSE-TREE:        OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE:        OmpClauseList -> OmpClause -> Defaultmap -> OmpDefaultmapClause
!PARSE-TREE:          ImplicitBehavior = Tofrom
!PARSE-TREE:          VariableCategory = Scalar

!CHECK: !$omp target defaultmap(alloc:scalar)
  !$omp target defaultmap(alloc:scalar)
   a = 4.56
!CHECK: !$omp end target
  !$omp end target

!PARSE-TREE:      OmpBeginBlockDirective
!PARSE-TREE:        OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE:        OmpClauseList -> OmpClause -> Defaultmap -> OmpDefaultmapClause
!PARSE-TREE:          ImplicitBehavior = Alloc
!PARSE-TREE:          VariableCategory = Scalar

!CHECK: !$omp target defaultmap(none)
  !$omp target defaultmap(none)
   a = 6.78
!CHECK: !$omp end target
  !$omp end target

!PARSE-TREE:      OmpBeginBlockDirective
!PARSE-TREE:        OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE:        OmpClauseList -> OmpClause -> Defaultmap -> OmpDefaultmapClause
!PARSE-TREE:          ImplicitBehavior = None

!CHECK: !$omp target defaultmap(none:scalar)
  !$omp target defaultmap(none:scalar)
   a = 4.78
!CHECK: !$omp end target 
  !$omp end target

!PARSE-TREE:      OmpBeginBlockDirective
!PARSE-TREE:        OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE:        OmpClauseList -> OmpClause -> Defaultmap -> OmpDefaultmapClause
!PARSE-TREE:          ImplicitBehavior = None
!PARSE-TREE:          VariableCategory = Scalar

!CHECK: !$omp target defaultmap(to:scalar)
  !$omp target defaultmap(to:scalar)
   a = 2.39
!CHECK: !$omp end target
  !$omp end target

!PARSE-TREE:      OmpBeginBlockDirective
!PARSE-TREE:        OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE:        OmpClauseList -> OmpClause -> Defaultmap -> OmpDefaultmapClause
!PARSE-TREE:          ImplicitBehavior = To
!PARSE-TREE:          VariableCategory = Scalar

!CHECK: !$omp target defaultmap(firstprivate:scalar)
  !$omp target defaultmap(firstprivate:scalar)
   a = 9.45
!CHECK: !$omp end target
  !$omp end target

!PARSE-TREE:      OmpBeginBlockDirective
!PARSE-TREE:        OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE:        OmpClauseList -> OmpClause -> Defaultmap -> OmpDefaultmapClause
!PARSE-TREE:          ImplicitBehavior = Firstprivate
!PARSE-TREE:          VariableCategory = Scalar
 
!CHECK: !$omp target defaultmap(tofrom:aggregate)
  !$omp target defaultmap(tofrom:aggregate)
   arrC(1) = 10
   data01_a%a = 11
   data01_a%arr(1) = 100
   data01_a%arr(2) = 245
!CHECK: !$omp end target
  !$omp end target

!PARSE-TREE:      OmpBeginBlockDirective
!PARSE-TREE:        OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE:        OmpClauseList -> OmpClause -> Defaultmap -> OmpDefaultmapClause
!PARSE-TREE:          ImplicitBehavior = Tofrom
!PARSE-TREE:          VariableCategory = Aggregate
 
!CHECK: !$omp target defaultmap(tofrom:allocatable)  
  !$omp target defaultmap(tofrom:allocatable)
   alloc_arr(23) = 234
!CHECK: !$omp end target
  !$omp end target

!PARSE-TREE:      OmpBeginBlockDirective
!PARSE-TREE:        OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE:        OmpClauseList -> OmpClause -> Defaultmap -> OmpDefaultmapClause
!PARSE-TREE:          ImplicitBehavior = Tofrom
!PARSE-TREE:          VariableCategory = Allocatable
 
!CHECK: !$omp target defaultmap(default:pointer)
  !$omp target defaultmap(default:pointer)
   ptrArr=>arrC
   ptrArr(2) = 5
   prtArr(200) = 34
!CHECK: !$omp end target
  !$omp end target 

!PARSE-TREE:      OmpBeginBlockDirective
!PARSE-TREE:        OmpBlockDirective -> llvm::omp::Directive = target
!PARSE-TREE:        OmpClauseList -> OmpClause -> Defaultmap -> OmpDefaultmapClause
!PARSE-TREE:          ImplicitBehavior = Default
!PARSE-TREE:          VariableCategory = Pointer

end program main
!CHECK-LABEL: end program main
