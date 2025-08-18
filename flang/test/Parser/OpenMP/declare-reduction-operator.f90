! RUN: %flang_fc1 -fdebug-unparse -fopenmp %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp %s | FileCheck --check-prefix="PARSE-TREE" %s

!CHECK-LABEL: SUBROUTINE reduce_1 (n, tts)
subroutine reduce_1 ( n, tts )
  type :: tt
    integer :: x
    integer :: y
 end type tt
  type :: tt2
    real(8) :: x
    real(8) :: y
  end type
 
  integer :: n
  type(tt) :: tts(n)
  type(tt2) :: tts2(n)

!CHECK: !$OMP DECLARE REDUCTION (+:tt: omp_out=tt(x=omp_out%x-omp_in%x,y=omp_out%y-omp_in%y)
!CHECK: ) INITIALIZER(omp_priv=tt(x=0_4,y=0_4))
!PARSE-TREE:  DeclarationConstruct -> SpecificationConstruct -> OpenMPDeclarativeConstruct -> OpenMPDeclareReductionConstruct
!PARSE-TREE: Verbatim
!PARSE-TREE: OmpReductionSpecifier
!PARSE-TREE: OmpReductionIdentifier -> DefinedOperator -> IntrinsicOperator = Add
!PARSE-TREE: OmpReductionCombiner -> AssignmentStmt = 'omp_out=tt(x=omp_out%x-omp_in%x,y=omp_out%y-omp_in%y)'
!PARSE-TREE:    OmpInitializerClause -> AssignmentStmt = 'omp_priv=tt(x=0_4,y=0_4)'
  
  !$omp declare reduction(+ : tt :  omp_out = tt(omp_out%x - omp_in%x , omp_out%y - omp_in%y)) initializer(omp_priv = tt(0,0))

  
!CHECK: !$OMP DECLARE REDUCTION (+:tt2: omp_out=tt2(x=omp_out%x-omp_in%x,y=omp_out%y-omp_in%y)
!CHECK: ) INITIALIZER(omp_priv=tt2(x=0._8,y=0._8)
!PARSE-TREE:  DeclarationConstruct -> SpecificationConstruct -> OpenMPDeclarativeConstruct -> OpenMPDeclareReductionConstruct
!PARSE-TREE: Verbatim
!PARSE-TREE: OmpReductionSpecifier
!PARSE-TREE: OmpReductionIdentifier -> DefinedOperator -> IntrinsicOperator = Add
!PARSE-TREE: OmpReductionCombiner -> AssignmentStmt = 'omp_out=tt2(x=omp_out%x-omp_in%x,y=omp_out%y-omp_in%y)'
!PARSE-TREE:    OmpInitializerClause -> AssignmentStmt = 'omp_priv=tt2(x=0._8,y=0._8)'
  
  !$omp declare reduction(+ :tt2 :  omp_out = tt2(omp_out%x - omp_in%x , omp_out%y - omp_in%y)) initializer(omp_priv = tt2(0,0))
  
  type(tt) :: diffp = tt( 0, 0 )
  type(tt2) :: diffp2 = tt2( 0, 0 )
  integer :: i

  !$omp parallel do reduction(+ : diffp)
  do i = 1, n
     diffp%x = diffp%x + tts(i)%x
     diffp%y = diffp%y + tts(i)%y
  end do

  !$omp parallel do reduction(+ : diffp2)
  do i = 1, n
     diffp2%x = diffp2%x + tts2(i)%x
     diffp2%y = diffp2%y + tts2(i)%y
  end do

end subroutine reduce_1
!CHECK: END SUBROUTINE reduce_1
