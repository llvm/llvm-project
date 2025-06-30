! REQUIRES: openmp_runtime

! RUN: %flang_fc1 %openmp_flags -fopenmp-version=52 -fdebug-dump-parse-tree %s | FileCheck %s
! RUN: %flang_fc1 %openmp_flags -fdebug-unparse -fopenmp-version=52 %s | FileCheck %s --check-prefix="UNPARSE"

module functions
  implicit none

  interface
  function func() result(i)
    character(1) :: i
  end function
  end interface

contains
  function func1() result(i)
    !$omp declare target enter(func1) indirect(.true.)
    !CHECK: | | | | | OmpDeclareTargetSpecifier -> OmpDeclareTargetWithClause -> OmpClauseList -> OmpClause -> Enter -> OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'func1'
    !CHECK-NEXT: | | | | | OmpClause -> Indirect -> OmpIndirectClause -> Scalar -> Logical -> Expr = '.true._4'
    !CHECK-NEXT: | | | | | | LiteralConstant -> LogicalLiteralConstant
    !CHECK-NEXT: | | | | | | | bool = 'true'
    character(1) :: i
    i = 'a'
    return
  end function

  function func2() result(i)
    !$omp declare target enter(func2) indirect
    !CHECK: | | | | | OmpDeclareTargetSpecifier -> OmpDeclareTargetWithClause -> OmpClauseList -> OmpClause -> Enter -> OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'func2'
    !CHECK-NEXT: | | | | | OmpClause -> Indirect -> OmpIndirectClause ->
    character(1) :: i
    i = 'b'
    return
  end function
end module

program main
  use functions
  implicit none
  procedure (func), pointer :: ptr1=>func1, ptr2=>func2
  character(1) :: val1, val2

  !$omp target map(from: val1)
  val1 = ptr1()
  !$omp end target
  !$omp target map(from: val2)
  val2 = ptr2()
  !$omp end target

end program

!UNPARSE: !$OMP DECLARE TARGET  ENTER(func1) INDIRECT(.true._4)
!UNPARSE: !$OMP DECLARE TARGET  ENTER(func2) INDIRECT()
