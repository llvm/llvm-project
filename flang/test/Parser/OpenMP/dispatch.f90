! RUN: %flang_fc1 -fopenmp -fdebug-dump-parse-tree %s | FileCheck %s
! RUN: %flang_fc1 -fopenmp -fdebug-unparse %s | FileCheck %s --check-prefix="UNPARSE"

integer function func(a, b, c)
  integer  :: a, b, c
  func = a + b + c
end function func

subroutine sub(x)
  use iso_c_binding
  integer :: func
  integer :: r
  type(c_ptr) :: x
  integer :: a = 14, b = 7, c = 21
!UNPARSE: !$OMP DISPATCH DEVICE(3_4) NOWAIT NOCONTEXT(.false._4) NOVARIANTS(.true._4)
!CHECK: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPDispatchConstruct
!CHECK-NEXT: | | | OmpDispatchDirective
!CHECK: | | | | OmpClauseList -> OmpClause -> Device -> OmpDeviceClause
!CHECK-NEXT: | | | | | Scalar -> Integer -> Expr = '3_4'
!CHECK-NEXT: | | | | | | LiteralConstant -> IntLiteralConstant = '3'
!CHECK-NEXT: | | | | OmpClause -> Nowait
!CHECK-NEXT: | | | | OmpClause -> Nocontext -> Scalar -> Logical -> Expr = '.false._4'
!CHECK-NEXT: | | | | | LiteralConstant -> LogicalLiteralConstant
!CHECK-NEXT: | | | | | | bool = 'false'
!CHECK-NEXT: | | | | OmpClause -> Novariants -> Scalar -> Logical -> Expr = '.true._4'
!CHECK-NEXT: | | | | | EQ
!CHECK-NEXT: | | | | | | Expr = '1_4'
!CHECK-NEXT: | | | | | | | LiteralConstant -> IntLiteralConstant = '1'
!CHECK-NEXT: | | | | | | Expr = '1_4'
!CHECK-NEXT: | | | | | | | LiteralConstant -> IntLiteralConstant = '1'
!CHECK-NEXT: | | | Block
 
  !$omp dispatch device(3) nowait nocontext(.false.) novariants(1.eq.1)
  r = func(a, b, c)
!UNPARSE: !$OMP END DISPATCH
!CHECK: | | | OmpEndDispatchDirective
  !$omp end dispatch

!! Test the "no end dispatch" option.
!UNPARSE: !$OMP DISPATCH  DEVICE(3_4) IS_DEVICE_PTR(x)
!CHECK: | | ExecutionPartConstruct -> ExecutableConstruct -> OpenMPConstruct -> OpenMPDispatchConstruct
!CHECK-NEXT: | | | OmpDispatchDirective
!CHECK: | | | | OmpClause -> IsDevicePtr ->  OmpObjectList -> OmpObject -> Designator -> DataRef -> Name = 'x'  
  !$omp dispatch device(3) is_device_ptr(x)
  r = func(a+1, b+2, c+3)
!CHECK-NOT: | | | OmpEndDispatchDirective

end subroutine sub



