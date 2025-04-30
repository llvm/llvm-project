!RUN: %flang -g -S -emit-llvm %s -o - | FileCheck %s

!CHECK: call void @llvm.dbg.declare(metadata ptr %rvar_{{[0-9]+}}, metadata [[RESULT:![0-9]+]], metadata !DIExpression())
!CHECK: [[RESULT]] = !DILocalVariable(name: "rvar"

function func(arg) result(rvar)
  integer, intent(in) :: arg ! input
  integer :: rvar ! output
  rvar = arg + 2
end function func
