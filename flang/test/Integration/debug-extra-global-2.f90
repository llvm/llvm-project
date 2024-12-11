! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck  %s

module m
  integer XcX
end

! Test that global starting with 'X' don't get filtered.
! CHECK: !DIGlobalVariable(name: "xcx", linkageName: "_QMmExcx"{{.*}})
