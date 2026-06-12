! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck  %s

program test
 type t1
   integer :: XcX
   integer :: xdtx
 end type
  type(t1) :: var
  var%XcX = 2
  var%xdtx = 3
end

! Test that there is no debug info for compiler generated globals.
! CHECK-NOT: DIGlobalVariable
