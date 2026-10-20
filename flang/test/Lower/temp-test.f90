! RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck %s

!CHECK-LABEL: add
function add(a,b)
  integer :: a, b, add
  add = a + b
end function
