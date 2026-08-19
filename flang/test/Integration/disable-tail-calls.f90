! test -fno-optimize-sibling-calls flag disables tail call optimization

! RUN: %flang_fc1 -emit-llvm -fno-optimize-sibling-calls -o - %s | FileCheck %s

recursive subroutine f(n)
  integer, intent(in) :: n
  if (n > 0) call f(n - 1)
end subroutine f

! CHECK: define void @f_{{.*}}#[[ATTRS:[0-9]+]]
! CHECK: call void @f_
! CHECK: attributes #[[ATTRS]]{{.*}}"disable-tail-calls"="true"
