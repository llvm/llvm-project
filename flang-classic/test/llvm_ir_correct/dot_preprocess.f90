! RUN: %flang -cpp -S -emit-llvm %s -o - | FileCheck %s
#define PP_N 10
subroutine pp(n)
implicit none
integer :: n
if (n-1.ne.PP_N) then
  print *, 'Success'
end if
end subroutine pp

! //CHECK: {{.*}}  = icmp eq i32 {{.*}}, 11
