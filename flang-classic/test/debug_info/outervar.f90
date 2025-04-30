!RUN: %flang -g -S -emit-llvm %s -o - | FileCheck %s

!CHECK: distinct !DIGlobalVariable(name: "prog_i"
!CHECK-NOT: distinct !DIGlobalVariable(name: "prog_i"

program main
  integer :: prog_i
  prog_i = 99
  call sub()
contains
  subroutine sub()
    print *,prog_i
  end subroutine sub
end program main
