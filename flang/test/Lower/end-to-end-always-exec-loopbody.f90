! RUN: bbc --always-execute-loop-body %s -o - | tco | llc --relocation-model=pic --filetype=obj -o %temp.o
! RUN: %CC %temp.o -L%L -lFortranRuntime -lFortranDecimal -lstdc++ -lm -o %temp.loop
! RUN: %temp.loop | FileCheck %s

program alwaysexecuteloopbody
  implicit none
  integer :: i,j
  do i=4, 1, 1
    ! CHECK: In goto loop
    print *, "In goto loop"
    return
  end do
  ! CHECK-NOT: Should not exec
  print *, "Should not exec"
end program

