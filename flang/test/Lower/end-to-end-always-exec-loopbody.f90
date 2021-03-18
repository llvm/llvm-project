! RUN: bbc --always-execute-loop-body %s -o - | tco | llc --relocation-model=pic --filetype=obj -o %t.o
! RUN: %CC %t.o -L%L -Wl,-rpath -Wl,%L -lFortran_main -lFortranRuntime -lFortranDecimal -lm -o %t.loop
! RUN: %t.loop | FileCheck %s

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

