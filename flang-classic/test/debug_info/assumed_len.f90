!RUN: %flang -g -S -emit-llvm %s -o - | FileCheck %s

!Verify the DebugInfo metadata contains DIStringType followed by DILocalVariable
!CHECK-DAG: !DIStringType(name: "character(*)!2", stringLength: [[N:![0-9]+]]
!CHECK-DAG: [[N]] = !DILocalVariable(

program assumedLength
  call sub('Hello')
  contains
  subroutine sub(string)
    implicit none
    character(len=*), intent(in) :: string
    print *, string
  end subroutine sub
end program assumedLength
