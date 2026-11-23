! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck %s

! Test that the main program name is emitted in lowercase, like the name of any
! other subprogram.

subroutine Example_Function()
  print *, "in the subroutine"
end subroutine Example_Function

program Example
  call Example_Function
end program Example

! CHECK-DAG: !DISubprogram(name: "example_function", linkageName: "example_function_"
! CHECK-DAG: !DISubprogram(name: "example", linkageName: "_QQmain"

! CHECK-NOT: !DISubprogram(name: "EXAMPLE"
