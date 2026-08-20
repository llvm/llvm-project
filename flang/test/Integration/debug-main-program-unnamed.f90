! RUN: %flang_fc1 -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck %s

! Test a main program without a PROGRAM statement. There is no name to spell, so
! the entry symbol is used as it is.

subroutine Example_Function()
  print *, "in the subroutine"
end subroutine Example_Function

call Example_Function
end

! CHECK-DAG: !DISubprogram(name: "example_function", linkageName: "example_function_"
! CHECK-DAG: !DISubprogram(name: "_QQmain", linkageName: "_QQmain"{{.*}}DISPFlagMainSubprogram
