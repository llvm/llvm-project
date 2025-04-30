!RUN: %flang -g -S -emit-llvm %s -o - | FileCheck %s

!CHECK: call void @llvm.dbg.value(metadata i32 1, metadata [[ENM1:![0-9]+]], metadata !DIExpression())
!CHECK: call void @llvm.dbg.value(metadata i32 2, metadata [[ENM2:![0-9]+]], metadata !DIExpression())
!CHECK: call void @llvm.dbg.value(metadata i32 5, metadata [[ENM3:![0-9]+]], metadata !DIExpression())
!CHECK: call void @llvm.dbg.value(metadata i32 6, metadata [[ENM4:![0-9]+]], metadata !DIExpression())
!CHECK: [[ENM1]] = !DILocalVariable(name: "red"
!CHECK: [[ENM2]] = !DILocalVariable(name: "blue"
!CHECK: [[ENM3]] = !DILocalVariable(name: "black"
!CHECK: [[ENM4]] = !DILocalVariable(name: "pink"

program main
  enum, bind(c)
    enumerator :: red =1, blue, black =5
    enumerator :: pink
  endenum
  integer (kind=8) :: svar1, svar2, svar3, svar4
  svar1 = red
  svar2 = blue
  svar3 = black
  svar4 = pink

  print *, svar1, svar2, svar3, svar4

end program main
