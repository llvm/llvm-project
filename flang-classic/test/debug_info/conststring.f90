!! This should be enabled only after #888
!! check if conststrings are stored correctly for special characters
!RUN: %flang -g -S -emit-llvm %s -o - | FileCheck %s

!! \0 = 0 , " = 34 = 0x22
!CHECK: call void @llvm.dbg.value(metadata [10 x i8] c"a\00d\22g     ", metadata [[CONSTR1:![0-9a-f]+]], metadata !DIExpression())
!CHECK: call void @llvm.dbg.value(metadata [10 x i8] c"h i~j     ", metadata [[CONSTR2:![0-9a-f]+]], metadata !DIExpression())
!! \n = 10 \r = 13
!CHECK: call void @llvm.dbg.value(metadata [10 x i8] c"k\0Al\0Dm     ", metadata [[CONSTR3:![0-9a-f]+]], metadata !DIExpression())
!CHECK-LABEL: distinct !DISubprogram(name: "main"
!CHECK: [[CONSTR1]] = !DILocalVariable(name: "constr1",
!CHECK: [[CONSTR2]] = !DILocalVariable(name: "constr2",
!CHECK: [[CONSTR3]] = !DILocalVariable(name: "constr3",

program main
  character(10),parameter :: constr1 = "a"//achar(0)//"d"//achar(34)//"g"
  character(10),parameter :: constr2 = "h i~j"
  character(10),parameter :: constr3 = "k"//achar(10)//"l"//achar(13)//"m"

  print *,constr1
  print *,constr2
  print *,constr3
end
