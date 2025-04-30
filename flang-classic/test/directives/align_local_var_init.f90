! RUN: %flang -O0 -S -emit-llvm %s -o - | FileCheck %s
! CHECK: %struct.[[BLOCK1:STATICS[0-9]+]] = type <{ [1028 x i8] }>
! CHECK: @.[[BLOCK1]] = internal global %struct.[[BLOCK1]] <{ [1028 x i8] {{[^,]+}} }>, align 1024
program MainProgram
   implicit none
! CHECK: {{[^g]+}}getelementptr {{[^,]+}}, {{[^,]+}}, i64 1024
! CHECK: {{[^g]+}}getelementptr {{[^,]+}}, {{[^,]+}}, i64 1024
   !dir$ align 1024
   integer(kind=4) :: a1024 =30
   !dir$ align 1024
   integer(kind=4) :: b1024 =40
   a1024 = a1024 + b1024

end program MainProgram
