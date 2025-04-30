! RUN: %flang -O0 -S -emit-llvm %s -o - | FileCheck %s

program MainProgram
   implicit none

   call subroutine_init()
   call subroutine_no_init()

end program MainProgram


! CHECK: %struct.[[BLOCK1:STATICS[0-9]+]] = type <{ [2052 x i8] }>
! CHECK: @.[[BLOCK1]] = internal global %struct.[[BLOCK1]] <{ [2052 x i8] {{[^,]+}} }>, align 2048
subroutine subroutine_init()
   !dir$ align 2048
   integer(kind=4) :: a2048 = 10
   !dir$ align 2048
   integer(kind=4) :: b2048 = 20
   ! CHECK: {{[^g]+}}getelementptr {{[^,]+}}, {{[^,]+}}, i64 2048
   ! CHECK: {{[^g]+}}getelementptr {{[^,]+}}, {{[^,]+}}, i64 2048
   a2048 = a2048 + 1
   b2048 = b2048 + 2

end subroutine subroutine_init

subroutine subroutine_no_init()
   ! CHECK: {{[^=]+}} = alloca i32, align 1024
   !dir$ align 1024
   integer(kind=4) :: a1024
   ! CHECK: {{[^=]+}} = alloca i32, align 1024
   !dir$ align 1024
   integer(kind=4) :: b1024
   a1024 = 1
   b1024 = 2

end subroutine subroutine_no_init
