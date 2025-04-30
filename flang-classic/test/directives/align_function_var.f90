! RUN: %flang -O0 -S -emit-llvm %s -o - | FileCheck %s

! CHECK: %struct.[[BLOCK1:STATICS[0-9]+]] = type <{ [132 x i8] }>
! CHECK: @.[[BLOCK1]] = internal global %struct.[[BLOCK1]] <{ [132 x i8] {{[^,]+}} }>, align 128
integer(kind=4) function function_init()
   !dir$ align 128
   integer(kind=4) :: a128 = 10
   !dir$ align 128
   integer(kind=4) :: b128 = 20
! CHECK: {{[^g]+}}getelementptr {{[^,]+}}, {{[^,]+}}, i64 128
   a128 = a128 + 1
! CHECK: {{[^g]+}}getelementptr {{[^,]+}}, {{[^,]+}}, i64 128
   b128 = b128 + 2
! CHECK: {{[^g]+}}getelementptr {{[^,]+}}, {{[^,]+}}, i64 128
   function_init = a128 + b128
end function function_init


integer(kind=4) function function_no_init()
! CHECK: {{[^=]+}} = alloca i32, align 128
   !dir$ align 128
   integer(kind=4) :: a128
! CHECK: {{[^=]+}} = alloca i32, align 128
   !dir$ align 128
   integer(kind=4) :: b128

   a128 = 1
   b128 = 2

   function_no_init = a128 + b128
end function function_no_init


program MainProgram
   implicit none


   integer(kind=4) :: res
   integer(kind=4) :: function_init
   integer(kind=4) :: function_no_init

   res = function_init()
   res = function_no_init()

end program MainProgram

