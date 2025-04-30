! RUN: %flang -O0 -S -emit-llvm %s -o - | FileCheck %s
! CHECK: %struct[[STRUCT1:[^=]+]] = type <{ [2052 x i8] }>
! CHECK: @[[STRUCT1]] = common global %struct[[STRUCT1]] zeroinitializer, align 1024
module MyModule
   implicit none

   !dir$ align 512
   integer(kind=4) :: a512
   !dir$ align 512
   integer(kind=4) :: b512
   !dir$ align 512
   integer, dimension (5,5) :: c

   !dir$ align 128
   integer(kind=4) :: d128

   integer(kind=4) :: e4

   !dir$ align 1024
   integer(kind=4) :: f1024

   interface
      module subroutine module_interface_subroutine()
      end subroutine module_interface_subroutine
   end interface

end module MyModule

submodule (MyModule) submodule_align

contains
   module subroutine module_interface_subroutine()

      a512 = 11
      ! CHECK: {{[^g]+}}getelementptr {{[^,]+}}, {{[^,]+}}, i64 512
      b512 = 12
      ! CHECK: {{[^g]+}}getelementptr {{[^,]+}}, {{[^,]+}}, i64 1072
      c(3, 3) = 13
      ! CHECK: {{[^g]+}}getelementptr {{[^,]+}}, {{[^,]+}}, i64 1152
      d128 = 14
      ! CHECK: {{[^g]+}}getelementptr {{[^,]+}}, {{[^,]+}}, i64 1156
      e4 = 15
      ! CHECK: {{[^g]+}}getelementptr {{[^,]+}}, {{[^,]+}}, i64 2048
      f1024 = 16
   end subroutine module_interface_subroutine
end submodule submodule_align


program MainProgram
   use MyModule
   call module_interface_subroutine()
end program MainProgram
