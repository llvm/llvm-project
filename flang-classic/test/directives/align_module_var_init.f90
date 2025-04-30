! RUN: %flang -O0 -S -emit-llvm %s -o - | FileCheck %s

! CHECK: %struct.[[BLOCK1:STATICS[0-9]+]] = type <{ [260 x i8] }>
! CHECK: %struct[[STRUCT1:[^=]+]] = type <{ [516 x i8] }>

! CHECK: @.[[BLOCK1]] = internal global %struct.[[BLOCK1]] <{ [260 x i8] {{[^,]+}} }>, align 256
! CHECK: @[[STRUCT1]] = global %struct[[STRUCT1]] <{ [516 x i8] {{[^,]+}} }>, align 512
module module_align
   implicit none

   !dir$ align 128
   integer :: a128 = 123

   !dir$ align 512
   integer :: b512 = 234

   interface
      module subroutine module_interface_subroutine()
      end subroutine module_interface_subroutine
   end interface

end module module_align

submodule (module_align) submodule_align

contains
   module subroutine module_interface_subroutine()


      !dir$ align 256
      integer :: a256 = 12

      !dir$ align 128
      integer :: b128 = 14

! CHECK: {{[^g]+}}getelementptr {{[^,]+}}, {{[^,]+}}, i64 256
! CHECK: {{[^g]+}}getelementptr {{[^,]+}}, {{[^,]+}}, i64 512
      a128 = 12 + a256
      b512 = 23 + b128


   end subroutine module_interface_subroutine
end submodule submodule_align

program MainProgram

   use module_align
   call module_interface_subroutine()

end program MainProgram
