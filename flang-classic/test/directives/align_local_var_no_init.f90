! RUN: %flang -O0 -S -emit-llvm %s -o - | FileCheck %s
program main
   implicit none
! CHECK: {{[^=]+}} = alloca i64, align 256
! CHECK: {{[^=]+}} = alloca i64, align 256

   !dir$ align 256
   integer(kind=8) :: d,e

   d=e

end program main
