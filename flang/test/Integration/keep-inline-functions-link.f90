! -fkeep-inline-functions must only record alwaysinline definitions that
! belong to this translation unit. A builtin bitcode library brought in via
! -mlink-builtin-bitcode can itself contain an alwaysinline function (for
! its own, unrelated reasons); that function was never named by a
! !DIR$ INLINEALWAYS in this file and must not be added to llvm.used /
! llvm.compiler.used because of this flag. This requires the check to run
! before the builtin bitcode library is linked into the module.

! RUN: %flang -emit-llvm -c -o %t.bc %S/Inputs/keep-inline-lib.f90
! RUN: %flang_fc1 -emit-llvm -O2 -fkeep-inline-functions -mlink-builtin-bitcode %t.bc -o - %s | FileCheck %s

program main
  external lib_add_two
  integer :: lib_add_two
  integer :: n
  n = lib_add_two(5)
  n = own_add_three(n)
  print *, n
contains
  integer function own_add_three(n)
    !dir$ inlinealways own_add_three
    integer :: n
    own_add_three = n + 3
  end function
end program

! The list has exactly one entry: this translation unit's own
! own_add_three, and not the linked library's lib_add_two.
! CHECK: @llvm.{{(compiler.)?}}used = appending global [1 x ptr] [ptr @_QFPown_add_three]
