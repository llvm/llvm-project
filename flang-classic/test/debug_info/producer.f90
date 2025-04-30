!RUN: %flang -gdwarf-4 -S -emit-llvm %s -o - | FileCheck %s

!CHECK: !DICompileUnit(language: DW_LANG_Fortran90
!CHECK-SAME: flang -gdwarf-4 -S -emit-llvm

program main
end program main
