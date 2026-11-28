! REQUIRES: x86-registered-target
! RUN: rm -f %t.min.bc
! RUN: %flang -target x86_64-unknown-linux-gnu -g -flto=thin -fthin-link-bitcode=%t.min.bc -c %s -o %t.o
! RUN: llvm-bcanalyzer -dump %t.o | FileCheck %s --check-prefix=FULL
! RUN: llvm-bcanalyzer -dump %t.min.bc | FileCheck %s --check-prefix=MIN --implicit-check-not='<COMPILE_UNIT'

! FULL: <COMPILE_UNIT
! MIN: <GLOBALVAL_SUMMARY_BLOCK

subroutine f
end
