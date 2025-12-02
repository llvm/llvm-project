! Test that the output is LLVM bitcode for LTO and not a native objectfile by
! disassembling it to LLVM IR. Also tests that module summaries are emitted for LTO

! RUN: %flang %s -c -o - | not llvm-dis -o %t
! RUN: %flang_fc1 %s -emit-llvm-bc -o - | llvm-dis -o - | FileCheck %s
! CHECK: define void @_QQmain()
! CHECK-NEXT:  ret void
! CHECK-NEXT: }
! CHECK-NOT: !{{.*}} = !{i32 1, !"ThinLTO", i32 0}
! CHECK-NOT: ^{{.*}} = module:
! CHECK-NOT: ^{{.*}} = gv: (name:
! CHECK-NOT: ^{{.*}} = blockcount:

! RUN: %flang -flto=thin %s -c -o - | llvm-dis -o - | FileCheck %s --check-prefix=THIN
! THIN: define void @_QQmain()
! THIN-NEXT:  ret void
! THIN-NEXT: }
! THIN-NOT: !{{.*}} = !{i32 1, !"ThinLTO", i32 0}
! THIN-NOT: ^{{.*}} = module:
! THIN-NOT: ^{{.*}} = gv: (name:
! THIN-NOT: ^{{.*}} = blockcount:

! RUN: %flang -flto %s -c -o - | llvm-dis -o - | FileCheck %s --check-prefix=FULL
! FULL: define void @_QQmain()
! FULL-NEXT:  ret void
! FULL-NEXT: }
! FULL: !{{.*}} = !{i32 1, !"ThinLTO", i32 0}
! FULL: ^{{.*}} = module:
! FULL: ^{{.*}} = gv: (name:
! FULL: ^{{.*}} = blockcount:

! RUN: %flang_fc1 -flto -emit-llvm-bc %s -o - | llvm-bcanalyzer -dump| FileCheck --check-prefix=MOD-SUMM %s
! MOD-SUMM: FULL_LTO_GLOBALVAL_SUMMARY_BLOCK
program main
end program
