! Test that the output is LLVM bitcode for LTO and not a native objectfile by
! disassembling it to LLVM IR.
! Right now there is nothing special about it and it is similar to non-lto IR,
! more work is needed to add things like module summaries.

! RUN: %flang %s -c -o - | not llvm-dis -o %t
! RUN: %flang_fc1 %s -emit-llvm-bc -o - | llvm-dis -o - | FileCheck %s

! RUN: %flang -flto %s -c -o - | llvm-dis -o - | FileCheck %s
! RUN: %flang -flto=thin %s -c -o - | llvm-dis -o - | FileCheck %s

! CHECK: define void @_QQmain()
! CHECK-NEXT:  ret void
! CHECK-NEXT: }

! CHECK-NOT: ^0 = module:
! CHECK-NOT: ^1 = gv: (name:
! CHECK-NOT: ^2 = flags:
! CHECK-NOT: ^3 = blockcount:

end program
