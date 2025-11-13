! REQUIRES: x86-registered-target
! checks fatlto objects: that valid bitcode is included in the object file generated.

! RUN: %flang -fc1 -triple x86_64-unknown-linux-gnu -flto -ffat-lto-objects -emit-obj %s -o %t.o
! RUN: llvm-readelf -S %t.o | FileCheck %s --check-prefixes=ELF
! RUN: llvm-objcopy --dump-section=.llvm.lto=%t.bc %t.o
! RUN: llvm-dis %t.bc -o - | FileCheck %s --check-prefixes=DIS

! ELF: .llvm.lto
! DIS: define void @_QQmain()
! DIS-NEXT:  ret void
! DIS-NEXT: }

! RUN: %flang -fc1 -triple x86_64-unknown-linux-gnu -flto -ffat-lto-objects -S %s -o - | FileCheck %s --check-prefixes=ASM

!      ASM: .section        .llvm.lto,"e",@llvm_lto
! ASM-NEXT: .Lllvm.embedded.object:
! ASM-NEXT:        .asciz  "BC
! ASM-NEXT: .size   .Lllvm.embedded.object
program test
end program
