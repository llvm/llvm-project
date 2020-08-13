# REQUIRES: x86

# RUN: mkdir -p %t
#
# RUN: llvm-mc -filetype obj -triple x86_64-apple-darwin %s -o %t/test.o
# RUN: lld -flavor darwinnew -o %t/test -syslibroot %S/Inputs/MacOSX.sdk -lSystem -framework CoreFoundation %t/test.o
#
# RUN: llvm-objdump --bind --no-show-raw-insn -d -r %t/test | FileCheck %s

# CHECK: Disassembly of section __TEXT,__text:
# CHECK: movq {{.*}} # [[ADDR:[0-9a-f]+]]

# CHECK: Bind table:
# CHECK-DAG: __DATA_CONST __got 0x[[ADDR]] pointer 0 libSystem ___nan
# CHECK-DAG: __DATA __data {{.*}} pointer 0 CoreFoundation _OBJC_CLASS_$_NSObject
# CHECK-DAG: __DATA __data {{.*}} pointer 0 CoreFoundation _OBJC_METACLASS_$_NSObject
# CHECK-DAG: __DATA __data {{.*}} pointer 0 CoreFoundation _OBJC_IVAR_$_NSConstantArray._count
# CHECK-DAG: __DATA __data {{.*}} pointer 0 CoreFoundation _OBJC_EHTYPE_$_NSException

.section __TEXT,__text
.global _main

_main:
  movq ___nan@GOTPCREL(%rip), %rax
  ret

.data
  .quad _OBJC_CLASS_$_NSObject
  .quad _OBJC_METACLASS_$_NSObject
  .quad _OBJC_IVAR_$_NSConstantArray._count
  .quad _OBJC_EHTYPE_$_NSException
