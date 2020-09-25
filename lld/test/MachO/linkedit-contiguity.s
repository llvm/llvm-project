# REQUIRES: x86
# RUN: mkdir -p %t

## codesign requires that each setion in __LINKEDIT ends where the next one
## starts. This test enforces that invariant.
## TODO: Test other __LINKEDIT sections here as support for them gets added.
## Examples of such sections include the data for LC_CODE_SIGNATURE and
## LC_DATA_IN_CODE.

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %p/Inputs/libhello.s \
# RUN:   -o %t/libhello.o
# RUN: %lld -dylib \
# RUN:   -install_name @executable_path/libhello.dylib %t/libhello.o \
# RUN:   -o %t/libhello.dylib

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/test.o
# RUN: %lld -o %t/test \
# RUN:   -L%t -lhello %t/test.o -lSystem

# RUN: llvm-objdump --macho --all-headers %t/test | FileCheck %s

# CHECK:      cmd LC_DYLD_INFO_ONLY
# CHECK-NEXT: cmdsize 48
# CHECK-NEXT: rebase_off 0
# CHECK-NEXT: rebase_size 0
# CHECK-NEXT: bind_off [[#BIND_OFF:]]
# CHECK-NEXT: bind_size [[#BIND_SIZE:]]
# CHECK-NEXT: weak_bind_off 0
# CHECK-NEXT: weak_bind_size 0
# CHECK-NEXT: lazy_bind_off [[#LAZY_OFF: BIND_OFF + BIND_SIZE]]
# CHECK-NEXT: lazy_bind_size [[#LAZY_SIZE:]]
# CHECK-NEXT: export_off [[#EXPORT_OFF: LAZY_OFF + LAZY_SIZE]]
# CHECK-NEXT: export_size [[#]]

.text
.globl _main
_main:
  sub $8, %rsp # 16-byte-align the stack; dyld checks for this
  callq _print_hello
  add $8, %rsp
  ret
