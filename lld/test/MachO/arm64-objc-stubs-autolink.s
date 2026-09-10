# REQUIRES: aarch64

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/main.s \
# RUN:   -o %t/main.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/dep.s \
# RUN:   -o %t/dep.o
# RUN: llvm-ar rcs %t/libdep.a %t/dep.o
# RUN: %lld -arch arm64 -lSystem -o %t/out %t/main.o -L%t \
# RUN:   -objc_stubs_fast -U _objc_msgSend
# RUN: llvm-objdump --no-show-raw-insn --section=__TEXT,__objc_stubs \
# RUN:   --macho %t/out | FileCheck %s

# CHECK:      Contents of (__TEXT,__objc_stubs) section
# CHECK-NEXT: _objc_msgSend$plain:
# CHECK-NEXT: adrp    x1,
# CHECK-NEXT: ldr     x1, {{.*}} ; Objc selector ref: plain
# CHECK-NEXT: adrp    x16,
# CHECK-NEXT: ldr     x16, {{.*}} ; literal pool symbol address: _objc_msgSend
# CHECK-NEXT: br      x16

#--- main.s
.linker_option "-ldep"
.text
.globl _main
_main:
  bl _dep
  ret

#--- dep.s
.text
.globl _dep
_dep:
  bl _objc_msgSend$plain
  ret
