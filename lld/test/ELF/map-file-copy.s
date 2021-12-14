# REQUIRES: x86

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/1.s -o %t/1.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/2.s -o %t/2.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/3.s -o %t/3.o
# RUN: ld.lld -shared -soname=3 --version-script=%t/3.ver %t/3.o -o %t/3.so
# RUN: ld.lld -Map=%t/1.map %t/1.o %t/2.o %t/3.so -o %t/1
# RUN: FileCheck %s --input-file=%t/1.map

## Both TUs reference func/copy which need a canonical PLT entry/copy relocation.
## Test we print func/copy just once.
# CHECK:      {{ }}.plt
# CHECK-NEXT:         <internal>:(.plt)
# CHECK-NEXT:                 func@v1{{$}}
# CHECK-NEXT: .dynamic

# CHECK:      .bss.rel.ro
# CHECK-NEXT:         <internal>:(.bss.rel.ro)
## Ideally this is displayed as copy@v2.
# CHECK-NEXT:                 copy{{$}}
# CHECK-NEXT: .got.plt

#--- 1.s
.global _start
_start:
.symver func, func@@@v1
  mov $copy, %eax
  mov $func - ., %eax

#--- 2.s
.symver func, func@@@v1
  mov $copy, %eax
  mov $func - ., %eax

#--- 3.s
.globl func
.symver func, func@v1, remove
.type func, @function
func:
  ret

.section .rodata,"a"
.globl copy
.type copy, @object
copy:
.byte 1
.size copy, 1

#--- 3.ver
v1 { func; };
v2 { copy; };
