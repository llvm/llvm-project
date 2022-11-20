# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=i386 %s -o %t.o

# RUN: ld.lld -shared -z now %t.o -o %t.so
# RUN: llvm-readobj -r %t.so | FileCheck --check-prefix=LD-REL %s
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t.so | FileCheck --check-prefix=LD %s

# RUN: ld.lld -z now %t.o -o %t
# RUN: llvm-readelf -r %t | FileCheck --check-prefix=NOREL %s
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t | FileCheck --check-prefix=LE %s

## Check _TLS_MODULE_BASE_ used by LD produces a dynamic relocation with a value of 0.
# LD-REL:      .rel.dyn {
# LD-REL-NEXT:   R_386_TLS_DESC -
# LD-REL-NEXT: }

## 0x2318-0x1267 = 4273
## dtpoff(a) = 8, dtpoff(b) = 12
# LD:      leal -8(%ebx), %eax
# LD-NEXT: calll *(%eax)
# LD-NEXT: leal 8(%eax), %ecx
# LD-NEXT: leal 12(%eax), %edx

## When producing an executable, the LD code sequence can be relaxed to LE.
## It is the same as GD->LE.
## tpoff(_TLS_MODULE_BASE_) = 0, tpoff(a) = -8, tpoff(b) = -4

# NOREL: no relocations

# LE:      leal 0, %eax
# LE-NEXT: nop
# LE-NEXT: leal -8(%eax), %ecx
# LE-NEXT: leal -4(%eax), %edx
# LE-NEXT: addl %gs:0, %ecx
# LE-NEXT: addl %gs:0, %edx

leal _TLS_MODULE_BASE_@tlsdesc(%ebx), %eax
call *_TLS_MODULE_BASE_@tlscall(%eax)
leal a@dtpoff(%eax), %ecx
leal b@dtpoff(%eax), %edx
addl %gs:0, %ecx
addl %gs:0, %edx

.section .tbss
.zero 8
a:
.zero 4
b:
.zero 4
