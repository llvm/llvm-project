# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=i386 %s -o %t.o
# RUN: echo '.tbss; .globl c; c: .zero 4' | llvm-mc -filetype=obj -triple=i386 - -o %t1.o
# RUN: ld.lld -shared -soname=t1.so %t1.o -o %t1.so

# RUN: ld.lld -shared -z now %t.o %t1.o -o %t.so
# RUN: llvm-readobj -r -x .got %t.so | FileCheck --check-prefix=GD-REL %s
# RUN: llvm-objdump -h -d --no-show-raw-insn %t.so | FileCheck --check-prefix=GD %s

# RUN: ld.lld -shared -z now %t.o %t1.o -o %t-rela.so -z rela
# RUN: llvm-readobj -r -x .got %t-rela.so | FileCheck --check-prefix=GD-RELA %s

# RUN: ld.lld -z now %t.o %t1.o -o %t
# RUN: llvm-readelf -r %t | FileCheck --check-prefix=NOREL %s
# RUN: llvm-objdump -h -d --no-show-raw-insn %t | FileCheck --check-prefix=LE %s

# RUN: ld.lld -z now %t.o %t1.so -o %t
# RUN: llvm-readobj -r %t | FileCheck --check-prefix=IE-REL %s
# RUN: llvm-objdump -h -d --no-show-raw-insn %t | FileCheck --check-prefix=IE %s

# GD-REL:      .rel.dyn {
# GD-REL-NEXT:   0x2258 R_386_TLS_DESC -
# GD-REL-NEXT:   0x2248 R_386_TLS_DESC a
# GD-REL-NEXT:   0x2250 R_386_TLS_DESC c
# GD-REL-NEXT: }
# GD-REL:      Hex dump of section '.got':
# GD-REL-NEXT: 0x00002248 00000000 00000000 00000000 00000000
# GD-REL-NEXT: 0x00002258 00000000 0b000000

# GD-RELA:      .rela.dyn {
# GD-RELA-NEXT:   0x2264 R_386_TLS_DESC - 0xB
# GD-RELA-NEXT:   0x2254 R_386_TLS_DESC a 0x0
# GD-RELA-NEXT:   0x225C R_386_TLS_DESC c 0x0
# GD-RELA-NEXT: }
# GD-RELA:      Hex dump of section '.got':
# GD-RELA-NEXT: 0x00002254 00000000 00000000 00000000 00000000
# GD-RELA-NEXT: 0x00002264 00000000 00000000

# GD:      .got     00000018 00002248
# GD:      .got.plt 0000000c 00002260

# &.rel.dyn[a]-.got.plt = 0x2248-0x2260 = -24
# GD:      leal -24(%ebx), %eax
# GD-NEXT: calll *(%eax)
# GD-NEXT: movl %gs:(%eax), %eax

# &.rel.dyn[b]-.got.plt = 0x2258-0x2260 = -8
# GD-NEXT: leal -8(%ebx), %eax
# GD-NEXT: movl %edx, %ebx
# GD-NEXT: calll *(%eax)
# GD-NEXT: movl %gs:(%eax), %eax

# &.rel.dyn[c]-.got.plt = 0x2250-0x2260 = -16
# GD-NEXT: leal -16(%ebx), %eax
# GD-NEXT: calll *(%eax)
# GD-NEXT: movl %gs:(%eax), %eax

# NOREL: no relocations

## st_value(a) - tls_size = -8
# LE:      leal -8, %eax
# LE-NEXT: nop
# LE-NEXT: movl %gs:(%eax), %eax
## st_value(b) - tls_size = -5
# LE:      leal -5, %eax
# LE-NEXT: movl %edx, %ebx
# LE-NEXT: nop
# LE-NEXT: movl %gs:(%eax), %eax
## st_value(c) - tls_size = -4
# LE:      leal -4, %eax
# LE-NEXT: nop
# LE-NEXT: movl %gs:(%eax), %eax

# IE-REL:      .rel.dyn {
# IE-REL-NEXT:   0x40222C R_386_TLS_TPOFF c
# IE-REL-NEXT: }

# IE:      .got     00000004 0040222c
# IE:      .got.plt 0000000c 00402230

## a and b are relaxed to use LE.
# IE:      leal -4, %eax
# IE-NEXT: nop
# IE-NEXT: movl %gs:(%eax), %eax
# IE-NEXT: leal -1, %eax
# IE-NEXT: movl %edx, %ebx
# IE-NEXT: nop
# IE-NEXT: movl %gs:(%eax), %eax
## &.got[a]-.got.plt = 0x2220 - 0x2224 = -4
# IE-NEXT: movl -4(%ebx), %eax
# IE-NEXT: nop
# IE-NEXT: movl %gs:(%eax), %eax

leal a@tlsdesc(%ebx), %eax
call *a@tlscall(%eax)
movl %gs:(%eax), %eax

leal b@tlsdesc(%ebx), %eax
movl %edx, %ebx  # GCC -O0 may add an extra insn in between.
call *b@tlscall(%eax)
movl %gs:(%eax), %eax

leal c@tlsdesc(%ebx), %eax
call *c@tlscall(%eax)
movl %gs:(%eax), %eax

.section .tbss
.globl a
.zero 8
a:
.zero 3
b:
.zero 1
