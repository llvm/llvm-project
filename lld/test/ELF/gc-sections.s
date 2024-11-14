# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: ld.lld %t -o %t2
# RUN: llvm-readobj --sections --symbols %t2 | FileCheck -check-prefix=NOGC %s
# RUN: ld.lld --gc-sections --print-gc-sections %t -o %t2 | FileCheck --check-prefix=GC1-DISCARD %s
# RUN: llvm-readobj --sections --symbols %t2 | FileCheck -check-prefix=GC1 %s
# RUN: ld.lld --export-dynamic --gc-sections %t -o %t2
# RUN: llvm-readobj --sections --symbols %t2 | FileCheck -check-prefix=GC2 %s

# RUN: llvm-mc -filetype=obj -triple=x86_64 --crel %s -o %t.o
# RUN: ld.lld --gc-sections --print-gc-sections %t.o -o %t2 | FileCheck --check-prefix=GC1-DISCARD %s
# RUN: llvm-readobj --sections --symbols %t2 | FileCheck -check-prefix=GC1 %s

# NOGC: Name: .eh_frame
# NOGC: Name: .text
# NOGC: Name: .init
# NOGC: Name: .init_x
# NOGC: Name: .fini
# NOGC: Name: .tdata
# NOGC: Name: .tbss
# NOGC: Name: .ctors
# NOGC: Name: .dtors
# NOGC: Name: .fini_array
# NOGC: Name: .init_array
# NOGC: Name: .preinit_array
# NOGC: Name: .jcr
# NOGC: Name: .jcr_x
# NOGC: Name: .debug_pubtypes
# NOGC: Name: .comment
# NOGC: Name: a
# NOGC: Name: b
# NOGC: Name: c
# NOGC: Name: e
# NOGC: Name: f
# NOGC: Name: g
# NOGC: Name: h
# NOGC: Name: x
# NOGC: Name: y
# NOGC: Name: d

# GC1-DISCARD:      removing unused section {{.*}}:(.text.d)
# GC1-DISCARD-NEXT: removing unused section {{.*}}:(.text.x)
# GC1-DISCARD-NEXT: removing unused section {{.*}}:(.text.y)
# GC1-DISCARD-NEXT: removing unused section {{.*}}:(.tbss.f)
# GC1-DISCARD-NEXT: removing unused section {{.*}}:(.tdata.h)
# GC1-DISCARD-NEXT: removing unused section {{.*}}:(.init_x)
# GC1-DISCARD-NEXT: removing unused section {{.*}}:(.jcr_x)
# GC1-DISCARD-EMPTY:

# GC1:     Name: .eh_frame
# GC1:     Name: .text
# GC1:     Name: .init
# GC1:     Name: .fini
# GC1:     Name: .tdata
# GC1:     Name: .tbss
# GC1:     Name: .ctors
# GC1:     Name: .dtors
# GC1:     Name: .fini_array
# GC1:     Name: .init_array
# GC1:     Name: .preinit_array
# GC1:     Name: .jcr
# GC1:     Name: .debug_pubtypes
# GC1:     Name: .comment
# GC1:     Name: a
# GC1:     Name: b
# GC1:     Name: c
# GC1:     Name: e
# GC1-NOT: Name: f
# GC1:     Name: g
# GC1-NOT: Name: h
# GC1-NOT: Name: x
# GC1-NOT: Name: y
# GC1-NOT: Name: d

# GC2:     Name: .eh_frame
# GC2:     Name: .text
# GC2:     Name: .init
# GC2:     Name: .fini
# GC2:     Name: .tdata
# GC2:     Name: .tbss
# GC2:     Name: .ctors
# GC2:     Name: .dtors
# GC2:     Name: .fini_array
# GC2:     Name: .init_array
# GC2:     Name: .preinit_array
# GC2:     Name: .jcr
# GC2:     Name: .debug_pubtypes
# GC2:     Name: .comment
# GC2:     Name: a
# GC2:     Name: b
# GC2:     Name: c
# GC2:     Name: e
# GC2-NOT: Name: f
# GC2:     Name: g
# GC2-NOT: Name: h
# GC2-NOT: Name: x
# GC2-NOT: Name: y
# GC2:     Name: d

.globl _start, d
.protected a, b, c, e, f, g, h, x, y
_start:
  call a

.section .text.a,"ax",@progbits
a:
  call _start
  call b

.section .text.b,"ax",@progbits
b:
  leaq e@tpoff(%rax),%rdx
  call c

.section .text.c,"ax",@progbits
c:
  leaq g@tpoff(%rax),%rdx

.section .text.d,"ax",@progbits
d:
  nop

.section .text.x,"ax",@progbits
x:
  call y

.section .text.y,"ax",@progbits
y:
  call x

.section .tbss.e,"awT",@nobits
e:
  .quad 0

.section .tbss.f,"awT",@nobits
f:
  .quad 0

.section .tdata.g,"awT",@progbits
g:
  .quad 0

.section .tdata.h,"awT",@progbits
h:
  .quad 0

.section .ctors,"aw",@progbits
  .quad 0

.section .dtors,"aw",@progbits
  .quad 0

.section .init,"ax"
  .quad 0

.section .init_x,"ax"
  .quad 0

.section .fini,"ax"
  .quad 0

.section .fini_array,"aw",@fini_array
  .quad 0

# https://golang.org/cl/373734
.section .init_array,"aw",@progbits
  .quad 0

# Work around https://github.com/rust-lang/rust/issues/92181
.section .init_array.00001,"aw",@progbits
  .quad 0

.section .preinit_array,"aw",@preinit_array
  .quad 0

.section .jcr,"aw"
  .quad 0

.section .jcr_x,"aw"
  .quad 0

.section .eh_frame,"a",@unwind
  .quad 0

.section .debug_pubtypes,"",@progbits
  .quad 0

.section .comment,"MS",@progbits,8
  .quad 0
