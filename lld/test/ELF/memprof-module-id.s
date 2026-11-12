# REQUIRES: x86
# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/a.s -o %t/a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/b.s -o %t/b.o
# RUN: ld.lld -shared %t/a.o %t/b.o -o %t/ab.so
# RUN: ld.lld -shared %t/b.o %t/a.o -o %t/ba.so
# RUN: ld.lld -shared %t/a.o -o %t/a.so
# RUN: llvm-readelf --sections --wide %t/ab.so | \
# RUN:   FileCheck %s --check-prefix=SHARED
# RUN: llvm-readelf --notes %t/ab.so | FileCheck %s --check-prefix=NOTE
# RUN: llvm-objcopy --dump-section \
# RUN:   .note.llvm.memprof.module-id=%t/ab.note %t/ab.so
# RUN: llvm-objcopy --dump-section \
# RUN:   .note.llvm.memprof.module-id=%t/ba.note %t/ba.so
# RUN: llvm-objcopy --dump-section \
# RUN:   .note.llvm.memprof.module-id=%t/a.note %t/a.so
# RUN: cmp %t/ab.note %t/ba.note
# RUN: not cmp %t/ab.note %t/a.note
# RUN: ld.lld -e a %t/a.o -o %t/a.exe
# RUN: llvm-readelf --sections --wide %t/a.exe | \
# RUN:   FileCheck %s --check-prefix=EXEC

# SHARED: .note.llvm.memprof.module-id NOTE
# SHARED-NOT: .note.llvm.memprof.tu-id
# NOTE: Displaying notes found in: .note.llvm.memprof.module-id
# NOTE: LLVM
# NOTE-SAME: 0x00000010
# NOTE: description data: b4 ff cb 23 73 7c ec 31 5a 4a 4d 1a a2 a6 20 ce
# EXEC: .note.llvm.memprof.tu-id NOTE
# EXEC-NOT: .note.llvm.memprof.module-id

#--- a.s
.section .note.llvm.memprof.tu-id,"a",@note
.p2align 2
.long 5
.long 16
.long 0x4d505401
.asciz "LLVM"
.p2align 2
.byte 0, 1, 2, 3, 4, 5, 6, 7
.byte 8, 9, 10, 11, 12, 13, 14, 15

.text
.globl a
a:
  ret

#--- b.s
.section .note.llvm.memprof.tu-id,"a",@note
.p2align 2
.long 5
.long 16
.long 0x4d505401
.asciz "LLVM"
.p2align 2
.byte 16, 17, 18, 19, 20, 21, 22, 23
.byte 24, 25, 26, 27, 28, 29, 30, 31

.text
.globl b
b:
  ret
