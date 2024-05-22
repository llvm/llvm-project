# REQUIRES: loongarch

## Edge case: when a TLS symbol is being accessed in both GD and IE manners,
## correct reloc behavior should be preserved for both kinds of accesses.

# RUN: llvm-mc --filetype=obj --triple=loongarch32 %s -o %t.la32.o
# RUN: ld.lld %t.la32.o -shared -o %t.la32
# RUN: llvm-mc --filetype=obj --triple=loongarch64 %s -o %t.la64.o
# RUN: ld.lld %t.la64.o -shared -o %t.la64

# RUN: llvm-readelf -Wr %t.la32 | FileCheck --check-prefix=LA32-REL %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.la32 | FileCheck --check-prefix=LA32 %s

# RUN: llvm-readelf -Wr %t.la64 | FileCheck --check-prefix=LA64-REL %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.la64 | FileCheck --check-prefix=LA64 %s

# LA32-REL-NOT:  R_LARCH_32
# LA32-REL:      0002023c  00000206 R_LARCH_TLS_DTPMOD32   00000000   y + 0
# LA32-REL-NEXT: 00020240  00000208 R_LARCH_TLS_DTPREL32   00000000   y + 0
# LA32-REL-NEXT: 00020244  0000020a R_LARCH_TLS_TPREL32    00000000   y + 0

# LA64-REL-NOT:  R_LARCH_64
# LA64-REL:      00000000000203a0  0000000200000007 R_LARCH_TLS_DTPMOD64   0000000000000000 y + 0
# LA64-REL-NEXT: 00000000000203a8  0000000200000009 R_LARCH_TLS_DTPREL64   0000000000000000 y + 0
# LA64-REL-NEXT: 00000000000203b0  000000020000000b R_LARCH_TLS_TPREL64    0000000000000000 y + 0

# LA32:      101d4: pcalau12i $a0, 16
# LA32-NEXT:        ld.w $a0, $a0, 580
# LA32-NEXT:        pcalau12i $a1, 16
# LA32-NEXT:        addi.w $a1, $a1, 572

# LA64:      102e0: pcalau12i $a0, 16
# LA64-NEXT:        ld.d $a0, $a0, 944
# LA64-NEXT:        pcalau12i $a1, 16
# LA64-NEXT:        addi.d $a1, $a1, 928

.global _start
_start:
la.tls.ie $a0, y  # should refer to the GOT entry relocated by the R_LARCH_TLS_TPRELnn record
la.tls.gd $a1, y  # should refer to the GOT entry relocated by the R_LARCH_TLS_DTPMODnn record

.section .tbss,"awT",@nobits
.global y
y:
.word 0
.size y, 4
