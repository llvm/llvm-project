# REQUIRES: loongarch
# RUN: rm -rf %t && split-file %s %t

# RUN: llvm-mc --filetype=obj --triple=loongarch32-unknown-elf %t/a.s -o %t/a.la32.o
# RUN: llvm-mc --filetype=obj --triple=loongarch64-unknown-elf %t/a.s -o %t/a.la64.o

# RUN: ld.lld %t/a.la32.o -shared -T %t/a.t -o %t/a.la32.so
# RUN: ld.lld %t/a.la64.o -shared -T %t/a.t -o %t/a.la64.so

# RUN: llvm-objdump -d --no-show-raw-insn %t/a.la32.so | FileCheck --check-prefixes=DIS,DIS32 %s
# RUN: llvm-objdump -d --no-show-raw-insn %t/a.la64.so | FileCheck --check-prefixes=DIS,DIS64 %s

## PLT should be present in this case.
# DIS:        Disassembly of section .plt:
# DIS:        <.plt>:
# DIS:        234020: pcaddu12i $t3, 510
# DIS32-NEXT:         ld.w $t3, $t3, 84
# DIS64-NEXT:         ld.d $t3, $t3, 184
# DIS-NEXT:           jirl $t1, $t3, 0
# DIS-NEXT:           nop

# DIS:      Disassembly of section .text:
# DIS:      <foo>:
# DIS-NEXT: nop
# DIS-NEXT: nop
# DIS-NEXT: nop
# DIS-NEXT: pcalau12i $t0, -510
# DIS-NEXT: jirl $zero, $t0, 32

# RUN: llvm-mc --filetype=obj --triple=loongarch32-unknown-elf %t/error.s -o %t/error.la32.o
# RUN: llvm-mc --filetype=obj --triple=loongarch64-unknown-elf %t/error.s -o %t/error.la64.o
# RUN: not ld.lld %t/error.la32.o -shared -o %t/error.la32.so 2>&1 | FileCheck --check-prefix=ERR %s
# RUN: not ld.lld %t/error.la64.o -shared -o %t/error.la64.so 2>&1 | FileCheck --check-prefix=ERR %s
# ERR: error: relocation R_LARCH_PCALA_LO12 cannot be used against symbol 'bar'; recompile with -fPIC

#--- a.t
SECTIONS {
 .plt   0x234000: { *(.plt) }
 .text  0x432000: { *(.text) }
}

#--- a.s
.p2align 12
.global foo
foo:
## The nops are for pushing the relocs off page boundary, to better see the
## page-aligned semantics in action.
    nop
    nop
    nop
    ## The offsets should be -510 (0x234 - 0x432) and 32 (PLT header size + 0)
    ## respectively.
    pcalau12i   $t0, %pc_hi20(bar)
    jirl        $zero, $t0, %pc_lo12(bar)

#--- error.s
.global foo
foo:
    pcalau12i   $t0, %pc_hi20(bar)
    ld.w        $t0, $t0, %pc_lo12(bar)
