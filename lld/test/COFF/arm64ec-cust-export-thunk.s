# REQUIRES: aarch64, x86
# RUN: split-file %s %t.dir && cd %t.dir

# Test that metadata is generated when a custom export thunk is supplied.

# RUN: llvm-mc -filetype=obj -triple=arm64ec-windows func.s -o func.obj
# RUN: llvm-mc -filetype=obj -triple=arm64ec-windows hp-func.s -o hp-func.obj
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows thunk.s -o thunk.obj
# RUN: llvm-mc -filetype=obj -triple=arm64ec-windows %S/Inputs/loadconfig-arm64ec.s -o loadconfig-arm64ec.obj

# RUN: lld-link -out:out.dll -machine:arm64ec func.obj thunk.obj loadconfig-arm64ec.obj -dll -noentry "-export:#func,EXPORTAS,func"

# RUN: llvm-objdump -d out.dll | FileCheck --check-prefixes=DISASM,DISASM-EXP %s
# DISASM:      Disassembly of section .text:
# DISASM-EMPTY:
# DISASM-NEXT: 0000000180001000 <.text>:
# DISASM-NEXT: 180001000: 52800040     mov     w0, #0x2                // =2
# DISASM-NEXT: 180001004: d65f03c0     ret
# DISASM-NEXT:                 ...
# DISASM-EXP-EMPTY:
# DISASM-EXP-NEXT: 0000000180002000 <func>:
# DISASM-NEXT: 180002000: b8 03 00 00 00               movl    $0x3, %eax
# DISASM-NEXT: 180002005: c3                           retq

# RUN: llvm-objdump -p out.dll | FileCheck --check-prefix=EXPORT %s
# EXPORT:      Ordinal      RVA  Name
# EXPORT-NEXT:       1   0x2000  func

# RUN: llvm-readobj --coff-load-config out.dll | FileCheck --check-prefix=CHPE %s
# CHPE:       CodeMap [
# CHPE-NEXT:    0x1000 - 0x1008  ARM64EC
# CHPE-NEXT:    0x2000 - 0x2006  X64
# CHPE-NEXT:  ]
# CHPE-NEXT:  CodeRangesToEntryPoints [
# CHPE-NEXT:    0x2000 - 0x2006 -> 0x2000
# CHPE-NEXT:  ]
# CHPE-NEXT:  RedirectionMetadata [
# CHPE-NEXT:    0x2000 -> 0x1000
# CHPE-NEXT:  ]

# RUN: lld-link -out:out2.dll -machine:arm64ec hp-func.obj thunk.obj loadconfig-arm64ec.obj -dll -noentry
# RUN: llvm-objdump -d out2.dll | FileCheck --check-prefix=DISASM %s
# RUN: llvm-readobj --coff-load-config out2.dll | FileCheck --check-prefix=CHPE %s

#--- func.s
    .globl "#func"
    .p2align 2, 0x0
"#func":
    mov w0, #2
    ret

#--- hp-func.s
    .section .text,"xr",discard,"#func$hp_target"
    .globl "#func$hp_target"
    .p2align 2, 0x0
"#func$hp_target":
    mov w0, #2
    ret

    .def "EXP+#func"
    .scl 2
    .type 32
    .endef
    .weak func
.set func, "EXP+#func"
    .weak "#func"
.set "#func", "#func$hp_target"

    .data
    .rva func

#--- thunk.s
    .def "EXP+#func"
    .scl 2
    .type 32
    .endef
    .section .wowthk$aa,"xr",discard,"EXP+#func"
    .globl "EXP+#func"
    .p2align 2, 0x0
"EXP+#func":
    movl $3, %eax
    retq
