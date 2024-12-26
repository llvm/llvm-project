// REQUIRES: aarch64
// RUN: split-file %s %t.dir && cd %t.dir

#--- test-simple.s
// Build a simple function with an entry thunk.

    .section .text,"xr",discard,func
    .globl func
    .p2align 2
func:
    mov w0, #1
    ret

    .section .wowthk$aa,"xr",discard,thunk
    .globl thunk
    .p2align 2
thunk:
    mov w0, #10
    ret

    .section .hybmp$x, "yi"
    .symidx func
    .symidx thunk
    .word 1  // entry thunk

    .data
    .rva func

// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows %S/Inputs/loadconfig-arm64ec.s -o loadcfg.obj
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows %S/Inputs/loadconfig-arm64ec.s -o native-loadcfg.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows test-simple.s -o test-simple.obj
// RUN: lld-link -machine:arm64ec -dll -noentry -out:out-simple.dll loadcfg.obj test-simple.obj
// RUN: llvm-objdump -d out-simple.dll | FileCheck --check-prefix=DISASM %s

// DISASM:      Disassembly of section .text:
// DISASM-EMPTY:
// DISASM-NEXT: 0000000180001000 <.text>:
// DISASM-NEXT: 180001000: 00000009     udf     #0x9
// DISASM-NEXT: 180001004: 52800020     mov     w0, #0x1                // =1
// DISASM-NEXT: 180001008: d65f03c0     ret
// DISASM-NEXT: 18000100c: 52800140     mov     w0, #0xa                // =10
// DISASM-NEXT: 180001010: d65f03c0     ret

// RUN: llvm-readobj --sections out-simple.dll | FileCheck --check-prefix=HYBMP %s
// HYBMP-NOT: .hybmp

// RUN: lld-link -machine:arm64x -dll -noentry -out:out-simplex.dll native-loadcfg.obj loadcfg.obj test-simple.obj
// RUN: llvm-objdump -d out-simplex.dll | FileCheck --check-prefix=DISASM %s

#--- test-split-func.s
// Build a simple function with an entry thunk, but pass it in multiple files.

    .section .text,"xr",discard,func
    .globl func
    .p2align 2
func:
    mov w0, #1
    ret

#--- test-split-thunk.s
    .section .wowthk$aa,"xr",discard,thunk
    .globl thunk
    .p2align 2
thunk:
    mov w0, #10
    ret

#--- test-split-hybmp.s
    .section .hybmp$x, "yi"
    .symidx func
    .symidx thunk
    .word 1  // entry thunk

#--- test-split-data.s
    .data
    .rva func

// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows test-split-func.s -o test-split-func.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows test-split-thunk.s -o test-split-thunk.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows test-split-hybmp.s -o test-split-hybmp.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows test-split-data.s -o test-split-data.obj
// RUN: lld-link -machine:arm64ec -dll -noentry -out:out-split.dll loadcfg.obj \
// RUN:          test-split-func.obj test-split-thunk.obj test-split-data.obj test-split-hybmp.obj
// RUN: llvm-objdump -d out-split.dll | FileCheck --check-prefix=DISASM %s

#--- test-align.s
// Build multiple functions with thunks and various alignments and check that entry thunk offsets
// are correctly placed.

    .section .text,"xr",discard,func
    .globl func
    .p2align 2
func:
    mov w0, #1
    nop
    ret

    .section .text,"xr",discard,func2
    .globl func2
    .p2align 2
func2:
    mov w0, #2
    ret

    .section .text,"xr",discard,func3
    .globl func3
    .p2align 3
func3:
    mov w0, #3
    nop
    ret

    .section .text,"xr",discard,func4
    .globl func4
    .p2align 3
func4:
    mov w0, #4
    ret

    .section .text,"xr",discard,func5
    .globl func5
    .p2align 3
func5:
    mov w0, #5
    ret

    .section .text,"xr",discard,func6
    .globl func6
    .p2align 4
func6:
    mov w0, #6
    ret

    .section .wowthk$aa,"xr",discard,thunk
    .globl thunk
    .p2align 2
thunk:
    mov w0, #10
    ret

    .section .wowthk$aa,"xr",discard,thunk2
    .globl thunk2
    .p2align 2
thunk2:
    mov w0, #20
    ret

    .section .hybmp$x, "yi"
    .symidx func
    .symidx thunk
    .word 1  // entry thunk
    .symidx func2
    .symidx thunk2
    .word 1  // entry thunk
    .symidx func3
    .symidx thunk
    .word 1  // entry thunk
    .symidx func4
    .symidx thunk
    .word 1  // entry thunk
    .symidx func5
    .symidx thunk
    .word 1  // entry thunk
    .symidx func6
    .symidx thunk
    .word 1  // entry thunk

    .data
    .rva func
    .rva func2
    .rva func3
    .rva func4
    .rva func5
    .rva func6

// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows test-align.s -o test-align.obj
// RUN: lld-link -machine:arm64ec -dll -noentry -out:out-align.dll loadcfg.obj test-align.obj
// RUN: llvm-objdump -d out-align.dll | FileCheck --check-prefix=ALIGN %s

// ALIGN:      Disassembly of section .text:
// ALIGN-EMPTY:
// ALIGN-NEXT: 0000000180001000 <.text>:
// ALIGN-NEXT: 180001000: 00000055     udf     #0x55
// ALIGN-NEXT: 180001004: 52800020     mov     w0, #0x1                // =1
// ALIGN-NEXT: 180001008: d503201f     nop
// ALIGN-NEXT: 18000100c: d65f03c0     ret
// ALIGN-NEXT: 180001010: 0000004d     udf     #0x4d
// ALIGN-NEXT: 180001014: 52800040     mov     w0, #0x2                // =2
// ALIGN-NEXT: 180001018: d65f03c0     ret
// ALIGN-NEXT: 18000101c: 00000039     udf     #0x39
// ALIGN-NEXT: 180001020: 52800060     mov     w0, #0x3                // =3
// ALIGN-NEXT: 180001024: d503201f     nop
// ALIGN-NEXT: 180001028: d65f03c0     ret
// ALIGN-NEXT: 18000102c: 00000029     udf     #0x29
// ALIGN-NEXT: 180001030: 52800080     mov     w0, #0x4                // =4
// ALIGN-NEXT: 180001034: d65f03c0     ret
// ALIGN-NEXT: 180001038: 00000000     udf     #0x0
// ALIGN-NEXT: 18000103c: 00000019     udf     #0x19
// ALIGN-NEXT: 180001040: 528000a0     mov     w0, #0x5                // =5
// ALIGN-NEXT: 180001044: d65f03c0     ret
// ALIGN-NEXT: 180001048: 00000000     udf     #0x0
// ALIGN-NEXT: 18000104c: 00000009     udf     #0x9
// ALIGN-NEXT: 180001050: 528000c0     mov     w0, #0x6                // =6
// ALIGN-NEXT: 180001054: d65f03c0     ret
// ALIGN-NEXT: 180001058: 52800140     mov     w0, #0xa                // =10
// ALIGN-NEXT: 18000105c: d65f03c0     ret
// ALIGN-NEXT: 180001060: 52800280     mov     w0, #0x14               // =20
// ALIGN-NEXT: 180001064: d65f03c0     ret

#--- test-icf-thunk.s
// Build two functions with identical entry thunks and check that thunks are merged by ICF.

    .section .text,"xr",discard,func
    .globl func
    .p2align 2
func:
    mov w0, #1
    ret

    .section .text,"xr",discard,func2
    .globl func2
    .p2align 2
func2:
    mov w0, #2
    ret

    .section .wowthk$aa,"xr",discard,thunk
    .globl thunk
    .p2align 2
thunk:
    mov w0, #10
    ret

    .section .wowthk$aa,"xr",discard,thunk2
    .globl thunk2
    .p2align 2
thunk2:
    mov w0, #10
    ret

    .section .hybmp$x, "yi"
    .symidx func
    .symidx thunk
    .word 1  // entry thunk
    .symidx func2
    .symidx thunk2
    .word 1  // entry thunk

    .data
    .rva func
    .rva func2

// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows test-icf-thunk.s -o test-icf-thunk.obj
// RUN: lld-link -machine:arm64ec -dll -noentry -out:out-icf-thunk.dll loadcfg.obj test-icf-thunk.obj
// RUN: llvm-objdump -d out-icf-thunk.dll | FileCheck --check-prefix=ICF-THUNK %s

// ICF-THUNK:      Disassembly of section .text:
// ICF-THUNK-EMPTY:
// ICF-THUNK-NEXT: 0000000180001000 <.text>:
// ICF-THUNK-NEXT: 180001000: 00000015     udf     #0x15
// ICF-THUNK-NEXT: 180001004: 52800020     mov     w0, #0x1                // =1
// ICF-THUNK-NEXT: 180001008: d65f03c0     ret
// ICF-THUNK-NEXT: 18000100c: 00000009     udf     #0x9
// ICF-THUNK-NEXT: 180001010: 52800040     mov     w0, #0x2                // =2
// ICF-THUNK-NEXT: 180001014: d65f03c0     ret
// ICF-THUNK-NEXT: 180001018: 52800140     mov     w0, #0xa                // =10
// ICF-THUNK-NEXT: 18000101c: d65f03c0     ret

#--- test-icf-diff-thunk.s
// Build two identical functions with different entry thunks and check that they are not merged by ICF.

    .section .text,"xr",discard,func
    .globl func
    .p2align 2
func:
    mov w0, #1
    ret

    .section .text,"xr",discard,func2
    .globl func2
    .p2align 2
func2:
    mov w0, #1
    ret

    .section .wowthk$aa,"xr",discard,thunk
    .globl thunk
    .p2align 2
thunk:
    mov w0, #10
    ret

    .section .wowthk$aa,"xr",discard,thunk2
    .globl thunk2
    .p2align 2
thunk2:
    mov w0, #20
    ret

    .section .hybmp$x, "yi"
    .symidx func
    .symidx thunk
    .word 1  // entry thunk
    .symidx func2
    .symidx thunk2
    .word 1  // entry thunk

    .data
    .rva func
    .rva func2

// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows test-icf-diff-thunk.s -o test-icf-diff-thunk.obj
// RUN: lld-link -machine:arm64ec -dll -noentry -out:out-icf-diff-thunk.dll loadcfg.obj test-icf-diff-thunk.obj
// RUN: llvm-objdump -d out-icf-diff-thunk.dll | FileCheck --check-prefix=ICF-DIFF-THUNK %s

// ICF-DIFF-THUNK:      Disassembly of section .text:
// ICF-DIFF-THUNK-EMPTY:
// ICF-DIFF-THUNK-NEXT: 0000000180001000 <.text>:
// ICF-DIFF-THUNK-NEXT: 180001000: 00000015     udf     #0x15
// ICF-DIFF-THUNK-NEXT: 180001004: 52800020     mov     w0, #0x1                // =1
// ICF-DIFF-THUNK-NEXT: 180001008: d65f03c0     ret
// ICF-DIFF-THUNK-NEXT: 18000100c: 00000011     udf     #0x11
// ICF-DIFF-THUNK-NEXT: 180001010: 52800020     mov     w0, #0x1                // =1
// ICF-DIFF-THUNK-NEXT: 180001014: d65f03c0     ret
// ICF-DIFF-THUNK-NEXT: 180001018: 52800140     mov     w0, #0xa                // =10
// ICF-DIFF-THUNK-NEXT: 18000101c: d65f03c0     ret
// ICF-DIFF-THUNK-NEXT: 180001020: 52800280     mov     w0, #0x14               // =20
// ICF-DIFF-THUNK-NEXT: 180001024: d65f03c0     ret

#--- test-icf-both.s
// Build two identical functions with identical entry thunks and check that they are merged by ICF.

    .section .text,"xr",discard,func
    .globl func
    .p2align 2
func:
    mov w0, #1
    ret

    .section .text,"xr",discard,func2
    .globl func2
    .p2align 2
func2:
    mov w0, #1
    ret

    .section .wowthk$aa,"xr",discard,thunk
    .globl thunk
    .p2align 2
thunk:
    mov w0, #10
    ret

    .section .wowthk$aa,"xr",discard,thunk2
    .globl thunk2
    .p2align 2
thunk2:
    mov w0, #10
    ret

    .section .hybmp$x, "yi"
    .symidx func
    .symidx thunk
    .word 1  // entry thunk
    .symidx func2
    .symidx thunk2
    .word 1  // entry thunk

    .data
    .rva func
    .rva func2

// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows test-icf-both.s -o test-icf-both.obj
// RUN: lld-link -machine:arm64ec -dll -noentry -out:out-icf-both.dll loadcfg.obj test-icf-both.obj
// RUN: llvm-objdump -d out-icf-both.dll | FileCheck --check-prefix=DISASM %s

