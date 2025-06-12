REQUIRES: aarch64
RUN: split-file %s %t.dir && cd %t.dir

RUN: llvm-mc -filetype=obj -triple=arm64ec-windows ext.s -o ext.obj
RUN: llvm-mc -filetype=obj -triple=arm64ec-windows impl.s -o impl.obj
RUN: llvm-mc -filetype=obj -triple=arm64ec-windows impl-cpp.s -o impl-cpp.obj
RUN: llvm-mc -filetype=obj -triple=arm64ec-windows %S/Inputs/loadconfig-arm64ec.s -o loadconfig.obj

# Ensure -alternatename can change a mangled function symbol aliasing a defined symbol (typically a guest exit thunk).

RUN: lld-link -machine:arm64ec -dll -noentry -out:out1.dll ext.obj loadconfig.obj "-alternatename:#func=altsym"

RUN: llvm-objdump -d out1.dll | FileCheck --check-prefix=DISASM %s
DISASM:      0000000180001000 <.text>:
DISASM-NEXT: 180001000: 52800020     mov     w0, #0x1                // =1
DISASM-NEXT: 180001004: d65f03c0     ret
DISASM-NOT: .thnk

RUN: llvm-readobj --hex-dump=.test out1.dll | FileCheck --check-prefix=TESTSEC %s
TESTSEC: 0x180005000 00100000 00100000

# Ensure -alternatename can change a demangled function symbol aliasing an anti-dependency symbol.

RUN: lld-link -machine:arm64ec -dll -noentry -out:out2.dll ext.obj loadconfig.obj -alternatename:func=altsym

RUN: llvm-objdump -d out2.dll | FileCheck --check-prefix=DISASM2 %s
DISASM2:      Disassembly of section .text:
DISASM2-EMPTY:
DISASM2-NEXT: 0000000180001000 <.text>:
DISASM2-NEXT: 180001000: 52800020     mov     w0, #0x1                // =1
DISASM2-NEXT: 180001004: d65f03c0     ret
DISASM2-EMPTY:
DISASM2-NEXT: Disassembly of section .thnk:
DISASM2-EMPTY:
DISASM2-NEXT: 0000000180006000 <.thnk>:
DISASM2-NEXT: 180006000: 52800040     mov     w0, #0x2                // =2
DISASM2-NEXT: 180006004: d65f03c0     ret

RUN: llvm-readobj --hex-dump=.test out2.dll | FileCheck --check-prefix=TESTSEC2 %s
TESTSEC2: 0x180005000 00100000 00600000

# Ensure -alternatename cannot modify a demangled function symbol aliasing a defined symbol.

RUN: lld-link -machine:arm64ec -dll -noentry -out:out3.dll impl.obj loadconfig.obj -alternatename:func=altsym
RUN: llvm-objdump -d out3.dll | FileCheck --check-prefix=DISASM %s
RUN: llvm-readobj --hex-dump=.test out3.dll | FileCheck --check-prefix=TESTSEC %s

RUN: lld-link -machine:arm64ec -dll -noentry -out:out4.dll impl-cpp.obj loadconfig.obj -alternatename:func=altsym
RUN: llvm-objdump -d out4.dll | FileCheck --check-prefix=DISASM %s
RUN: llvm-readobj --hex-dump=.test out4.dll | FileCheck --check-prefix=TESTSEC %s

#--- ext.s
        .weak_anti_dep func
.set func, "#func"
        .weak_anti_dep "#func"
.set "#func", thunksym

        .section .test, "r"
        .rva func
        .rva "#func"

        .section .thnk,"xr",discard,thunksym
thunksym:
        mov w0, #2
        ret

        .section .text,"xr",discard,altsym
        .globl altsym
altsym:
        mov w0, #1
        ret

#--- impl.s
        .weak_anti_dep func
.set func, "#func"

        .section .test, "r"
        .rva func
        .rva "#func"

        .section .text,"xr",discard,"#func"
"#func":
        mov w0, #1
        ret

        .section .text,"xr",discard,altsym
        .globl altsym
altsym:
        mov w0, #2
        ret

#--- impl-cpp.s
        .weak_anti_dep func
.set func, "?func@@$$hYAXXZ"

        .section .test, "r"
        .rva func
        .rva "?func@@$$hYAXXZ"

        .section .text,"xr",discard,"?func@@$$hYAXXZ"
"?func@@$$hYAXXZ":
        mov w0, #1
        ret

        .section .text,"xr",discard,altsym
        .globl altsym
altsym:
        mov w0, #2
        ret
