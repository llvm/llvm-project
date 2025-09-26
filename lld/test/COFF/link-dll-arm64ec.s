REQUIRES: aarch64, x86
RUN: split-file %s %t.dir && cd %t.dir

RUN: llvm-mc -filetype=obj -triple=arm64ec-windows test.s -o test.obj
RUN: llvm-mc -filetype=obj -triple=arm64ec-windows dll.s -o dll.obj
RUN: llvm-mc -filetype=obj -triple=arm64ec-windows %S/Inputs/loadconfig-arm64ec.s -o loadconfig-arm64ec.obj

RUN: lld-link -machine:arm64ec -dll -noentry -out:import.dll loadconfig-arm64ec.obj dll.obj \
RUN:          -export:func -export:func2=func -export:func3=func -export:func4=func \
RUN:          -export:data,DATA -noimplib

RUN: lld-link -machine:arm64ec -dll -noentry -out:out.dll loadconfig-arm64ec.obj test.obj import.dll \
RUN:          -lldmingw -exclude-all-symbols -auto-import:no

RUN: llvm-readobj --coff-imports out.dll | FileCheck --check-prefix=IMPORTS %s
IMPORTS:      Import {
IMPORTS-NEXT:   Name: import.dll
IMPORTS-NEXT:   ImportLookupTableRVA:
IMPORTS-NEXT:   ImportAddressTableRVA:
IMPORTS-NEXT:   Symbol: data (0)
IMPORTS-NEXT:   Symbol: func (0)
IMPORTS-NEXT:   Symbol: func2 (0)
IMPORTS-NEXT:   Symbol: func3 (0)
IMPORTS-NEXT:   Symbol: func4 (0)
IMPORTS-NEXT: }

#--- test.s
    .weak_anti_dep func
    .weak_anti_dep "#func"
    .set func,"#func"
    .set "#func",thunk

    .section .test, "r"
    .rva __imp_data
    .rva func
    .rva "#func2"
    .rva __imp_aux_func3
    .rva __imp_func4

    .text
    .globl thunk
thunk:
    ret

    .globl __icall_helper_arm64ec
    .p2align 2, 0x0
__icall_helper_arm64ec:
    mov w0, #0
    ret

#--- dll.s
    .text
    .globl func
func:
    ret

    .data
    .globl data
data:
    .word 0
