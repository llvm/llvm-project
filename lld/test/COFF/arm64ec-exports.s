; REQUIRES: aarch64
; RUN: split-file %s %t.dir && cd %t.dir

; RUN: llvm-mc -filetype=obj -triple=arm64ec-windows test.s -o test.obj
; RUN: llvm-mc -filetype=obj -triple=arm64ec-windows drectve.s -o drectve.obj
; RUN: llvm-mc -filetype=obj -triple=arm64ec-windows %S/Inputs/loadconfig-arm64ec.s -o loadconfig-arm64ec.obj

; Check various forms of export directive and make sure that function export name is demangled.

; RUN: lld-link -out:out.dll test.obj loadconfig-arm64ec.obj -dll -noentry -machine:arm64ec \
; RUN:          -export:unmangled_func '-export:#mangled_func' '-export:#exportas_func,EXPORTAS,exportas_func' \
; RUN:          '-export:?cxx_func@@$$hYAHXZ' -export:data_sym,DATA '-export:#mangled_data_sym,DATA'


; RUN: llvm-readobj --coff-exports out.dll | FileCheck --check-prefix=EXP %s
; EXP:      Export {
; EXP-NEXT:   Ordinal: 1
; EXP-NEXT:   Name: #mangled_data_sym
; EXP-NEXT:   RVA: 0x4000
; EXP-NEXT: }
; EXP-NEXT: Export {
; EXP-NEXT:   Ordinal: 2
; EXP-NEXT:   Name: ?cxx_func@@YAHXZ
; EXP-NEXT:   RVA: 0x2030
; EXP-NEXT: }
; EXP-NEXT: Export {
; EXP-NEXT:   Ordinal: 3
; EXP-NEXT:   Name: data_sym
; EXP-NEXT:   RVA: 0x4004
; EXP-NEXT: }
; EXP-NEXT: Export {
; EXP-NEXT:   Ordinal: 4
; EXP-NEXT:   Name: exportas_func
; EXP-NEXT:   RVA: 0x2020
; EXP-NEXT: }
; EXP-NEXT: Export {
; EXP-NEXT:   Ordinal: 5
; EXP-NEXT:   Name: mangled_func
; EXP-NEXT:   RVA: 0x2010
; EXP-NEXT: }
; EXP-NEXT: Export {
; EXP-NEXT:   Ordinal: 6
; EXP-NEXT:   Name: unmangled_func
; EXP-NEXT:   RVA: 0x2000
; EXP-NEXT: }

; RUN: llvm-nm --print-armap out.lib | FileCheck --check-prefix=IMPLIB %s
; IMPLIB:      Archive EC map
; IMPLIB-NEXT: #exportas_func in out
; IMPLIB-NEXT: #mangled_func in out
; IMPLIB-NEXT: #unmangled_func in out
; IMPLIB-NEXT: ?cxx_func@@$$hYAHXZ in out
; IMPLIB-NEXT: ?cxx_func@@YAHXZ in out
; IMPLIB-NEXT: __IMPORT_DESCRIPTOR_out{{.*}} in out
; IMPLIB-NEXT: __NULL_IMPORT_DESCRIPTOR in out
; IMPLIB-NEXT: __imp_?cxx_func@@YAHXZ in out
; IMPLIB-NEXT: __imp_aux_?cxx_func@@YAHXZ in out
; IMPLIB-NEXT: __imp_aux_exportas_func in out
; IMPLIB-NEXT: __imp_aux_mangled_func in out
; IMPLIB-NEXT: __imp_aux_unmangled_func in out
; IMPLIB-NEXT: __imp_data_sym in out
; IMPLIB-NEXT: __imp_exportas_func in out
; IMPLIB-NEXT: __imp_mangled_data_sym in out
; IMPLIB-NEXT: __imp_mangled_func in out
; IMPLIB-NEXT: __imp_unmangled_func in out
; IMPLIB-NEXT: exportas_func in out
; IMPLIB-NEXT: mangled_func in out
; IMPLIB-NEXT: unmangled_func in out
; IMPLIB-NEXT: out{{.*}}_NULL_THUNK_DATA in out


; Check that using .drectve section has the same effect.

; RUN: lld-link -out:out2.dll test.obj loadconfig-arm64ec.obj -dll -noentry -machine:arm64ec drectve.obj
; RUN: llvm-readobj --coff-exports out2.dll | FileCheck --check-prefix=EXP %s
; RUN: llvm-nm --print-armap out2.lib | FileCheck --check-prefix=IMPLIB %s

#--- test.s
        .text
        .globl unmangled_func
        .p2align 2, 0x0
unmangled_func:
        mov w0, #1
        ret

        .globl "#mangled_func"
        .p2align 2, 0x0
"#mangled_func":
        mov w0, #2
        ret

        .globl "#exportas_func"
        .p2align 2, 0x0
"#exportas_func":
        mov w0, #3
        ret

        .globl "?cxx_func@@$$hYAHXZ"
        .p2align 2, 0x0
"?cxx_func@@$$hYAHXZ":
        mov w0, #4
        ret

        .data
        .globl "#mangled_data_sym"
        .p2align 2, 0x0
"#mangled_data_sym":
        .word 0x01010101
        .globl data_sym
        .p2align 2, 0x0
data_sym:
        .word 0x01010101

#--- drectve.s
        .section .drectve, "yn"
        .ascii " -export:unmangled_func"
        .ascii " -export:#mangled_func"
        .ascii " -export:#exportas_func,EXPORTAS,exportas_func"
        .ascii " -export:?cxx_func@@$$hYAHXZ"
        .ascii " -export:data_sym,DATA"
        .ascii " -export:#mangled_data_sym,DATA"
