## Checks that symbols are allocated in correct sections, and that empty
## fragments are not allocated at all.

# REQUIRES: x86_64-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: %clangxx %cxxflags %t.o -o %t.exe -Wl,-q -no-pie
# RUN: llvm-bolt %t.exe -o %t.bolt --split-functions --split-strategy=all \
# RUN:         --print-split --print-only=_Z3foov 2>&1 | \
# RUN:     FileCheck %s --check-prefix=CHECK-SPLIT
# RUN: llvm-nm %t.bolt | FileCheck %s --check-prefix=CHECK-COLD0
# RUN: llvm-objdump --syms %t.bolt | \
# RUN:     FileCheck %s --check-prefix=CHECK-SYMS

# CHECK-SPLIT: .LLP0 (4 instructions, align : 1)
# CHECK-SPLIT: -------   HOT-COLD SPLIT POINT   -------
# CHECK-SPLIT-EMPTY:
# CHECK-SPLIT-NEXT: -------   HOT-COLD SPLIT POINT   -------
# CHECK-SPLIT-EMPTY:
# CHECK-SPLIT-NEXT: .LFT0 (2 instructions, align : 1)

# CHECK-COLD0-NOT: _Z3foov.cold.0

# CHECK-SYMS: .text.cold.1
# CHECK-SYMS-SAME: _Z3foov.cold.1
# CHECK-SYMS: .text.cold.2
# CHECK-SYMS-SAME: _Z3foov.cold.2
# CHECK-SYMS: .text.cold.3
# CHECK-SYMS-SAME: _Z3foov.cold.3


        .text
        .globl  _Z3barv
        .type   _Z3barv, @function
_Z3barv:                            # void bar();
        .cfi_startproc
        ret
        .cfi_endproc
        .size   _Z3barv, .-_Z3barv


        .globl  _Z3bazv
        .type   _Z3bazv, @function
_Z3bazv:                            # void baz() noexcept;
        .cfi_startproc
        ret
        .cfi_endproc
        .size   _Z3bazv, .-_Z3bazv


        .globl  _Z3foov
        .type   _Z3foov, @function
_Z3foov:                            # void foo() noexcept;
.LFB1265:                           # _Z3foov
        .cfi_startproc
        .cfi_personality 0x3,__gxx_personality_v0
        .cfi_lsda 0x3,.LLSDA1265
        subq    $8, %rsp
        .cfi_def_cfa_offset 16
.LEHB0:
        call    _Z3barv             # LP: .L5
.LEHE0:
        jmp     .L4
.L5:                                # (_Z3foov.cold.0), landing pad, hot
        movq    %rax, %rdi
        cmpq    $1, %rdx
        je      .L3
        call    _ZSt9terminatev     # _Z3foov.cold.1
.L3:                                # _Z3foov.cold.2
        call    __cxa_begin_catch
        call    _Z3bazv
        call    __cxa_end_catch
.L4:                                # _Z3foov.cold.3
        call    _Z3bazv
        addq    $8, %rsp
        .cfi_def_cfa_offset 8
        ret
        .cfi_endproc
        .globl  __gxx_personality_v0
        .section        .gcc_except_table,"a",@progbits
        .align 4
.LLSDA1265:
        .byte   0xff
        .byte   0x3
        .uleb128 .LLSDATT1265-.LLSDATTD1265
.LLSDATTD1265:
        .byte   0x1
        .uleb128 .LLSDACSE1265-.LLSDACSB1265
.LLSDACSB1265:
        .uleb128 .LEHB0-.LFB1265
        .uleb128 .LEHE0-.LEHB0
        .uleb128 .L5-.LFB1265
        .uleb128 0x3
.LLSDACSE1265:
        .byte   0
        .byte   0
        .byte   0x1
        .byte   0x7d
        .align 4
        .long   _ZTISt13runtime_error
.LLSDATT1265:
        .text
        .size   _Z3foov, .-_Z3foov


        .globl  main
        .type   main, @function
main:
        .cfi_startproc
        subq    $8, %rsp
        .cfi_def_cfa_offset 16
        call    _Z3foov
        movl    $0, %eax
        addq    $8, %rsp
        .cfi_def_cfa_offset 8
        ret
        .cfi_endproc
        .size   main, .-main
