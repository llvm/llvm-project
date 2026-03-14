// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
// RUN: ld.lld %t -o %t.so -shared
// RUN: llvm-objdump -d --no-show-raw-insn %t.so | FileCheck --check-prefix=DISASM %s
// RUN: llvm-readobj --symbols -r %t.so | FileCheck -check-prefix=SYMBOL %s

// DISASM: <_start>:
// DISASM:    1330:       callq   0x133f <__start_foo>
// DISASM:    1335:       callq   0x1342 <__start_bar>
// DISASM:    133a:       callq   0x1342 <__start_bar>
// DISASM: Disassembly of section foo:
// DISASM-EMPTY:
// DISASM: <__start_foo>:
// DISASM:    133f:       nop
// DISASM:                nop
// DISASM:                nop
// DISASM: Disassembly of section bar:
// DISASM-EMPTY:
// DISASM: <__start_bar>:
// DISASM:    1342:       nop
// DISASM:                nop
// DISASM:                nop

// SYMBOL:      Relocations [
// SYMBOL-NEXT:   Section ({{.*}}) .rela.dyn {
// SYMBOL-NEXT:     R_X86_64_RELATIVE
// SYMBOL-NEXT:     R_X86_64_RELATIVE
// SYMBOL-NEXT:     R_X86_64_RELATIVE
// SYMBOL-NEXT:     R_X86_64_RELATIVE
// SYMBOL-NEXT:   }
// SYMBOL-NEXT: ]

// SYMBOL: Symbol {
// SYMBOL:   Name: __start_foo
// SYMBOL:   Value: 0x133F
// SYMBOL:   STV_HIDDEN
// SYMBOL:   Section: foo
// SYMBOL: }
// SYMBOL: Symbol {
// SYMBOL:   Name: __stop_foo
// SYMBOL:   Value: 0x1342
// SYMBOL:   STV_HIDDEN
// SYMBOL:   Section: foo
// SYMBOL: }
// SYMBOL: Symbol {
// SYMBOL:   Name: __start_bar
// SYMBOL:   Value: 0x1342
// SYMBOL:   STV_HIDDEN
// SYMBOL:   Section: bar
// SYMBOL: }
// SYMBOL-NOT:   Section: __stop_bar

// SYMBOL: Symbol {
// SYMBOL:   Name: __stop_zed2
// SYMBOL:   Value: 0x3418
// SYMBOL:   STV_PROTECTED
// SYMBOL:   Section: zed2
// SYMBOL: }
// SYMBOL: Symbol {
// SYMBOL:   Name: __stop_zed1
// SYMBOL:   Value: 0x3408
// SYMBOL:   STV_PROTECTED
// SYMBOL:   Section: zed1
// SYMBOL: }

.hidden __start_foo
.hidden __stop_foo
.hidden __start_bar
.global _start
.text
_start:
	call __start_foo
	call __stop_foo
	call __start_bar

.section foo,"ax"
	nop
	nop
	nop

.section bar,"ax"
	nop
	nop
	nop

.section zed1, "aw"
        .quad __stop_zed2
        .quad __stop_zed2 + 1

.section zed2, "aw"
        .quad __stop_zed1
        .quad __stop_zed1 + 1
