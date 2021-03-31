// REQUIRES: x86

/// Test -z ifunc-noplt.

// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-freebsd %S/Inputs/shared2-x86-64.s -o %t1.o
// RUN: ld.lld %t1.o --shared -soname=so -o %t.so
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-freebsd %s -o %t.o

/// The default -z text is not compatible with -z ifunc-noplt.
// RUN: not ld.lld -z ifunc-noplt %t.o -o /dev/null 2>&1| FileCheck --check-prefix=INCOMPATIBLE %s
// RUN: not ld.lld -z ifunc-noplt -z text %t.o -o /dev/null 2>&1| FileCheck --check-prefix=INCOMPATIBLE %s
// INCOMPATIBLE: -z text and -z ifunc-noplt may not be used together

// RUN: ld.lld -z ifunc-noplt -z notext --hash-style=sysv %t.so %t.o -o %tout
// RUN: llvm-objdump -d --no-show-raw-insn %tout | FileCheck %s --check-prefix=DISASM
// RUN: llvm-readobj -r --dynamic-table %tout | FileCheck %s

// Check that we emitted relocations for the ifunc calls
// CHECK: Relocations [
// CHECK-NEXT:   Section (4) .rela.dyn {
// CHECK-NEXT:     0x201323 R_X86_64_PLT32 foo 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0x201328 R_X86_64_PLT32 bar 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:   }
// CHECK-NEXT:   Section (5) .rela.plt {
// CHECK-NEXT:     0x203498 R_X86_64_JUMP_SLOT bar2 0x0
// CHECK-NEXT:     0x2034A0 R_X86_64_JUMP_SLOT zed2 0x0
// CHECK-NEXT:   }

// Check that ifunc call sites still require relocation
// DISASM: Disassembly of section .text:
// DISASM-EMPTY:
// DISASM-NEXT: 0000000000201320 <foo>:
// DISASM-NEXT:   201320:      	retq
// DISASM-EMPTY:
// DISASM-NEXT: 0000000000201321 <bar>:
// DISASM-NEXT:   201321:      	retq
// DISASM-EMPTY:
// DISASM-NEXT: 0000000000201322 <_start>:
// DISASM-NEXT:   201322:      	callq	0x201327 <_start+0x5>
// DISASM-NEXT:   201327:      	callq	0x20132c <_start+0xa>
// DISASM-NEXT:   20132c:      	callq	0x201350 <bar2@plt>
// DISASM-NEXT:   201331:      	callq	0x201360 <zed2@plt>
// DISASM-EMPTY:
// DISASM-NEXT: Disassembly of section .plt:
// DISASM-EMPTY:
// DISASM-NEXT: 0000000000201340 <.plt>:
// DISASM-NEXT:   201340:      	pushq	8514(%rip)
// DISASM-NEXT:   201346:      	jmpq	*8516(%rip)
// DISASM-NEXT:   20134c:      	nopl	(%rax)
// DISASM-EMPTY:
// DISASM-NEXT: 0000000000201350 <bar2@plt>:
// DISASM-NEXT:   201350:      	jmpq	*8514(%rip)
// DISASM-NEXT:   201356:      	pushq	$0
// DISASM-NEXT:   20135b:      	jmp	0x201340 <.plt>
// DISASM-EMPTY:
// DISASM-NEXT: 0000000000201360 <zed2@plt>:
// DISASM-NEXT:   201360:      	jmpq	*8506(%rip)
// DISASM-NEXT:   201366:      	pushq	$1
// DISASM-NEXT:   20136b:      	jmp	0x201340 <.plt>

.text
.type foo STT_GNU_IFUNC
.globl foo
foo:
 ret

.type bar STT_GNU_IFUNC
.globl bar
bar:
 ret

.globl _start
_start:
 call foo
 call bar
 call bar2
 call zed2
