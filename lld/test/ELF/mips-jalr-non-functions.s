# REQUIRES: mips
## Check that we ignore R_MIPS_JALR relocations agains non-function symbols.
## Older versions of clang were erroneously generating them for function pointers
## loaded from any table (not just the GOT) as well as against TLS function
## pointers (when using the local-dynamic model), so we need to ignore these
## relocations to avoid generating binaries that crash when executed.

# RUN: llvm-mc -filetype=obj -triple=mips64-unknown-linux %s -o %t.o
## Link in another object file with a .bss as a regression test:
## Previously LLD asserted when skipping over .bss sections when determining the
## location for a warning/error message. By adding another file with a .bss
## section before the actual %t.o we can reproduce this case.
# RUN: llvm-mc -filetype=obj -triple=mips64-unknown-linux %S/Inputs/common.s -o %t-common.o
# RUN: ld.lld -shared %t-common.o %t.o -o %t.so 2>&1 | FileCheck %s -check-prefix WARNING-MESSAGE
# RUN: llvm-objdump --no-show-raw-insn --no-leading-addr -d %t.so | FileCheck %s

.set	noreorder
test:
  .reloc .Ltmp1, R_MIPS_JALR, tls_obj
.Ltmp1:
  jr  $t9
  nop
# WARNING-MESSAGE: warning: {{.+}}.tmp.o:(.text+0x0): found R_MIPS_JALR relocation against non-function symbol tls_obj. This is invalid and most likely a compiler bug.

  .reloc .Ltmp2, R_MIPS_JALR, reg_obj
.Ltmp2:
  jr  $t9
  nop
# WARNING-MESSAGE: warning: {{.+}}.tmp.o:(.text+0x8): found R_MIPS_JALR relocation against non-function symbol reg_obj. This is invalid and most likely a compiler bug.

  .reloc .Ltmp3, R_MIPS_JALR, untyped
.Ltmp3:
  jr  $t9
  nop

## However, we do perform the optimization for untyped symbols:
untyped:
  nop

  .type  tls_obj,@object
  .section  .tbss,"awT",@nobits
tls_obj:
  .word 0

  .type  reg_obj,@object
  .data
reg_obj:
  .word 0

# CHECK-LABEL: Disassembly of section .text:
# CHECK-EMPTY:
# CHECK-NEXT: <test>:
# CHECK-NEXT: jr	$25
# CHECK-NEXT: nop
# CHECK-NEXT: jr	$25
# CHECK-NEXT: nop
# CHECK-NEXT: b	8 <untyped>
# CHECK-NEXT: nop
