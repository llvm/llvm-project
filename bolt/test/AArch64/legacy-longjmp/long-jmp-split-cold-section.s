## Check that the LongJmp tentative layout places function fragments in their
## output sections. Without a profile, F and G stay in .text and the block split
## off from F goes to .text.cold, which follows .text by default and precedes it
## with --hot-functions-at-end. Either way, the TBZ in F cannot reach the block
## past G's 64KB constant island without a stub.

# RUN: llvm-mc -filetype=obj -triple=aarch64-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe -q --entry=F
# RUN: llvm-bolt %t.exe -o %t.bolt --split-functions --split-strategy=random2 \
# RUN:   | FileCheck %s
# RUN: llvm-objdump -d --no-show-raw-insn --disassemble-symbols=F %t.bolt \
# RUN:   | FileCheck %s --check-prefix=DISASM
# RUN: llvm-bolt %t.exe -o %t.hfe.bolt --split-functions \
# RUN:   --split-strategy=random2 --hot-functions-at-end | FileCheck %s
# RUN: llvm-objdump -d --no-show-raw-insn --disassemble-symbols=F %t.hfe.bolt \
# RUN:   | FileCheck %s --check-prefix=DISASM

# CHECK: BOLT-INFO: Inserted 1 stubs in the hot area and 0 stubs in the cold area

# DISASM:      <F>:
# DISASM-NEXT:   tbz w0, #0x0, 0x[[#%x,STUB:]]
# DISASM-NEXT:   ret
# DISASM-NEXT: [[#%x,STUB]]: b {{.*}} <F.cold.0>

  .text
  .globl F
  .type F, %function
F:
  tbz x0, #0, 1f
  ret
1:
  mov x0, #1
  ret
  .size F, .-F

  .globl G
  .type G, %function
G:
  ret
  .space 0x10000
  .size G, .-G

## Force relocation mode.
  .reloc 0, R_AARCH64_NONE
