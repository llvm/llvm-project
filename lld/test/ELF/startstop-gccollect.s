# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t

## Default run: sections foo and bar exist in output
# RUN: ld.lld %t -o %tout
# RUN: llvm-objdump -d %tout | FileCheck -check-prefix=DISASM %s

## Check that foo and bar sections are not garbage collected,
## we do not want to reclaim sections if they can be referred
## by __start_* and __stop_* symbols.
# RUN: ld.lld %t --gc-sections -o %tout
# RUN: llvm-objdump -d %tout | FileCheck -check-prefix=DISASM %s

# DISASM:      _start:
# DISASM-NEXT:    201000:       90      nop
# DISASM-NEXT: Disassembly of section foo:
# DISASM-NEXT: foo:
# DISASM-NEXT:    201001:       90      nop
# DISASM-NEXT: Disassembly of section bar:
# DISASM-NEXT: bar:
# DISASM-NEXT:    201002:       90      nop

.global _start
.text
_start:
 nop

.section foo,"ax"
 nop

.section bar,"ax"
 nop
