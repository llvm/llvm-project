# REQUIRES: aarch64

## Different symbols and addends identify the same callee at a nonzero offset:
## _target == _alias == _target_start+64 == _target_mid+32. They must share an
## island, while _target+4 needs a separate island for the next instruction.
##
## Final layout:
##
##   +0          +0x6000018       +0xc000020       +0xc000060    +0xc000064
##   _main       two islands     _target_start   _target       _target_end
##                               |---- 64 B ----|
##
## Expected routing:
##   _main (first four calls) --> _target.island.0 --> _target
##   _main (last call) --> _target+4.island.0 --> _target_end

# RUN: rm -rf %t; mkdir %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t/input.o
# RUN: %lld -arch arm64 -dylib -o %t/out %t/input.o \
# RUN:   -arm64-islands-max-hops 2 --verbose 2> %t/link.log
# RUN: FileCheck %s --check-prefix=LOG --input-file=%t/link.log
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t/out \
# RUN:   | FileCheck %s --check-prefix=DIS

# LOG: branch extender for __TEXT:
# LOG-SAME: total extenders = 2

# DIS-LABEL: <_main>:
# DIS-NEXT: bl 0x[[#%x, ISLAND:]] <_target.island.0>
# DIS-NEXT: bl 0x[[#ISLAND]] <_target.island.0>
# DIS-NEXT: bl 0x[[#ISLAND]] <_target.island.0>
# DIS-NEXT: bl 0x[[#ISLAND]] <_target.island.0>
# DIS-NEXT: bl 0x[[#%x, NEXT_ISLAND:]] <_target+4.island.0>
# DIS-NEXT: ret
# DIS: [[#ISLAND]] <_target.island.0>:
# DIS-NEXT: b 0x[[#%x, TARGET:]] <_target>
# DIS: [[#NEXT_ISLAND]] <_target+4.island.0>:
# DIS-NEXT: b 0x[[#TARGET + 4]] <_target_end>

.subsections_via_symbols
.text
.globl _main
_main:
  bl _target
  bl _alias
  bl _target_start+64
  bl _target_mid+32
  bl _target+4
  ret

_pad0:
  .space 0x6000000

_pad1:
  .space 0x6000000

.globl _target_start
_target_start:
  .space 32
.globl _target_mid
.alt_entry _target_mid
_target_mid:
  .space 32
.globl _target
.alt_entry _target
_target:
  ret
.globl _alias
.alt_entry _alias
_alias = _target

.globl _target_end
.alt_entry _target_end
_target_end:
  ret
