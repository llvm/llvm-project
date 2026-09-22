# REQUIRES: aarch64

## A coincident alt entry belongs to the atom started by the ordinary symbol
## at the same address, even when the local alt entry precedes the external
## symbol in the Mach-O symbol table.

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/input.s -o %t/input.o
# RUN: %lld -arch arm64 -dylib -platform_version macos 13.0 13.0 \
# RUN:   -order_file %t/order.txt -map %t/map %t/input.o -o %t/out
# RUN: llvm-objdump --syms %t/out | FileCheck %s --check-prefix=SYMS
# RUN: llvm-objdump -d --no-show-raw-insn %t/out | \
# RUN:   FileCheck %s --check-prefix=DIS
# RUN: FileCheck %s --check-prefix=MAP --input-file=%t/map
# RUN: %lld -arch arm64 -dylib -platform_version macos 13.0 13.0 \
# RUN:   --icf=all %t/input.o -o %t/icf-out
# RUN: llvm-objdump --syms %t/icf-out | FileCheck %s --check-prefix=ICF

# SYMS-DAG: [[TARGET:[0-9a-f]+]] l     F __TEXT,__text target.alt
# SYMS-DAG: [[TARGET]] g     F __TEXT,__text _target

# DIS-LABEL: <{{(_target|target[.]alt)}}>:
# DIS:         mov
# DIS:         ret
# DIS-LABEL: <_main>:
# DIS-NEXT:    b {{.*}} <{{(_target|target[.]alt)}}>
# DIS-LABEL: <_previous>:
# DIS-NEXT:    ret

## Listing interior.alt before its atom's defining symbol must move the whole
## atom. If the alt entry split the atom, these labels would appear reversed.
# DIS-LABEL: <_interior_owner>:
# DIS:         mov
# DIS-LABEL: <interior.alt>:
# DIS-NEXT:    ret

## The ordinary symbol carries the body size; the coincident alt entry is a
## zero-sized alias.
# MAP:      0x[[TARGET_ADDR:[0-9A-F]+]]	0x00000000	{{.*}} target.alt
# MAP-NEXT: 0x[[TARGET_ADDR]]	0x00000008	{{.*}} _target

## The coincident alt entry must not mark _previous as containing an interior
## alt entry, so _previous remains eligible to fold with _fold_candidate.
# ICF-DAG: [[FOLD:[0-9a-f]+]] l     F __TEXT,__text _previous
# ICF-DAG: [[FOLD]] l     F __TEXT,__text _fold_candidate

#--- order.txt
_target
_main
_previous
interior.alt
_interior_owner

#--- input.s
.subsections_via_symbols
.text

.p2align 2
## Put the coincident symbols at a nonzero section offset. At offset zero,
## symbolOffset == 0 would bypass the behavior under test.
_previous:
  ret

.alt_entry target.alt
target.alt:
.globl _target
_target:
  mov w0, #42
  ret

.globl _main
_main:
  b target.alt

.globl _interior_owner
_interior_owner:
  mov w0, #7
.alt_entry interior.alt
interior.alt:
  ret

_fold_candidate:
  ret
