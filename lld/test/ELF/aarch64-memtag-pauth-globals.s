# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-android %s -o %t.o
# RUN: ld.lld --shared --android-memtag-mode=sync %t.o -o %t
# RUN: llvm-readobj -r %t | FileCheck %s --check-prefix=RELA
# RUN: llvm-readelf -x.data %t | FileCheck %s --check-prefix=DATA

## Verify that, when composing PAuth and Memtag ABIs, R_AARCH64_AUTH_RELATIVE
## relocations follow R_AARCH64_RELATIVE in emitting the (negated) original
## addend at the relocated target for tagged globals when not within the
## symbol's bounds.

# RELA-LABEL: .rela.dyn {
# RELA-NEXT:    0x303C0 R_AARCH64_AUTH_RELATIVE - 0x303BF
# RELA-NEXT:    0x303C8 R_AARCH64_AUTH_RELATIVE - 0x303D0
# RELA-NEXT:  }

# DATA-LABEL: Hex dump of section '.data':
# DATA-NEXT:  0x000303c0 01000000 2a000020 f0ffffff 2a000020

.data
.balign 16
.memtag foo
.type foo, @object
foo:
.quad (foo - 1)@AUTH(da,42)
.quad (foo + 16)@AUTH(da,42)
.size foo, 16
