# REQUIRES: x86
## Test --btf-merge: merging and deduplication of .BTF sections.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 empty.s -o empty.o

## Without --btf-merge, input .BTF sections pass through as-is.
# RUN: ld.lld a.o b.o -o no-merge
# RUN: llvm-readelf -S no-merge | FileCheck %s --check-prefix=NO-MERGE

## --btf-merge produces a single merged .BTF section.
# RUN: ld.lld --btf-merge a.o b.o -o merged
# RUN: llvm-readelf -S merged | FileCheck %s --check-prefix=MERGED
# RUN: llvm-readelf -x .BTF merged | FileCheck %s --check-prefix=BTF-HEX

## --no-btf-merge disables merging.
# RUN: ld.lld --btf-merge --no-btf-merge a.o b.o -o disabled
# RUN: llvm-readelf -S disabled | FileCheck %s --check-prefix=NO-MERGE

## No .BTF input: no .BTF output.
# RUN: ld.lld --btf-merge empty.o -o no-btf
# RUN: llvm-readelf -S no-btf | FileCheck %s --check-prefix=NO-BTF

## Single file: passthrough.
# RUN: ld.lld --btf-merge a.o -o single
# RUN: llvm-readelf -x .BTF single | FileCheck %s --check-prefix=BTF-HEX

## Dedup: both files have INT "int"; after merge only one should remain.
# RUN: llvm-mc -filetype=obj -triple=x86_64 dup.s -o dup.o
# RUN: ld.lld --btf-merge a.o dup.o -o dedup
# RUN: llvm-readelf -x .BTF dedup | FileCheck %s --check-prefix=BTF-HEX

## -r should not merge .BTF sections even with --btf-merge.
## The .BTF content is the raw concatenation of both inputs (size 0x5b = 91),
## not a parsed/deduped blob.
# RUN: ld.lld -r --btf-merge a.o b.o -o reloc.o
# RUN: llvm-readelf -S reloc.o | FileCheck %s --check-prefix=RELOC

## --gc-sections: verify --btf-merge works together with --gc-sections.
# RUN: ld.lld --btf-merge --gc-sections a.o b.o -o gc
# RUN: llvm-readelf -x .BTF gc | FileCheck %s --check-prefix=BTF-HEX

## isLive: a .BTF section in a discarded COMDAT group should be skipped.
## Both comdat files define the same group "grp". The second is discarded,
## so only the first file's .BTF (INT "int") is merged with a.o's.
# RUN: llvm-mc -filetype=obj -triple=x86_64 comdat1.s -o comdat1.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 comdat2.s -o comdat2.o
# RUN: ld.lld --btf-merge a.o comdat1.o comdat2.o -o comdat
# RUN: llvm-readelf -x .BTF comdat | FileCheck %s --check-prefix=BTF-HEX

# NO-MERGE: .BTF
# MERGED:     .BTF PROGBITS
# MERGED-NOT: .BTF PROGBITS
# NO-BTF-NOT: .BTF
# BTF-HEX: Hex dump of section '.BTF':
# BTF-HEX: 0x{{[0-9a-f]+}} 9feb0100

## -r: raw concatenation of both inputs (45 + 46 = 91 = 0x5b bytes).
# RELOC: .BTF PROGBITS {{.*}} 00005b

#--- a.s
.text
.globl _start
_start:
  ret

.section .BTF,"",@progbits
.short 0xeb9f           # magic
.byte 1                 # version
.byte 0                 # flags
.long 24                # hdr_len
.long 0                 # type_off
.long 16                # type_len
.long 16                # str_off
.long 5                 # str_len
## Type 1: INT "int" size=4
.long 1                 # name_off
.long 0x01000000        # info: kind=INT(1), vlen=0
.long 4                 # size
.long 0x00000020        # encoding: bits=32
## String table: "\0int\0"
.byte 0
.ascii "int"
.byte 0

#--- b.s
.text
.globl bar
.type bar, @function
bar:
  ret

.section .BTF,"",@progbits
.short 0xeb9f           # magic
.byte 1                 # version
.byte 0                 # flags
.long 24                # hdr_len
.long 0                 # type_off
.long 16                # type_len
.long 16                # str_off
.long 6                 # str_len
## Type 1: INT "long" size=8
.long 1                 # name_off
.long 0x01000000        # info: kind=INT(1), vlen=0
.long 8                 # size
.long 0x00000040        # encoding: bits=64
## String table: "\0long\0"
.byte 0
.ascii "long"
.byte 0

#--- empty.s
.text
.globl _start
_start:
  ret

#--- dup.s
.text
.globl dup_fn
.type dup_fn, @function
dup_fn:
  ret

.section .BTF,"",@progbits
.short 0xeb9f           # magic
.byte 1                 # version
.byte 0                 # flags
.long 24                # hdr_len
.long 0                 # type_off
.long 16                # type_len
.long 16                # str_off
.long 5                 # str_len
## Type 1: INT "int" size=4 (identical to a.s)
.long 1                 # name_off
.long 0x01000000        # info: kind=INT(1), vlen=0
.long 4                 # size
.long 0x00000020        # encoding: bits=32
## String table: "\0int\0"
.byte 0
.ascii "int"
.byte 0

#--- comdat1.s
## COMDAT group "grp" with a .BTF section containing INT "int".
.section .text.foo,"axG",@progbits,grp,comdat
.globl foo
foo:
  ret

.section .BTF,"G",@progbits,grp,comdat
.short 0xeb9f           # magic
.byte 1                 # version
.byte 0                 # flags
.long 24                # hdr_len
.long 0                 # type_off
.long 16                # type_len
.long 16                # str_off
.long 5                 # str_len
## Type 1: INT "int" size=4
.long 1                 # name_off
.long 0x01000000        # info: kind=INT(1), vlen=0
.long 4                 # size
.long 0x00000020        # encoding: bits=32
## String table: "\0int\0"
.byte 0
.ascii "int"
.byte 0

#--- comdat2.s
## Same COMDAT group "grp"; this copy will be discarded during dedup.
## Its .BTF section (INT "long") should NOT be merged.
.section .text.foo,"axG",@progbits,grp,comdat
.globl foo
foo:
  ret

.section .BTF,"G",@progbits,grp,comdat
.short 0xeb9f           # magic
.byte 1                 # version
.byte 0                 # flags
.long 24                # hdr_len
.long 0                 # type_off
.long 16                # type_len
.long 16                # str_off
.long 6                 # str_len
## Type 1: INT "long" size=8 (should be discarded)
.long 1                 # name_off
.long 0x01000000        # info: kind=INT(1), vlen=0
.long 8                 # size
.long 0x00000040        # encoding: bits=64
## String table: "\0long\0"
.byte 0
.ascii "long"
.byte 0
