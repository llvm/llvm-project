# REQUIRES: x86

# Test that a non-leader symbol from a superseded
# IMAGE_COMDAT_SELECT_LARGEST group can be replaced by providers of other
# symbol kinds.
#
# The smaller candidate defines all provider symbols as regular external
# symbols. The larger candidate supersedes that group without defining them.
# A subsequent provider must then become the prevailing definition.

# RUN: split-file %s %t.dir

# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/small.s -o %t.small.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/large.s -o %t.large.obj

# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/provider-regular.s -o %t.provider-regular.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/provider-comdat.s -o %t.provider-comdat.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/provider-common.s -o %t.provider-common.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/provider-absolute.s -o %t.provider-absolute.obj

# RUN: sed 's/SYMBOL/foo_regular/g' %t.dir/root.s | \
# RUN:   llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:     -o %t.root-regular.obj
# RUN: sed 's/SYMBOL/foo_comdat/g' %t.dir/root.s | \
# RUN:   llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:     -o %t.root-comdat.obj
# RUN: sed 's/SYMBOL/foo_common/g' %t.dir/root.s | \
# RUN:   llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:     -o %t.root-common.obj
# RUN: sed 's/SYMBOL/foo_absolute/g' %t.dir/root.s | \
# RUN:   llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:     -o %t.root-absolute.obj

# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/root-synthetic.s -o %t.root-synthetic.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/root-import.s -o %t.root-import.obj

# RUN: llvm-dlltool -m i386:x86-64 -d %t.dir/provider.def \
# RUN:   -l %t.provider-import.lib

# DefinedRegular control case.

# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.large.obj %t.provider-regular.obj \
# RUN:   %t.root-regular.obj /map:%t.regular.map \
# RUN:   /out:%t.regular.exe
# RUN: FileCheck --check-prefix=REGULAR %s < %t.regular.map

# Under /force:multiple, retain a regular provider that was ignored while the
# smaller group prevailed.
# RUN: lld-link /force:multiple /opt:ref /entry:entry /subsystem:console \
# RUN:   /nodefaultlib %t.small.obj %t.provider-regular.obj %t.large.obj \
# RUN:   %t.root-regular.obj /map:%t.regular-between.map \
# RUN:   /out:%t.regular-between.exe
# RUN: FileCheck --check-prefix=REGULAR %s < %t.regular-between.map

# A later COMDAT leader must replace the discarded regular provider.

# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.large.obj %t.provider-comdat.obj \
# RUN:   %t.root-comdat.obj /map:%t.comdat.map \
# RUN:   /out:%t.comdat.exe
# RUN: FileCheck --check-prefix=COMDAT %s < %t.comdat.map

# A common symbol must replace the discarded regular provider.

# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.large.obj %t.provider-common.obj \
# RUN:   %t.root-common.obj /map:%t.common.map \
# RUN:   /out:%t.common.exe
# RUN: FileCheck --check-prefix=COMMON %s < %t.common.map

# A common provider seen before the larger candidate must remain available if
# the regular definition that suppressed it is superseded.
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.provider-common.obj %t.large.obj \
# RUN:   %t.root-common.obj /map:%t.common-between.map \
# RUN:   /out:%t.common-between.exe
# RUN: FileCheck --check-prefix=COMMON %s < %t.common-between.map

# An absolute symbol must replace the discarded regular provider.

# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.large.obj %t.provider-absolute.obj \
# RUN:   %t.root-absolute.obj /map:%t.absolute.map \
# RUN:   /out:%t.absolute.exe
# RUN: FileCheck --check-prefix=ABSOLUTE %s < %t.absolute.map

# The same ordering rule applies to absolute providers.
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.provider-absolute.obj %t.large.obj \
# RUN:   %t.root-absolute.obj /map:%t.absolute-between.map \
# RUN:   /out:%t.absolute-between.exe
# RUN: FileCheck --check-prefix=ABSOLUTE %s < %t.absolute-between.map

# Linker-generated synthetic and absolute symbols must replace discarded
# regular definitions with the same names. __guard_flags exercises the
# addAbsolute(StringRef, uint64_t) overload.

# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.large.obj %t.root-synthetic.obj \
# RUN:   /map:%t.synthetic.map /out:%t.synthetic.exe
# RUN: FileCheck --check-prefix=SYNTHETIC %s < %t.synthetic.map

# Force the import-library members to be loaded so that both the import thunk
# and the __imp_ data symbol encounter the discarded regular providers.

# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.large.obj \
# RUN:   /wholearchive:%t.provider-import.lib %t.root-import.obj \
# RUN:   /map:%t.import.map /out:%t.import.exe
# RUN: FileCheck --check-prefix=IMPORT %s < %t.import.map

# The map must associate each symbol with the replacement provider.

# REGULAR: foo_regular{{.*}}provider-regular.obj
# COMDAT: foo_comdat{{.*}}provider-comdat.obj
# COMMON: foo_common{{.*}}<common>
# ABSOLUTE: foo_absolute{{.*}}<absolute>
# SYNTHETIC-DAG: __ImageBase{{.*}}<linker-defined>
# SYNTHETIC-DAG: __guard_flags{{.*}}<absolute>

# The function import provides both a thunk and an import-address-table symbol.

# IMPORT-DAG: {{[[:space:]]foo_import[[:space:]]}}{{.*}}provider.dll
# IMPORT-DAG: {{[[:space:]]__imp_foo_import[[:space:]]}}{{.*}}provider.dll

#--- small.s
        .section .text$largest, "xr", largest, largest_leader
        .globl largest_leader
largest_leader:
        .byte 0x11

        .globl foo_regular
foo_regular:
        .byte 0x12

        .globl foo_comdat
foo_comdat:
        .byte 0x13

        .globl foo_common
foo_common:
        .byte 0x14

        .globl foo_absolute
foo_absolute:
        .byte 0x15

        .globl foo_import
foo_import:
        .byte 0x16

        .globl __imp_foo_import
__imp_foo_import:
        .byte 0x17

        .globl __ImageBase
__ImageBase:
        .byte 0x18

        .globl __guard_flags
__guard_flags:
        .byte 0x19

#--- large.s
        .section .text$largest, "xr", largest, largest_leader
        .globl largest_leader
largest_leader:
        .space 32, 0x44

#--- provider-regular.s
        .section .text$provider, "xr"
        .globl foo_regular
foo_regular:
        .long 0x55555555

#--- provider-comdat.s
        .section .text$provider, "xr", discard, foo_comdat
        .globl foo_comdat
foo_comdat:
        .long 0x66666666

#--- provider-common.s
        .comm foo_common, 8, 3

#--- provider-absolute.s
        .globl foo_absolute
        .set foo_absolute, 0x1234

#--- root.s
        .text
        .globl entry
entry:
        movabsq $SYMBOL, %rax
        retq

#--- root-synthetic.s
        .text
        .globl entry
entry:
        retq

#--- root-import.s
        .text
        .globl entry
entry:
        callq foo_import
        movq __imp_foo_import(%rip), %rax
        retq

#--- provider.def
LIBRARY provider.dll
EXPORTS
        foo_import
