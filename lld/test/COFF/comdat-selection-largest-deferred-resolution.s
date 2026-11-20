# REQUIRES: x86

# Exercise providers and references that are temporarily hidden by a secondary
# definition in an IMAGE_COMDAT_SELECT_LARGEST group. All orderings here must
# resolve as if the losing group had never published its symbols.

# RUN: split-file %s %t.dir
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/small.s -o %t.small.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/large.s -o %t.large.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/medium.s -o %t.medium.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/large-with-foo.s -o %t.large-with-foo.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/assoc-small.s -o %t.assoc-small.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/small-import.s -o %t.small-import.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/provider.s -o %t.provider.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/provider-a.s -o %t.provider-a.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/provider-b.s -o %t.provider-b.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/provider-comdat.s -o %t.provider-comdat.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/provider-comdat-small.s -o %t.provider-comdat-small.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/provider-comdat-large.s -o %t.provider-comdat-large.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/common.s -o %t.common.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/common-large.s -o %t.common-large.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/absolute-a.s -o %t.absolute-a.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/absolute-b.s -o %t.absolute-b.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/root.s -o %t.root.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/root-call.s -o %t.root-call.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/root-import.s -o %t.root-import.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/same-object.s -o %t.same-object.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/same-object-assoc.s -o %t.same-object-assoc.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/same-object-comdat.s -o %t.same-object-comdat.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/local-reference.s -o %t.local-reference.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/assoc-local-reference.s -o %t.assoc-local-reference.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/only-small-provider.s -o %t.only-small-provider.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/discarded-reference.s -o %t.discarded-reference.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/root-leader.s -o %t.root-leader.obj
# RUN: llvm-lib -machine:amd64 -out:%t.provider.lib %t.provider.obj
# RUN: llvm-lib -machine:amd64 -out:%t.provider-a.lib %t.provider-a.obj
# RUN: llvm-lib -machine:amd64 -out:%t.provider-b.lib %t.provider-b.obj
# RUN: llvm-lib -machine:amd64 -out:%t.only-small.lib \
# RUN:   %t.only-small-provider.obj
# RUN: llvm-dlltool -m i386:x86-64 -d %t.dir/provider.def \
# RUN:   -l %t.provider-import.lib

# A lazy provider registered before the provisional definition must survive.
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.provider.lib %t.small.obj %t.large.obj %t.root.obj \
# RUN:   /map:%t.provider-before.map /out:%t.provider-before.exe
# RUN: FileCheck %s --check-prefix=PROVIDER < %t.provider-before.map

# A provider remains deferred across multiple successive provisional winners.
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.provider.lib %t.medium.obj %t.large.obj %t.root.obj \
# RUN:   /map:%t.three-candidates.map /out:%t.three-candidates.exe
# RUN: FileCheck %s --check-prefix=PROVIDER < %t.three-candidates.map

# Explicit roots established through /include and /export are real references
# and must extract the recovered provider.
# RUN: lld-link /dll /noentry /nodefaultlib /include:foo %t.small.obj \
# RUN:   %t.provider.lib %t.large.obj /map:%t.include.map /out:%t.include.dll
# RUN: FileCheck %s --check-prefix=PROVIDER < %t.include.map
# RUN: lld-link /dll /noentry /nodefaultlib /export:foo %t.small.obj \
# RUN:   %t.provider.lib %t.large.obj /map:%t.export.map /out:%t.export.dll
# RUN: FileCheck %s --check-prefix=PROVIDER < %t.export.map
# RUN: llvm-readobj --coff-exports %t.export.dll | \
# RUN:   FileCheck %s --check-prefix=EXPORT

# The first archive retains normal precedence when several providers are
# suppressed between candidates.
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.provider-a.lib %t.provider-b.lib %t.large.obj \
# RUN:   %t.root.obj /map:%t.lazy-order.map /out:%t.lazy-order.exe
# RUN: FileCheck %s --check-prefix=FIRST-LAZY < %t.lazy-order.map

# An associative child inherits the provisional state of its leader.
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.assoc-small.obj %t.provider.lib %t.large.obj %t.root.obj \
# RUN:   /map:%t.assoc.map /out:%t.assoc.exe
# RUN: FileCheck %s --check-prefix=PROVIDER < %t.assoc.map

# /force:multiple must not warn about a provider whose apparent duplicate is
# removed by the later LARGEST selection.
# RUN: lld-link /force:multiple /opt:ref /entry:entry /subsystem:console \
# RUN:   /nodefaultlib %t.small.obj %t.provider.obj %t.large.obj %t.root.obj \
# RUN:   /out:%t.force-multiple.exe 2>&1 | \
# RUN:   FileCheck %s --allow-empty --check-prefix=NO-SPURIOUS-WARN

# If the provisional group remains selected, replay restores the duplicate
# diagnostic that normal symbol resolution would have produced.
# RUN: not lld-link /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.provider.obj %t.root.obj \
# RUN:   /out:%t.surviving-duplicate.exe 2>&1 | \
# RUN:   FileCheck %s --check-prefix=SURVIVING-DUPLICATE
# RUN: not lld-link /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.provider.obj %t.small.obj %t.root.obj \
# RUN:   /out:%t.preceding-duplicate.exe 2>&1 | \
# RUN:   FileCheck %s --check-prefix=SURVIVING-DUPLICATE

# A COMDAT provider is retained even though its section must be read before the
# current LARGEST group is superseded. Cover both sides of the provisional
# definition.
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.provider-comdat.obj %t.large.obj %t.root.obj \
# RUN:   /map:%t.comdat-between.map /out:%t.comdat-between.exe
# RUN: FileCheck %s --check-prefix=COMDAT < %t.comdat-between.map
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.provider-comdat.obj %t.small.obj %t.large.obj %t.root.obj \
# RUN:   /map:%t.comdat-before.map /out:%t.comdat-before.exe
# RUN: FileCheck %s --check-prefix=COMDAT < %t.comdat-before.map

# A deferred COMDAT that is rejected after the provisional winner survives is
# still discarded under /force:multiple.
# RUN: lld-link /force:multiple /opt:noref /entry:entry /subsystem:console \
# RUN:   /nodefaultlib %t.small.obj %t.provider-comdat.obj %t.root.obj \
# RUN:   /out:%t.comdat-not-selected.exe 2>&1 | \
# RUN:   FileCheck %s --check-prefix=COMDAT-DUPLICATE
# RUN: llvm-objdump -s %t.comdat-not-selected.exe | \
# RUN:   FileCheck %s --check-prefix=COMDAT-REJECT \
# RUN:   --implicit-check-not=cccccccc

# Multiple deferred COMDAT providers must go through normal selection rather
# than retaining the first provider unconditionally.
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   /include:provider_child \
# RUN:   %t.small.obj %t.provider-comdat-small.obj \
# RUN:   %t.provider-comdat-large.obj %t.large.obj %t.root.obj \
# RUN:   /out:%t.comdat-largest.exe
# RUN: llvm-objdump -s %t.comdat-largest.exe | \
# RUN:   FileCheck %s --check-prefix=COMDAT-LARGEST

# Eager imports loaded with /wholearchive work before and between candidates.
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj /wholearchive:%t.provider-import.lib %t.large.obj \
# RUN:   %t.root-call.obj /out:%t.import-between.exe
# RUN: llvm-readobj --coff-imports %t.import-between.exe | \
# RUN:   FileCheck %s --check-prefix=IMPORT
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   /wholearchive:%t.provider-import.lib %t.small.obj %t.large.obj \
# RUN:   %t.root-call.obj /out:%t.import-before.exe
# RUN: llvm-readobj --coff-imports %t.import-before.exe | \
# RUN:   FileCheck %s --check-prefix=IMPORT

# Exercise both symbols produced by a function import: the callable thunk and
# its __imp_ data entry.
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small-import.obj /wholearchive:%t.provider-import.lib \
# RUN:   %t.large.obj %t.root-import.obj /out:%t.import-pair.exe
# RUN: llvm-readobj --coff-imports %t.import-pair.exe | \
# RUN:   FileCheck %s --check-prefix=IMPORT
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   /wholearchive:%t.provider-import.lib %t.small-import.obj \
# RUN:   %t.large.obj %t.root-import.obj /out:%t.import-pair-before.exe
# RUN: llvm-readobj --coff-imports %t.import-pair-before.exe | \
# RUN:   FileCheck %s --check-prefix=IMPORT

# If the provisional definitions survive under /force:multiple, the eager
# import is a rejected duplicate and must not leak into the import table or
# leave replaced ImportFile symbol pointers behind.
# RUN: lld-link /force:multiple /opt:ref /entry:entry /subsystem:console \
# RUN:   /nodefaultlib %t.small-import.obj \
# RUN:   /wholearchive:%t.provider-import.lib %t.root-import.obj \
# RUN:   /out:%t.import-not-selected.exe 2>&1 | \
# RUN:   FileCheck %s --check-prefix=IMPORT-DUPLICATE
# RUN: llvm-readobj --coff-imports %t.import-not-selected.exe | \
# RUN:   FileCheck %s --allow-empty --check-prefix=NO-IMPORT

# If only the thunk name is shadowed, the IAT provider remains valid and the
# rejected thunk must not leave a replaced pointer in ImportFile.
# RUN: lld-link /force:multiple /opt:ref /entry:entry /subsystem:console \
# RUN:   /nodefaultlib /wholearchive:%t.provider-import.lib %t.small.obj \
# RUN:   %t.root-import.obj /out:%t.import-data-only.exe 2>&1 | \
# RUN:   FileCheck %s --check-prefix=IMPORT-THUNK-DUPLICATE
# RUN: llvm-readobj --coff-imports %t.import-data-only.exe | \
# RUN:   FileCheck %s --check-prefix=IMPORT

# A deferred common that is not selected must not leave an allocation behind.
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.common.obj %t.large-with-foo.obj %t.root.obj \
# RUN:   /out:%t.common-not-selected.exe
# RUN: llvm-readobj --sections %t.common-not-selected.exe | \
# RUN:   FileCheck %s --check-prefix=NO-COMMON

# Multiple deferred commons retain normal size precedence without keeping the
# allocations belonging to providers that were replayed but not selected.
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.common.obj %t.common-large.obj %t.large.obj \
# RUN:   %t.root.obj /out:%t.multiple-common.exe
# RUN: llvm-readobj --sections %t.multiple-common.exe | \
# RUN:   FileCheck %s --check-prefix=MULTIPLE-COMMON

# Common and absolute providers seen before the provisional group remain
# available if it loses.
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.common.obj %t.small.obj %t.large.obj %t.root.obj \
# RUN:   /map:%t.common-before.map /out:%t.common-before.exe
# RUN: FileCheck %s --check-prefix=COMMON < %t.common-before.map
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.absolute-a.obj %t.small.obj %t.large.obj %t.root.obj \
# RUN:   /map:%t.absolute-before.map /out:%t.absolute-before.exe
# RUN: FileCheck %s --check-prefix=ABSOLUTE < %t.absolute-before.map

# Replaying multiple absolute providers preserves the normal duplicate
# diagnostic instead of silently collapsing the list.
# RUN: not lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.absolute-a.obj %t.absolute-b.obj %t.large.obj \
# RUN:   %t.root.obj /out:%t.absolute-conflict.exe 2>&1 | \
# RUN:   FileCheck %s --check-prefix=ABSOLUTE-CONFLICT

# A relocation from a normal section in the same object is a real reference
# and therefore extracts a deferred provider after the group is discarded.
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.same-object.obj %t.only-small.lib %t.large.obj \
# RUN:   /map:%t.same-object.map /out:%t.same-object.exe
# RUN: FileCheck %s --check-prefix=SAME-OBJECT < %t.same-object.map

# The same provenance rule applies when the removed definition is in an
# associative child, and when the live reference originates in another COMDAT
# in the same object.
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.same-object-assoc.obj %t.only-small.lib %t.large.obj \
# RUN:   /map:%t.same-object-assoc.map /out:%t.same-object-assoc.exe
# RUN: FileCheck %s --check-prefix=ASSOC-SAME-OBJECT \
# RUN:   < %t.same-object-assoc.map
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.same-object-comdat.obj %t.only-small.lib %t.large.obj \
# RUN:   /map:%t.same-object-comdat.map /out:%t.same-object-comdat.exe
# RUN: FileCheck %s --check-prefix=SAME-OBJECT \
# RUN:   < %t.same-object-comdat.map

# Local symbols cannot be satisfied by another file. Diagnose live relocations
# to a discarded leader or associative child instead of emitting an invalid
# RVA.
# RUN: not lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.local-reference.obj %t.large.obj /out:%t.local-reference.exe \
# RUN:   2>&1 | FileCheck %s --check-prefix=LOCAL
# RUN: not lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.assoc-local-reference.obj %t.large.obj \
# RUN:   /out:%t.assoc-local-reference.exe 2>&1 | \
# RUN:   FileCheck %s --check-prefix=ASSOC-LOCAL

# Conversely, an undefined reference originating only in the losing group is
# removed with that group.
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.discarded-reference.obj %t.large.obj %t.root-leader.obj \
# RUN:   /out:%t.discarded-reference.exe
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.large.obj %t.discarded-reference.obj %t.root-leader.obj \
# RUN:   /out:%t.discarded-reference-reversed.exe

# PROVIDER: foo{{.*}}provider.obj
# FIRST-LAZY: foo{{.*}}provider-a.obj
# FIRST-LAZY-NOT: provider-b.obj
# COMDAT: foo{{.*}}provider-comdat.obj
# COMDAT-DUPLICATE: duplicate symbol: foo
# COMDAT-REJECT: 11111111
# COMDAT-LARGEST: Contents of section .rdata:
# COMDAT-LARGEST: 22222222 22222222 22222222 22222222
# COMDAT-LARGEST-NEXT: 44444444
# IMPORT: Symbol: foo
# IMPORT-DUPLICATE: duplicate symbol: __declspec(dllimport) foo
# IMPORT-THUNK-DUPLICATE: duplicate symbol: foo
# NO-IMPORT-NOT: Symbol: foo
# EXPORT: Name: foo
# NO-COMMON-NOT: Name: .data
# NO-COMMON-NOT: Name: .bss
# MULTIPLE-COMMON: Name: .data
# MULTIPLE-COMMON-NEXT: VirtualSize: 0x10
# COMMON: foo{{.*}}<common>
# ABSOLUTE: foo{{.*}}<absolute>
# ABSOLUTE-CONFLICT: duplicate symbol: foo
# SURVIVING-DUPLICATE: duplicate symbol: foo
# SAME-OBJECT: only_small{{.*}}only-small-provider.obj
# ASSOC-SAME-OBJECT: assoc_only{{.*}}only-small-provider.obj
# NO-SPURIOUS-WARN-NOT: duplicate symbol
# LOCAL: relocation against symbol in discarded section: local_only
# ASSOC-LOCAL: relocation against symbol in discarded section: assoc_local

#--- small.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .byte 0x11
        .globl foo
foo:
        .long 0x11111111
        .globl provider_child
provider_child:
        .long 0x33333333

#--- assoc-small.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .byte 0x11
        .section .rdata$assoc, "dr", associative, leader
        .globl foo
foo:
        .long 0x11111111

#--- large.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .space 32, 0x44

#--- medium.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .space 16, 0x33
        .globl foo
foo:
        .long 0x33333333

#--- small-import.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .byte 0x11
        .globl foo
foo:
        .byte 0x22
        .globl __imp_foo
__imp_foo:
        .quad 0

#--- large-with-foo.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .space 32, 0x44
        .globl foo
foo:
        .long 0x66666666

#--- provider.s
        .section .rdata$provider, "dr"
        .globl foo
foo:
        .long 0x55555555

#--- provider-a.s
        .section .rdata$a, "dr"
        .globl foo
foo:
        .long 0xaaaaaaaa

#--- provider-b.s
        .section .rdata$b, "dr"
        .globl foo
foo:
        .long 0xbbbbbbbb

#--- provider-comdat.s
        .section .rdata$provider, "dr", discard, foo
        .globl foo
foo:
        .long 0xcccccccc

#--- provider-comdat-small.s
        .section .rdata$foo, "dr", largest, foo
        .globl foo
foo:
        .long 0x11111111

#--- provider-comdat-large.s
        .section .rdata$foo, "dr", largest, foo
        .globl foo
foo:
        .space 16, 0x22
        .globl provider_child
provider_child:
        .long 0x44444444

#--- common.s
        .comm foo, 8, 3

#--- common-large.s
        .comm foo, 16, 4

#--- absolute-a.s
        .globl foo
        .set foo, 0x1111

#--- absolute-b.s
        .globl foo
        .set foo, 0x2222

#--- root.s
        .text
        .globl entry
entry:
        movabsq $foo, %rax
        retq

#--- root-call.s
        .text
        .globl entry
entry:
        callq foo
        retq

#--- root-import.s
        .text
        .globl entry
entry:
        callq foo
        movq __imp_foo(%rip), %rax
        retq

#--- same-object.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .byte 0x11
        .globl only_small
only_small:
        retq
        .text
        .globl entry
entry:
        callq only_small
        retq

#--- same-object-assoc.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .byte 0x11
        .section .text$assoc, "xr", associative, leader
        .globl assoc_only
assoc_only:
        retq
        .text
        .globl entry
entry:
        callq assoc_only
        retq

#--- same-object-comdat.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .byte 0x11
        .globl only_small
only_small:
        retq
        .section .text$entry, "xr", discard, entry
        .globl entry
entry:
        callq only_small
        retq

#--- only-small-provider.s
        .text
        .globl only_small
only_small:
        retq
        .globl assoc_only
assoc_only:
        retq

#--- local-reference.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .byte 0x11
local_only:
        retq
        .text
        .globl entry
entry:
        callq local_only
        retq

#--- assoc-local-reference.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .byte 0x11
        .section .text$assoc, "xr", associative, leader
assoc_local:
        retq
        .text
        .globl entry
entry:
        callq assoc_local
        retq

#--- discarded-reference.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        callq missing
        retq

#--- root-leader.s
        .text
        .globl entry
entry:
        leaq leader(%rip), %rax
        retq

#--- provider.def
LIBRARY provider.dll
EXPORTS
        foo
