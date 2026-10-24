# REQUIRES: x86

# Test that a regular definition from a lazily scanned archive can replace a
# non-leader symbol from a superseded IMAGE_COMDAT_SELECT_LARGEST group.
#
# The primary regression case scans provider.lib after the smaller group has
# been discarded but before a real undefined reference to foo_lazy appears.
# The discarded definition must not prevent registration and later extraction
# of the lazy archive member.

# RUN: split-file %s %t.dir
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/small.s -o %t.small.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/large.s -o %t.large.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/large-with-symbol.s -o %t.large-with-symbol.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/provider.s -o %t.provider.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/root.s -o %t.root.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/entry-only.s -o %t.entry-only.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/provider-dll.s -o %t.provider-dll.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/root-dll.s -o %t.root-dll.obj
# RUN: llvm-lib -machine:amd64 -out:%t.provider.lib %t.provider.obj
# RUN: lld-link /dll /noentry /nodefaultlib /export:foo_lazy \
# RUN:   %t.provider-dll.obj /out:%t.provider-dll.dll

# Archive scanned before the undefined reference: regression case.
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.large.obj %t.provider.lib %t.root.obj \
# RUN:   /map:%t.archive-first.map /out:%t.archive-first.exe
# RUN: llvm-objdump -s %t.archive-first.exe | FileCheck --check-prefix=IMAGE %s
# RUN: FileCheck --check-prefix=MAP %s < %t.archive-first.map

# Preserve an archive provider seen while the smaller group still prevails.
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.provider.lib %t.large.obj %t.root.obj \
# RUN:   /map:%t.archive-between.map /out:%t.archive-between.exe
# RUN: llvm-objdump -s %t.archive-between.exe | \
# RUN:   FileCheck --check-prefix=IMAGE %s
# RUN: FileCheck --check-prefix=MAP %s < %t.archive-between.map

# Merely deferring a provider between candidates must not extract it without a
# real reference.
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.provider.lib %t.large.obj %t.entry-only.obj \
# RUN:   /out:%t.unreferenced-between.exe
# RUN: llvm-objdump -s %t.unreferenced-between.exe | \
# RUN:   FileCheck --check-prefix=UNREFERENCED %s

# Do not extract the deferred provider if the replacing group defines the
# symbol itself.
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.provider.lib %t.large-with-symbol.obj %t.root.obj \
# RUN:   /out:%t.redefined-by-larger.exe
# RUN: llvm-objdump -s %t.redefined-by-larger.exe | \
# RUN:   FileCheck --check-prefix=REDEFINED %s

# Without a real reference, scanning the archive must register the lazy
# provider without extracting its member.
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.large.obj %t.provider.lib %t.entry-only.obj \
# RUN:   /out:%t.unreferenced.exe
# RUN: llvm-objdump -s %t.unreferenced.exe | \
# RUN:   FileCheck --check-prefix=UNREFERENCED %s

# Undefined reference seen before the archive: control case.
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.large.obj %t.root.obj %t.provider.lib \
# RUN:   /map:%t.reference-first.map /out:%t.reference-first.exe
# RUN: llvm-objdump -s %t.reference-first.exe | \
# RUN:   FileCheck --check-prefix=IMAGE %s
# RUN: FileCheck --check-prefix=MAP %s < %t.reference-first.map

# A /start-lib object follows the LazyObject path rather than LazyArchive.

# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj /start-lib %t.provider.obj /end-lib %t.large.obj \
# RUN:   %t.root.obj /map:%t.start-lib-between.map \
# RUN:   /out:%t.start-lib-between.exe
# RUN: llvm-objdump -s %t.start-lib-between.exe | \
# RUN:   FileCheck --check-prefix=IMAGE %s
# RUN: FileCheck --check-prefix=MAP %s < %t.start-lib-between.map

# A /start-lib provider registered before the provisional group must also be
# restored after replacement.
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   /start-lib %t.provider.obj /end-lib %t.small.obj %t.large.obj \
# RUN:   %t.root.obj /map:%t.start-lib-before.map \
# RUN:   /out:%t.start-lib-before.exe
# RUN: FileCheck --check-prefix=MAP %s < %t.start-lib-before.map

# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.large.obj \
# RUN:   /start-lib %t.provider.obj /end-lib %t.root.obj \
# RUN:   /map:%t.start-lib-first.map /out:%t.start-lib-first.exe
# RUN: llvm-objdump -s %t.start-lib-first.exe | \
# RUN:   FileCheck --check-prefix=IMAGE %s
# RUN: FileCheck --check-prefix=MAP %s < %t.start-lib-first.map

# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.large.obj %t.root.obj \
# RUN:   /start-lib %t.provider.obj /end-lib \
# RUN:   /map:%t.start-lib-reference-first.map \
# RUN:   /out:%t.start-lib-reference-first.exe
# RUN: llvm-objdump -s %t.start-lib-reference-first.exe | \
# RUN:   FileCheck --check-prefix=IMAGE %s
# RUN: FileCheck --check-prefix=MAP %s < %t.start-lib-reference-first.map

# A directly loaded DLL follows the MinGW-only LazyDLLSymbol path.

# RUN: lld-link /lldmingw /auto-import:no /opt:ref /entry:entry \
# RUN:   /subsystem:console /nodefaultlib %t.small.obj %t.provider-dll.dll \
# RUN:   %t.large.obj %t.root-dll.obj /out:%t.lazy-dll-between.exe
# RUN: llvm-readobj --coff-imports %t.lazy-dll-between.exe | \
# RUN:   FileCheck --check-prefix=DLL %s

# Preserve a LazyDLLSymbol registered before the provisional definition.
# RUN: lld-link /lldmingw /auto-import:no /opt:ref /entry:entry \
# RUN:   /subsystem:console /nodefaultlib %t.provider-dll.dll %t.small.obj \
# RUN:   %t.large.obj %t.root-dll.obj /out:%t.lazy-dll-before.exe
# RUN: llvm-readobj --coff-imports %t.lazy-dll-before.exe | \
# RUN:   FileCheck --check-prefix=DLL %s

# RUN: lld-link /lldmingw /auto-import:no /opt:ref /entry:entry \
# RUN:   /subsystem:console /nodefaultlib %t.small.obj %t.large.obj \
# RUN:   %t.provider-dll.dll %t.root-dll.obj /out:%t.lazy-dll.exe
# RUN: llvm-readobj --coff-imports %t.lazy-dll.exe | \
# RUN:   FileCheck --check-prefix=DLL %s

# Eager extraction proves the provider itself is valid.
# RUN: lld-link /opt:ref /entry:entry /subsystem:console /nodefaultlib \
# RUN:   %t.small.obj %t.large.obj /wholearchive:%t.provider.lib %t.root.obj \
# RUN:   /map:%t.wholearchive.map /out:%t.wholearchive.exe
# RUN: llvm-objdump -s %t.wholearchive.exe | FileCheck --check-prefix=IMAGE %s
# RUN: FileCheck --check-prefix=MAP %s < %t.wholearchive.map

# IMAGE: Contents of section .rdata:
# IMAGE: 55555555

# MAP: foo_lazy{{.*}}provider.obj

# DLL: Symbol: foo_lazy

# UNREFERENCED-NOT: 55555555

# REDEFINED: 66666666
# REDEFINED-NOT: 55555555

#--- small.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .byte 0x11

        .globl foo_lazy
foo_lazy:
        .byte 0x22

#--- large.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .space 32, 0x44

#--- large-with-symbol.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .space 32, 0x44

        .globl foo_lazy
foo_lazy:
        .long 0x66666666

#--- provider.s
        .section .rdata$provider, "dr"
        .globl foo_lazy
foo_lazy:
        .long 0x55555555

#--- root.s
        .section .text$root, "xr"
        .globl entry
entry:
        movl foo_lazy(%rip), %eax
        retq

#--- entry-only.s
        .text
        .globl entry
entry:
        retq

#--- provider-dll.s
        .text
        .globl foo_lazy
foo_lazy:
        retq

#--- root-dll.s
        .text
        .globl entry
entry:
        callq foo_lazy
        retq
