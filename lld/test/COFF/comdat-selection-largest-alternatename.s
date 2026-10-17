# REQUIRES: x86

# A secondary definition that disappears with a losing LARGEST group is an
# Undefined type-state placeholder, not a reference. It must not activate an
# /alternatename fallback or extract its archive member. A genuine relocation
# to the same name must still activate the fallback. Both properties are input
# order independent and apply to command-line and .drectve aliases.

# RUN: split-file %s %t.dir
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/small.s -o %t.small.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/large.s -o %t.large.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/root.s -o %t.root.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/root-ref.s -o %t.root-ref.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/provider.s -o %t.provider.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/directive.s -o %t.directive.obj
# RUN: llvm-ar --format=coff rcsD %t.provider.lib %t.provider.obj

# RUN: lld-link /entry:entry /subsystem:console /nodefaultlib /opt:ref \
# RUN:   /alternatename:foo=bar %t.small.obj %t.large.obj %t.root.obj \
# RUN:   %t.provider.lib /out:%t.no-ref-small-first.exe
# RUN: llvm-objdump -s %t.no-ref-small-first.exe | \
# RUN:   FileCheck %s --check-prefix=NO-BAR
# RUN: lld-link /entry:entry /subsystem:console /nodefaultlib /opt:ref \
# RUN:   /alternatename:foo=bar %t.large.obj %t.small.obj %t.root.obj \
# RUN:   %t.provider.lib /out:%t.no-ref-large-first.exe
# RUN: llvm-objdump -s %t.no-ref-large-first.exe | \
# RUN:   FileCheck %s --check-prefix=NO-BAR

# RUN: lld-link /entry:entry /subsystem:console /nodefaultlib /opt:ref \
# RUN:   /alternatename:foo=bar %t.small.obj %t.large.obj %t.root-ref.obj \
# RUN:   %t.provider.lib /out:%t.ref-small-first.exe
# RUN: llvm-objdump -s %t.ref-small-first.exe | FileCheck %s --check-prefix=BAR
# RUN: lld-link /entry:entry /subsystem:console /nodefaultlib /opt:ref \
# RUN:   /alternatename:foo=bar %t.large.obj %t.small.obj %t.root-ref.obj \
# RUN:   %t.provider.lib /out:%t.ref-large-first.exe
# RUN: llvm-objdump -s %t.ref-large-first.exe | FileCheck %s --check-prefix=BAR

# RUN: lld-link /entry:entry /subsystem:console /nodefaultlib /opt:ref \
# RUN:   %t.directive.obj %t.small.obj %t.large.obj %t.root.obj \
# RUN:   %t.provider.lib /out:%t.directive-no-ref.exe
# RUN: llvm-objdump -s %t.directive-no-ref.exe | \
# RUN:   FileCheck %s --check-prefix=NO-BAR
# RUN: lld-link /entry:entry /subsystem:console /nodefaultlib /opt:ref \
# RUN:   %t.directive.obj %t.small.obj %t.large.obj %t.root-ref.obj \
# RUN:   %t.provider.lib /out:%t.directive-ref.exe
# RUN: llvm-objdump -s %t.directive-ref.exe | FileCheck %s --check-prefix=BAR

# NO-BAR: Contents of section .text:
# NO-BAR-NOT: 42424242
# BAR: Contents of section .text:
# BAR: 42424242

#--- small.s
        .section .text$leader, "xr", largest, leader
        .globl leader
leader:
        retq
        .globl foo
foo:
        retq

#--- large.s
        .section .text$leader, "xr", largest, leader
        .globl leader
leader:
        .space 16, 0x90
        retq

#--- root.s
        .text
        .globl entry
entry:
        callq leader
        retq

#--- root-ref.s
        .text
        .globl entry
entry:
        callq leader
        callq foo
        retq

#--- provider.s
        .text
        .globl bar
bar:
        .long 0x42424242
        retq

#--- directive.s
        .section .drectve, "yn"
        .ascii " /alternatename:foo=bar"
