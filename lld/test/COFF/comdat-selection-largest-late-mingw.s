# REQUIRES: x86

# A stdcall fixup can extract an archive member after the first deferred-COMDAT
# fixed point. Providers hidden by the provisional winner must remain available
# when that late member replaces the winner.

# RUN: split-file %s %t.dir
# RUN: llvm-mc -triple i386-windows-gnu -filetype=obj %t.dir/weak.s \
# RUN:   -o %t.weak.obj
# RUN: llvm-mc -triple i386-windows-gnu -filetype=obj %t.dir/small.s \
# RUN:   -o %t.small.obj
# RUN: llvm-mc -triple i386-windows-gnu -filetype=obj %t.dir/small-ref.s \
# RUN:   -o %t.small-ref.obj
# RUN: llvm-mc -triple i386-windows-gnu -filetype=obj %t.dir/late.s \
# RUN:   -o %t.late.obj
# RUN: llvm-mc -triple i386-windows-gnu -filetype=obj %t.dir/foo.s \
# RUN:   -o %t.foo.obj
# RUN: llvm-mc -triple i386-windows-gnu -filetype=obj %t.dir/root-auto.s \
# RUN:   -o %t.root-auto.obj
# RUN: llvm-mc -triple i386-windows-gnu -filetype=obj %t.dir/small-auto.s \
# RUN:   -o %t.small-auto.obj
# RUN: llvm-mc -triple i386-windows-gnu -filetype=obj %t.dir/permanent.s \
# RUN:   -o %t.permanent.obj
# RUN: llvm-mc -triple i386-windows-gnu -filetype=obj %t.dir/small-dup.s \
# RUN:   -o %t.small-dup.obj
# RUN: llvm-mc -triple i386-windows-gnu -filetype=obj %t.dir/stdcall-loser.s \
# RUN:   -o %t.stdcall-loser.obj
# RUN: llvm-mc -triple i386-windows-gnu -filetype=obj %t.dir/stdcall-winner.s \
# RUN:   -o %t.stdcall-winner.obj
# RUN: llvm-mc -triple i386-windows-gnu -filetype=obj %t.dir/stdcall-foo.s \
# RUN:   -o %t.stdcall-foo.obj
# RUN: llvm-mc -triple i386-windows-gnu -filetype=obj %t.dir/leak-small.s \
# RUN:   -o %t.leak-small.obj
# RUN: llvm-mc -triple i386-windows-gnu -filetype=obj %t.dir/leak-late.s \
# RUN:   -o %t.leak-late.obj
# RUN: llvm-mc -triple i386-windows-gnu -filetype=obj \
# RUN:   %t.dir/auto-leak-small.s -o %t.auto-leak-small.obj
# RUN: llvm-mc -triple i386-windows-gnu -filetype=obj \
# RUN:   %t.dir/auto-leak-late.s -o %t.auto-leak-late.obj
# RUN: llvm-lib -machine:x86 -out:%t.late.lib %t.late.obj
# RUN: llvm-lib -machine:x86 -out:%t.foo.lib %t.foo.obj
# RUN: llvm-lib -machine:x86 -out:%t.stdcall-foo.lib %t.stdcall-foo.obj
# RUN: llvm-lib -machine:x86 -out:%t.leak-late.lib %t.leak-late.obj
# RUN: llvm-lib -machine:x86 -out:%t.auto-leak-late.lib \
# RUN:   %t.auto-leak-late.obj
# RUN: llvm-dlltool -m i386 -d %t.dir/provider.def -l %t.provider.lib
# RUN: lld-link /lldmingw /stdcall-fixup /safeseh:no /nodefaultlib /opt:noref \
# RUN:   /entry:entry %t.weak.obj %t.small.obj %t.late.lib /out:%t.exe
# RUN: llvm-objdump -s %t.exe | FileCheck %s
# RUN: lld-link /lldmingw /stdcall-fixup /safeseh:no /nodefaultlib /opt:noref \
# RUN:   /entry:entry %t.weak.obj %t.small-ref.obj %t.foo.lib %t.late.lib \
# RUN:   /out:%t.no-extract.exe
# RUN: llvm-objdump -s %t.no-extract.exe | \
# RUN:   FileCheck %s --check-prefix=NOT-EXTRACTED
# RUN: lld-link /lldmingw /stdcall-fixup /safeseh:no /nodefaultlib /opt:noref \
# RUN:   /entry:entry %t.root-auto.obj %t.small-auto.obj %t.late.lib \
# RUN:   %t.provider.lib /out:%t.no-autoimport.exe
# RUN: llvm-readobj --coff-imports %t.no-autoimport.exe | \
# RUN:   FileCheck %s --check-prefix=NO-AUTOIMPORT
# RUN: llvm-readobj --coff-exports %t.no-autoimport.exe | \
# RUN:   FileCheck %s --check-prefix=NO-AUTOEXPORT
# RUN: lld-link /lldmingw /stdcall-fixup /safeseh:no /nodefaultlib /opt:noref \
# RUN:   /entry:entry %t.root-auto.obj %t.permanent.obj %t.small-dup.obj \
# RUN:   %t.late.lib /out:%t.no-duplicate.exe 2>&1 | \
# RUN:   FileCheck %s --allow-empty --check-prefix=NO-DUPLICATE
# RUN: lld-link /lldmingw /stdcall-fixup /safeseh:no /nodefaultlib /opt:noref \
# RUN:   /entry:entry %t.root-auto.obj %t.stdcall-loser.obj \
# RUN:   %t.stdcall-winner.obj %t.stdcall-foo.lib \
# RUN:   /out:%t.no-loser-stdcall.exe
# RUN: llvm-objdump -s %t.no-loser-stdcall.exe | \
# RUN:   FileCheck %s --check-prefix=NO-LOSER-STDCALL
# RUN: lld-link /lldmingw /stdcall-fixup /safeseh:no /nodefaultlib /opt:noref \
# RUN:   /entry:entry %t.root-auto.obj %t.leak-small.obj %t.leak-late.lib \
# RUN:   /out:%t.no-provisional-reference.exe
# RUN: lld-link /lldmingw /auto-import /safeseh:no /nodefaultlib /opt:noref \
# RUN:   /entry:entry %t.root-auto.obj %t.auto-leak-small.obj \
# RUN:   %t.auto-leak-late.lib /out:%t.no-provisional-autoimport.exe

# CHECK: Contents of section .text:
# CHECK: 77777777
# CHECK: 44444444
# CHECK-NOT: 11111111
# NOT-EXTRACTED-NOT: aaaaaaaa
# NO-AUTOIMPORT-NOT: Import {
# NO-AUTOEXPORT-NOT: Name: only_auto_small
# NO-DUPLICATE-NOT: duplicate symbol: duplicate
# NO-LOSER-STDCALL-NOT: abababab

#--- weak.s
        .text
        .globl fallback
fallback:
        .long 0x77777777

        .weak victim
        victim = fallback

        .globl _entry
_entry:
        movl victim, %eax
        movl leader, %eax
        calll "_trigger@0"
        retl

#--- small.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .space 8, 0x11
        .globl victim
victim:
        .long 0x11111111

#--- late.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .space 32, 0x44

        .text
        .globl _trigger
_trigger:
        retl

# Automatic import must likewise be able to load a larger candidate without
# publishing unrelated references from the provisional group.
#--- auto-leak-small.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .long _missing_auto
        .long _variable_auto

#--- auto-leak-late.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .space 32, 0x77

        .data
        .globl __imp__variable_auto
__imp__variable_auto:
        .long 0

#--- small-ref.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .long foo
        .long 0x11111111
        .globl victim
victim:
        .long 0x11111111

#--- foo.s
        .text
        .globl foo
foo:
        .long 0xaaaaaaaa

#--- root-auto.s
        .text
        .globl _entry
_entry:
        movl leader, %eax
        retl

#--- small-auto.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        movl _variable, %eax
        calll "_trigger@0"
        retl
        .globl only_auto_small
only_auto_small:
        retl

#--- permanent.s
        .data
        .globl duplicate
duplicate:
        .long 0x22222222

#--- small-dup.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        calll "_trigger@0"
        retl
        .globl duplicate
duplicate:
        .long 0x11111111

#--- provider.def
LIBRARY provider.dll
EXPORTS
variable DATA

#--- stdcall-loser.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        calll "_foo@0"
        retl

#--- stdcall-winner.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .space 32, 0x55

#--- stdcall-foo.s
        .text
        .globl _foo
_foo:
        .long 0xabababab
        retl

# A stdcall fixup for _trigger@0 loads leak-late.obj and supersedes this
# group. The other fixup must not make _missing@0 a permanent reference; its
# undecorated secondary definition disappears with this provisional winner.
#--- leak-small.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        calll "_missing@0"
        calll "_trigger@0"
        retl
        .globl _missing
_missing:
        retl

#--- leak-late.s
        .section .text$largest, "xr", largest, leader
        .globl leader
leader:
        .space 32, 0x66

        .text
        .globl _trigger
_trigger:
        retl
