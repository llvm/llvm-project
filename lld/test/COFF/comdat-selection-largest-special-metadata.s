# REQUIRES: x86

# .drectve can be emitted as an ANY COMDAT by Clang's ASan instrumentation.
# NODUPLICATES cannot lose silently. For ANY, only the prevailing directives
# are exposed and candidates must be identical, making input order irrelevant.
# The LLVM addrsig and call-graph formats are object-level metadata and cannot
# carry COMDAT provenance, so explicitly reject COMDAT combinations.

# RUN: split-file %s %t.dir
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/directive-a.s -o %t.directive-a.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/directive-b.s -o %t.directive-b.raw.obj
# RUN: llvm-objcopy --remove-section=.text --remove-section=.data \
# RUN:   --remove-section=.bss %t.directive-b.raw.obj %t.directive-b.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/directive-conflict.s -o %t.directive-conflict.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/directive-largest.s -o %t.directive-largest.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/addrsig.s -o %t.addrsig.obj
# RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj \
# RUN:   %t.dir/cgprofile.s -o %t.cgprofile.obj

# RUN: lld-link /dll /noentry /nodefaultlib /timestamp:0 %t.directive-a.obj \
# RUN:   /out:%t.same-name.dll
# RUN: cp %t.same-name.dll %t.winner.dll
# RUN: lld-link /verbose /dll /noentry /nodefaultlib /timestamp:0 \
# RUN:   %t.directive-a.obj \
# RUN:   %t.directive-b.obj /out:%t.same-name.dll 2>&1 | \
# RUN:   FileCheck %s --check-prefix=DIRECTIVE-ONCE
# RUN: cmp %t.winner.dll %t.same-name.dll
# RUN: lld-link /verbose /dll /noentry /nodefaultlib /timestamp:0 \
# RUN:   %t.directive-b.obj \
# RUN:   %t.directive-a.obj /out:%t.same-name.dll 2>&1 | \
# RUN:   FileCheck %s --check-prefix=DIRECTIVE-ONCE
# RUN: cmp %t.winner.dll %t.same-name.dll
# RUN: llvm-readobj --coff-exports %t.winner.dll | \
# RUN:   FileCheck %s --check-prefix=EXPORT
# RUN: not lld-link /dll /noentry /nodefaultlib %t.directive-a.obj \
# RUN:   %t.directive-conflict.obj /out:%t.conflict.dll 2>&1 | \
# RUN:   FileCheck %s --check-prefix=DIRECTIVE-CONFLICT
# RUN: not lld-link /dll /noentry /nodefaultlib %t.directive-largest.obj \
# RUN:   /out:%t.largest.dll 2>&1 | \
# RUN:   FileCheck %s --check-prefix=DIRECTIVE-LARGEST
# RUN: not lld-link /dll /noentry /nodefaultlib %t.addrsig.obj \
# RUN:   /out:%t.addrsig.dll 2>&1 | FileCheck %s --check-prefix=ADDRSIG
# RUN: not lld-link /dll /noentry /nodefaultlib \
# RUN:   /discard-section:.llvm_addrsig %t.addrsig.obj \
# RUN:   /out:%t.addrsig-discard.dll 2>&1 | FileCheck %s --check-prefix=ADDRSIG
# RUN: not lld-link /dll /noentry /nodefaultlib %t.cgprofile.obj \
# RUN:   /out:%t.cgprofile.dll 2>&1 | FileCheck %s --check-prefix=CGPROFILE
# RUN: not lld-link /dll /noentry /nodefaultlib \
# RUN:   /discard-section:.llvm.call-graph-profile %t.cgprofile.obj \
# RUN:   /out:%t.cgprofile-discard.dll 2>&1 | \
# RUN:   FileCheck %s --check-prefix=CGPROFILE

# EXPORT: Name: exported
# DIRECTIVE-ONCE-COUNT-1: Directives:
# DIRECTIVE-CONFLICT: error: {{.*}}directive-conflict.obj: COMDAT .drectve
# DIRECTIVE-CONFLICT-SAME: candidates must use selection type ANY and have
# DIRECTIVE-CONFLICT-SAME: identical contents
# DIRECTIVE-LARGEST: error: {{.*}}directive-largest.obj: COMDAT .drectve must
# DIRECTIVE-LARGEST-SAME: use selection type ANY or NODUPLICATES
# ADDRSIG: error: {{.*}}addrsig.obj: .llvm_addrsig cannot be a COMDAT section
# CGPROFILE: error: {{.*}}cgprofile.obj: .llvm.call-graph-profile cannot be a
# CGPROFILE-SAME: COMDAT section

#--- directive-a.s
        .text
        .globl exported
exported:
        retq
        .section .drectve,"dr",discard,directive_group
        .globl directive_group
directive_group:
        .ascii " /export:exported"

#--- directive-b.s
        .section .drectve,"dr",discard,directive_group
        .globl directive_group
directive_group:
        .ascii " /export:exported"

#--- directive-conflict.s
        .section .drectve,"dr",discard,directive_group
        .globl directive_group
directive_group:
        .ascii " /export:different"

#--- directive-largest.s
        .section .drectve,"dr",largest,directive_group
        .globl directive_group
directive_group:
        .ascii " /export:exported"

#--- addrsig.s
        .section .llvm_addrsig,"dr",one_only,addrsig_group
        .globl addrsig_group
addrsig_group:
        .byte 0

#--- cgprofile.s
        .section ".llvm.call-graph-profile","dr",one_only,cgprofile_group
        .globl cgprofile_group
cgprofile_group:
        .quad 0
