# REQUIRES: x86, aarch64

# RUN: split-file %s %t.dir && cd %t.dir
# RUN: llvm-mc -filetype=obj -triple=i686-windows x86-export.s -o x86-export.obj
# RUN: lld-link /dll /noentry /safeseh:no /export:func /out:x86-export.dll \
# RUN:   /implib:x86-export.lib x86-export.obj
# RUN: llvm-mc -filetype=obj -triple=i686-windows x86-main.s -o x86-main.obj
# RUN: lld-link /debug /debugtype:cv,fixup /entry:main /subsystem:console \
# RUN:   /nodefaultlib /safeseh:no /out:x86.exe /pdb:x86.pdb x86-main.obj x86-export.lib
# RUN: llvm-pdbutil dump --publics x86.pdb > x86.txt
# RUN: llvm-pdbutil dump --fixups x86.pdb >> x86.txt
# RUN: FileCheck %s --check-prefix=X86 < x86.txt

# RUN: llvm-mc -filetype=obj -triple=x86_64-windows x64-export.s -o x64-export.obj
# RUN: lld-link /dll /noentry /export:func /out:x64-export.dll \
# RUN:   /implib:x64-export.lib x64-export.obj
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows x64-main.s -o x64-main.obj
# RUN: lld-link /debug /debugtype:cv,fixup /entry:main /subsystem:console \
# RUN:   /nodefaultlib /out:x64.exe /pdb:x64.pdb x64-main.obj x64-export.lib
# RUN: llvm-pdbutil dump --publics x64.pdb > x64.txt
# RUN: llvm-pdbutil dump --fixups x64.pdb >> x64.txt
# RUN: FileCheck %s --check-prefix=X64 < x64.txt
# RUN: lld-link /debug /entry:main /subsystem:console /nodefaultlib \
# RUN:   /out:x64-nofixup.exe /pdb:x64-nofixup.pdb x64-main.obj x64-export.lib
# RUN: llvm-pdbutil dump --fixups x64-nofixup.pdb | \
# RUN:   FileCheck %s --check-prefix=NOFIXUP

# RUN: llvm-mc -filetype=obj -triple=aarch64-windows arm64-export.s -o arm64-export.obj
# RUN: lld-link /dll /noentry /export:func /out:arm64-export.dll \
# RUN:   /implib:arm64-export.lib arm64-export.obj
# RUN: llvm-mc -filetype=obj -triple=aarch64-windows arm64-main.s -o arm64-main.obj
# RUN: lld-link /debug /debugtype:cv,fixup /entry:main /subsystem:console \
# RUN:   /nodefaultlib /out:arm64.exe /pdb:arm64.pdb arm64-main.obj arm64-export.lib
# RUN: llvm-pdbutil dump --publics arm64.pdb > arm64.txt
# RUN: llvm-pdbutil dump --fixups arm64.pdb >> arm64.txt
# RUN: FileCheck %s --check-prefix=ARM64 < arm64.txt
# RUN: lld-link /debug /entry:main /subsystem:console /nodefaultlib \
# RUN:   /out:arm64-nofixup.exe /pdb:arm64-nofixup.pdb arm64-main.obj arm64-export.lib
# RUN: llvm-pdbutil dump --fixups arm64-nofixup.pdb | \
# RUN:   FileCheck %s --check-prefix=NOFIXUP

# The x86 import thunk contains an absolute address at thunk +2. Its DIR32
# record names the IAT slot, not the imported function.
# X86:      S_PUB32 {{.*}} `__imp__func`
# X86-NEXT: flags = none, addr = 0002:[[#%u,X86_IAT:]]
# X86:      S_PUB32 {{.*}} `_func`
# X86-NEXT: flags = function, addr = 0001:[[#%u,X86_THUNK:]]
# X86:      Fixup Data
# X86:      Records: 3
# X86-NEXT:     RVA    | Source       | TargetRVA  | Target       | Type | Flags  | Tag
# X86-NEXT:  00001001 | .text | [[#%.8X,0x1000 + X86_THUNK]] | .text | 20 | 0x8000 | FUNCTION
# X86-NEXT:  [[#%.8X,0x1000 + X86_THUNK + 2]] | .text | [[#%.8X,0x2000 + X86_IAT]] | .rdata | 6 | 0x0000 | DATA

# The call in x64-main.obj contributes the first record. The synthetic import
# thunk contributes a REL32 record for its disp32 field at thunk +2, targeting
# the __imp_func IAT slot and carrying no code/function classification.
# X64:      S_PUB32 {{.*}} `func`
# X64-NEXT: flags = function, addr = 0001:[[#%u,X64_THUNK:]]
# X64:      S_PUB32 {{.*}} `__imp_func`
# X64-NEXT: flags = none, addr = 0002:[[#%u,X64_IAT:]]
# X64:      Fixup Data
# X64:      Records: 3
# X64-NEXT:     RVA    | Source       | TargetRVA  | Target       | Type | Flags  | Tag
# X64-NEXT:  00001001 | .text        | [[#%.8X,0x1000 + X64_THUNK]] | .text        |    4 | 0x8000 | FUNCTION
# X64-NEXT:  [[#%.8X,0x1000 + X64_THUNK + 2]] | .text        | [[#%.8X,0x2000 + X64_IAT]] | .rdata       |    4 | 0x0000 | DATA

# The call in arm64-main.obj contributes the first record. The synthetic thunk
# contributes PAGEBASE_REL21 and PAGEOFFSET_12L records for its ADRP and LDR,
# both targeting the same __imp_func IAT data slot.
# ARM64:      S_PUB32 {{.*}} `func`
# ARM64-NEXT: flags = function, addr = 0001:[[#%u,ARM64_THUNK:]]
# ARM64:      S_PUB32 {{.*}} `__imp_func`
# ARM64-NEXT: flags = none, addr = 0002:[[#%u,ARM64_IAT:]]
# ARM64:      Fixup Data
# ARM64:      Records: 4
# ARM64-NEXT:     RVA    | Source       | TargetRVA  | Target       | Type | Flags  | Tag
# ARM64-NEXT:  00001000 | .text        | [[#%.8X,0x1000 + ARM64_THUNK]] | .text        |    3 | 0x8000 | FUNCTION
# ARM64-NEXT:  [[#%.8X,0x1000 + ARM64_THUNK]] | .text        | [[#%.8X,0x2000 + ARM64_IAT]] | .rdata       |    4 | 0x0000 | DATA
# ARM64-NEXT:  [[#%.8X,0x1000 + ARM64_THUNK + 4]] | .text        | [[#%.8X,0x2000 + ARM64_IAT]] | .rdata       |    7 | 0x0000 | DATA

# NOFIXUP: Fixup stream not present

#--- x86-export.s
        .text
        .globl _func
_func:
        retl

#--- x86-main.s
        .text
        .globl _main
_main:
        calll _func
        retl

#--- x64-export.s
        .text
        .globl func
func:
        retq

#--- x64-main.s
        .text
        .globl main
main:
        callq func
        retq

#--- arm64-export.s
        .text
        .globl func
func:
        ret

#--- arm64-main.s
        .text
        .globl main
main:
        bl func
        ret
