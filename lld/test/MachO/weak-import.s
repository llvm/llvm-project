# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/weak-ref-only.s -o %t/weak-ref-only.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/weak-ref-sub-library.s -o %t/weak-ref-sub-library.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/mixed-ref.s -o %t/mixed-ref.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/bar.s -o %t/bar.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/weak-reexport.s -o %t/weak-reexport.o

# RUN: %lld -lSystem -dylib %t/bar.o -o %t/libbar.dylib
# RUN: %lld -lSystem -dylib %t/foo.o %t/libbar.dylib -sub_library libbar -o %t/libfoo.dylib

# RUN: %lld -weak-lSystem %t/test.o -weak_framework CoreFoundation -weak_library %t/libfoo.dylib -o %t/basic
# RUN: llvm-objdump --macho --all-headers %t/basic | FileCheck %s -DDIR=%t --check-prefixes=WEAK-SYS,WEAK-FOO
# RUN: %lld -weak-lSystem %t/test.o \
# RUN:   -framework CoreFoundation -weak_framework CoreFoundation -framework CoreFoundation \
# RUN:   %t/libfoo.dylib -weak_library %t/libfoo.dylib %t/libfoo.dylib -o %t/basic-weak-strong
# RUN: llvm-objdump --macho --all-headers %t/basic-weak-strong | FileCheck %s -DDIR=%t --check-prefixes=WEAK-SYS,WEAK-FOO
# RUN: %lld -lSystem -dylib %t/libfoo.dylib %t/weak-ref-only.o -o %t/weak-ref-only
# RUN: llvm-objdump --macho --all-headers %t/weak-ref-only | FileCheck %s -DDIR=%t --check-prefixes=SYS,WEAK-FOO
# RUN: %lld -lSystem -dylib %t/libfoo.dylib %t/weak-ref-sub-library.o -o %t/weak-ref-sub-library
# RUN: llvm-objdump --macho --all-headers %t/weak-ref-sub-library | FileCheck %s -DDIR=%t --check-prefixes=SYS,WEAK-FOO
# RUN: %lld -lSystem -dylib %t/libfoo.dylib %t/mixed-ref.o -o %t/mixed-ref
# RUN: llvm-objdump --macho --all-headers %t/mixed-ref | FileCheck %s -DDIR=%t --check-prefixes=SYS,FOO

# RUN: %lld -framework CoreFoundation %t/test.o -weak_framework CoreFoundation -o %t/strong-weak-import.out
# RUN: llvm-objdump --macho --bind %t/strong-weak-import.out | FileCheck %s --check-prefix=WEAK-IMP
# RUN: llvm-objdump --macho --bind %t/basic-weak-strong | FileCheck %s --check-prefix=WEAK-IMP

## This references the gxx_personality_v0 symbol, which is defined in libc++abi, which is then
## reexported by libc++.
## Check that reexported symbols from weak libraries are also weakly-referenced.
# RUN: %lld -weak-lc++ %t/weak-reexport.o -o %t/weak-reexport.out
# RUN: llvm-objdump --macho --bind %t/weak-reexport.out | FileCheck %s --check-prefix=WEAK-REEXPORT

# WEAK-SYS:          cmd LC_LOAD_WEAK_DYLIB
# WEAK-SYS-NEXT: cmdsize
# WEAK-SYS-NEXT:    name /usr/lib/libSystem.dylib

# WEAK-SYS:          cmd LC_LOAD_WEAK_DYLIB
# WEAK-SYS-NEXT: cmdsize
# WEAK-SYS-NEXT:    name /System/Library/Frameworks/CoreFoundation.framework/CoreFoundation

# SYS:               cmd LC_LOAD_DYLIB
# SYS-NEXT:      cmdsize
# SYS-NEXT:         name /usr/lib/libSystem.dylib

# WEAK-FOO:          cmd LC_LOAD_WEAK_DYLIB
# WEAK-FOO-NEXT: cmdsize
# WEAK-FOO-NEXT:    name [[DIR]]/libfoo.dylib

# FOO:               cmd LC_LOAD_DYLIB
# FOO-NEXT:      cmdsize
# FOO-NEXT:         name [[DIR]]/libfoo.dylib

# WEAK-IMP: {{.+}} pointer         0 CoreFoundation   __CFBigNumGetInt128 (weak_import)
# WEAK-REEXPORT: {{.+}} pointer    0 libc++abi        ___gxx_personality_v0 (weak_import)
#--- foo.s
.globl _foo
_foo:

#--- bar.s
.globl _bar
_bar:

#--- weak-ref-only.s
.weak_reference _foo
.data
.quad _foo

#--- weak-ref-sub-library.s
.weak_reference _bar
.data
.quad _bar

#--- mixed-ref.s
.weak_definition _foo
.data
.quad _foo
.quad _bar

#--- test.s
.globl _main
_main:
  movq __CFBigNumGetInt128@GOTPCREL(%rip), %rax        
  ret

#--- weak-reexport.s
.globl _main
_main:

.data
.quad ___gxx_personality_v0
