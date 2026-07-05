# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t

## Ensure that we resolve the entry point to the unmangled weak alias instead of
## the mangled library definition (which would fail with an undefined symbol).

# RUN: llvm-mc -triple x86_64-windows-msvc -filetype obj -o %t/entry-weak.obj %t/entry-weak.s
# RUN: llvm-mc -triple x86_64-windows-msvc -filetype obj -o %t/entry-mangled.obj %t/entry-mangled.s
# RUN: llvm-lib -out:%t/entry-mangled.lib %t/entry-mangled.obj
# RUN: lld-link -subsystem:console -entry:entry -out:%t/entry-weak-external.exe %t/entry-weak.obj %t/entry-mangled.lib
# RUN: llvm-readobj --file-headers %t/entry-weak-external.exe | FileCheck %s

## Ensure that we don't resolve the entry point to a weak alias pointing to an
## undefined symbol (which would have caused the entry point to be 0 instead of
## an actual address). I can't think of a way of triggering this edge case
## without using /force:unresolved, which means it likely doesn't matter in
## practice, but we still match link.exe's behavior for it.

# RUN: llvm-mc -triple x86_64-windows-msvc -filetype obj -o %t/entry-weak-undefined.obj %t/entry-weak-undefined.s
# RUN: lld-link -subsystem:console -entry:entry -force:unresolved -out:%t/entry-undefined-weak-external.exe \
# RUN:   %t/entry-weak-undefined.obj %t/entry-mangled.lib
# RUN: llvm-readobj --file-headers %t/entry-undefined-weak-external.exe | FileCheck %s

# CHECK: AddressOfEntryPoint: 0x1000

#--- entry-weak.s
.globl default_entry
default_entry:
	ret

.weak entry
entry = default_entry

#--- entry-mangled.s
.globl "?entry@@YAHXZ"
"?entry@@YAHXZ":
	jmp	does_not_exist

#--- entry-weak-undefined.s
.weak entry
entry = does_not_exist
