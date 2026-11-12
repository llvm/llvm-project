# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-mingw32 -o %t.o %s
# RUN: lld-link -lldmingw -dll -out:%t.dll %t.o -entry:__ImageBase 2>&1 | FileCheck %s --allow-empty --check-prefix=NOWARNING
# RUN: llvm-readobj --coff-exports %t.dll | FileCheck %s
# RUN: lld-link -lldmingw -dll -out:%t.dll %t.o -entry:__ImageBase -export:otherfunc 2>&1 | FileCheck %s --check-prefix=WARNING
# RUN: llvm-readobj --coff-exports %t.dll | FileCheck %s

# Check that the export table contains the manually crafted content
# instead of the linker generated exports.

# CHECK:      Export {
# CHECK-NEXT:   Ordinal: 1
# CHECK-NEXT:   Name: myfunc
# CHECK-NEXT:   RVA:
# CHECK-NEXT: }
# CHECK-EMPTY:

# NOWARNING-NOT: warning

# WARNING: warning: literal .edata sections override exports

    .text
    .globl myfunc
myfunc:
    ret
    .globl otherfunc
otherfunc:
    ret

// The object contains a manually crafted .edata section, which exports
// myfunc, not otherfunc.
    .section .edata, "drw"
    .align 4
exports:
    .long 0           // ExportFlags
    .long 0           // TimeDateStamp
    .long 0           // MajorVersion + MinorVersion
    .rva name         // NameRVA
    .long 1           // OrdinalBase
    .long 1           // AddressTableEntries
    .long 1           // NumberOfNamePointers
    .rva functions    // ExportAddressTableRVA
    .rva names        // NamePointerRVA
    .rva nameordinals // OrdinalTableRVA

names:
    .rva funcname_myfunc

nameordinals:
    .short 0

functions:
    .rva myfunc
    .long 0

funcname_myfunc:
    .asciz "myfunc"

name:
    .asciz "mydll.dll"
