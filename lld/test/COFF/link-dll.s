# REQUIRES: x86

## Test creating a DLL and linking against the DLL without using an import
## library.

## Explicitly creating an import library but naming it differently than the
## DLL, to avoid any risk of implicitly referencing it instead of the DLL
## itself.

## Linking the executable with -opt:noref, to make sure that we don't
## pull in more import entries than what's needed, even if not running GC.

# RUN: split-file %s %t.dir

# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-gnu %t.dir/lib.s -o %t.lib.o
# RUN: lld-link -noentry -dll -def:%t.dir/lib.def %t.lib.o -out:%t.lib.dll -implib:%t.implib.lib
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-gnu %t.dir/main.s -o %t.main.o
# RUN: lld-link -lldmingw %t.main.o -out:%t.main.exe %t.lib.dll -opt:noref -verbose 2>&1 | FileCheck --check-prefix=LOG %s
# RUN: llvm-readobj --coff-imports %t.main.exe | FileCheck %s

#--- lib.s
.text
.global func1
func1:
  ret
.global func2
func2:
  ret
.global func3
func3:
  ret
.data
.global variable
variable:
  .int 42

#--- lib.def
EXPORTS
func1
func2
func3
variable

#--- main.s
.text
.global mainCRTStartup
mainCRTStartup:
  call func2
  movq .refptr.variable(%rip), %rax
  movl (%rax), %eax
  ret

.section .rdata$.refptr.variable,"dr",discard,.refptr.variable
.globl .refptr.variable
.refptr.variable:
  .quad variable

# CHECK:      Import {
# CHECK-NEXT:   Name: link-dll.s.tmp.lib.dll
# CHECK-NEXT:   ImportLookupTableRVA:
# CHECK-NEXT:   ImportAddressTableRVA
# CHECK-NEXT:   Symbol: func2
# CHECK-NEXT:   Symbol: variable
# CHECK-NEXT: }

# LOG: Automatically importing variable from link-dll.s.tmp.lib.dll
