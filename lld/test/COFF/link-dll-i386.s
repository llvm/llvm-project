# REQUIRES: x86

## Test creating a DLL and linking against the DLL without using an import
## library.

## Test on i386 with cdecl decorated symbols.

## Linking the executable with -opt:noref, to make sure that we don't
## pull in more import entries than what's needed, even if not running GC.

# RUN: split-file %s %t.dir

# RUN: llvm-mc -filetype=obj -triple=i386-windows-gnu %t.dir/lib.s -o %t.lib.o
# RUN: lld-link -safeseh:no -noentry -dll -def:%t.dir/lib.def %t.lib.o -out:%t.lib.dll -implib:%t.implib.lib
# RUN: llvm-mc -filetype=obj -triple=i386-windows-gnu %t.dir/main.s -o %t.main.o
# RUN: lld-link -lldmingw %t.main.o -out:%t.main.exe %t.lib.dll -opt:noref -verbose 2>&1 | FileCheck --check-prefix=LOG %s
# RUN: llvm-readobj --coff-imports %t.main.exe | FileCheck %s

#--- lib.s
.text
.global _func1
_func1:
  ret
.global _func2
_func2:
  ret
.global _func3
_func3:
  ret
.data
.global _variable
_variable:
  .int 42

#--- lib.def
EXPORTS
func1
func2
func3
variable

#--- main.s
.text
.global _mainCRTStartup
_mainCRTStartup:
  call _func2
  movl .refptr._variable, %eax
  movl (%eax), %eax
  ret

.section .rdata$.refptr._variable,"dr",discard,.refptr._variable
.globl .refptr._variable
.refptr._variable:
  .long _variable

# CHECK:      Import {
# CHECK-NEXT:   Name: link-dll-i386.s.tmp.lib.dll
# CHECK-NEXT:   ImportLookupTableRVA:
# CHECK-NEXT:   ImportAddressTableRVA
# CHECK-NEXT:   Symbol: func2
# CHECK-NEXT:   Symbol: variable
# CHECK-NEXT: }

# LOG: Automatically importing _variable from link-dll-i386.s.tmp.lib.dll
