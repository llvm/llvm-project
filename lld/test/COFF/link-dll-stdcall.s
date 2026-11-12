# REQUIRES: x86

## Test creating a DLL and linking against the DLL without using an import
## library.

## Test on i386 with stdcall/fastcall/vectorcall decorated symbols.

## Check that we normally warn about these fixups. If -stdcall-fixup:no
## (--disable-stdcall-fixup on the MinGW linker level) is passed, we don't
## do these fixups. If we -stdcall-fixup (--enable-stdcall-fixup on the MinGW
## linker level) is passed, we don't warn about it at all.

# RUN: split-file %s %t.dir

# RUN: llvm-mc -filetype=obj -triple=i386-windows-gnu %t.dir/lib.s -o %t.lib.o
# RUN: lld-link -safeseh:no -noentry -dll -def:%t.dir/lib.def %t.lib.o -out:%t.lib.dll -implib:%t.implib.lib
# RUN: llvm-mc -filetype=obj -triple=i386-windows-gnu %t.dir/main.s -o %t.main.o
# RUN: lld-link -lldmingw %t.main.o -out:%t.main.exe %t.lib.dll -opt:noref 2>&1 | FileCheck --check-prefix=LOG %s
# RUN: llvm-readobj --coff-imports %t.main.exe | FileCheck %s
# RUN: not lld-link -lldmingw %t.main.o -out:%t.main.exe %t.lib.dll -opt:noref -stdcall-fixup:no 2>&1 | FileCheck --check-prefix=ERROR %s
# RUN: lld-link -lldmingw %t.main.o -out:%t.main.exe %t.lib.dll -opt:noref -stdcall-fixup 2>&1 | count 0

#--- lib.s
  .text
  .globl  _stdcall@8
  .globl  @fastcall@8
  .globl  vectorcall@@8
  .globl  __underscored
_stdcall@8:
  movl    8(%esp), %eax
  addl    4(%esp), %eax
  retl    $8
@fastcall@8:
  movl    8(%esp), %eax
  addl    4(%esp), %eax
  retl    $8
vectorcall@@8:
  movl    8(%esp), %eax
  addl    4(%esp), %eax
  retl    $8
__underscored:
  ret

#--- lib.def
EXPORTS
stdcall
fastcall
vectorcall
_underscored

#--- main.s
.text
.global _mainCRTStartup
_mainCRTStartup:
  pushl   $2
  pushl   $1
  calll   _stdcall@8
  movl    $1, %ecx
  movl    $2, %edx
  calll   @fastcall@8
  movl    $1, %ecx
  movl    $2, %edx
  calll   vectorcall@@8
  pushl   $2
  pushl   $1
  calll   __underscored
  addl    $8, %esp
  xorl    %eax, %eax
  popl    %ebp
  retl

# CHECK:      Import {
# CHECK-NEXT:   Name: link-dll-stdcall.s.tmp.lib.dll
# CHECK-NEXT:   ImportLookupTableRVA:
# CHECK-NEXT:   ImportAddressTableRVA
# CHECK-NEXT:   Symbol: _underscored
# CHECK-NEXT:   Symbol: fastcall
# CHECK-NEXT:   Symbol: stdcall
# CHECK-NEXT:   Symbol: vectorcall
# CHECK-NEXT: }

# LOG-DAG: Resolving vectorcall@@8 by linking to _vectorcall
# LOG-DAG: Resolving @fastcall@8 by linking to _fastcall
# LOG-DAG: Resolving _stdcall@8 by linking to _stdcall

# ERROR-DAG: undefined symbol: _stdcall@8
# ERROR-DAG: undefined symbol: @fastcall@8
# ERROR-DAG: undefined symbol: vectorcall@@8
