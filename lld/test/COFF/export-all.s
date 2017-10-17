# REQEUIRES: x86

# RUN: llvm-mc -triple=i686-windows-gnu %s -filetype=obj -o %t.obj

# RUN: lld-link -lldmingw -dll -out:%t.dll -entry:DllMainCRTStartup@12 %t.obj -implib:%t.lib
# RUN: llvm-readobj -coff-exports %t.dll | FileCheck %s

# CHECK-NOT: Name: DllMainCRTStartup
# CHECK: Name: foobar

.global _foobar
.global _DllMainCRTStartup@12
.text
_DllMainCRTStartup@12:
  ret
_foobar:
  ret

# Test specifying -export-all-symbols, on an object file that contains
# dllexport directive for some of the symbols.

# RUN: yaml2obj < %p/Inputs/export.yaml > %t.obj
#
# RUN: lld-link -out:%t.dll -dll %t.obj -lldmingw -export-all-symbols -output-def:%t.def
# RUN: llvm-readobj -coff-exports %t.dll | FileCheck -check-prefix=CHECK2 %s
# RUN: cat %t.def | FileCheck -check-prefix=CHECK2-DEF %s

# Note, this will actually export _DllMainCRTStartup as well, since
# it uses the standard spelling in this object file, not the MinGW one.

# CHECK2: Name: exportfn1
# CHECK2: Name: exportfn2
# CHECK2: Name: exportfn3

# CHECK2-DEF: EXPORTS
# CHECK2-DEF: exportfn1 @3
# CHECK2-DEF: exportfn2 @4
# CHECK2-DEF: exportfn3 @5
