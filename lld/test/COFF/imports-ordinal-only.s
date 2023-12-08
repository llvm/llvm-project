# REQUIRES: x86
#
# RUN: lld-link -machine:x86 -def:%p/Inputs/ordinal-only-implib.def -implib:%t-implib.a
# RUN: llvm-mc -triple=i386-pc-win32 %s -filetype=obj -o %t.obj
# RUN: lld-link -out:%t.exe -entry:main -subsystem:console -safeseh:no -debug %t.obj %t-implib.a
# RUN: llvm-objdump --private-headers %t.exe | FileCheck --match-full-lines %s

.text
.global _main
_main:
call _ByOrdinalFunction
ret

# CHECK: The Import Tables:
# CHECK:     DLL Name: test.dll
# CHECK-NEXT:     Hint/Ord  Name
# CHECK-NEXT:            1
# CHECK-EMPTY:
