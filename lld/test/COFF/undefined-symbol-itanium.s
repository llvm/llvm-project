# REQUIRES: x86
# RUN: llvm-mc -triple=x86_64-windows-gnu -filetype=obj -o %t.o %s
# RUN: not lld-link /lldmingw /out:%t.exe %t.o 2>&1 | FileCheck %s
# RUN: not lld-link /lldmingw /out:%t.exe /demangle %t.o 2>&1 | FileCheck %s
# RUN: not lld-link /lldmingw /out:%t.exe /demangle:no %t.o 2>&1 | FileCheck --check-prefix=NODEMANGLE %s

# NODEMANGLE: error: undefined symbol: _Z3fooi
# NODEMANGLE: error: undefined symbol: _Z3barPKc
# NODEMANGLE: error: undefined symbol: __imp__Z3bazv

# CHECK: error: undefined symbol: foo(int)
# CHECK-NEXT: >>> referenced by {{.*}}.o:(main)
# CHECK-NEXT: >>> referenced by {{.*}}.o:(main)
# CHECK-EMPTY:
# CHECK-NEXT: error: undefined symbol: bar(char const*)
# CHECK-NEXT: >>> referenced by {{.*}}.o:(main)
# CHECK-NEXT: >>> referenced by {{.*}}.o:(f1)
# CHECK-EMPTY:
# CHECK-NEXT: error: undefined symbol: __declspec(dllimport) baz()
# CHECK-NEXT: >>> referenced by {{.*}}.o:(f2)

        .section        .text,"xr",one_only,main
.globl main
main:
	call	_Z3fooi
	call	_Z3fooi
	call	_Z3barPKc

f1:
	call	_Z3barPKc
.Lfunc_end1:

        .section        .text,"xr",one_only,f2
.globl f2
f2:
	callq	*__imp__Z3bazv(%rip)
