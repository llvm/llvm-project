# REQUIRES: x86
# RUN: llvm-mc -triple=i386-windows-gnu -filetype=obj -o %t.o %s
# RUN: not lld-link /lldmingw /out:%t.exe %t.o 2>&1 | FileCheck %s
# RUN: not lld-link /lldmingw /out:%t.exe /demangle %t.o 2>&1 | FileCheck %s
# RUN: not lld-link /lldmingw /out:%t.exe /demangle:no %t.o 2>&1 | FileCheck --check-prefix=NODEMANGLE %s

# NODEMANGLE: error: undefined symbol: __Z3fooi
# NODEMANGLE: error: undefined symbol: __Z3barPKc
# NODEMANGLE: error: undefined symbol: __imp___Z3bazv
# NODEMANGLE: error: undefined symbol: _Z3fooi
# NODEMANGLE: error: undefined symbol: __imp__cfunc

# CHECK: error: undefined symbol: foo(int)
# CHECK-NEXT: >>> referenced by {{.*}}.o:(_main)
# CHECK-NEXT: >>> referenced by {{.*}}.o:(_main)
# CHECK-EMPTY:
# CHECK-NEXT: error: undefined symbol: bar(char const*)
# CHECK-NEXT: >>> referenced by {{.*}}.o:(_main)
# CHECK-NEXT: >>> referenced by {{.*}}.o:(_f1)
# CHECK-EMPTY:
# CHECK-NEXT: error: undefined symbol: __declspec(dllimport) baz()
# CHECK-NEXT: >>> referenced by {{.*}}.o:(_f2)
# CHECK-EMPTY:
# CHECK-NEXT: error: undefined symbol: _Z3fooi
# CHECK-NEXT: >>> referenced by {{.*}}.o:(_f2)
# CHECK-EMPTY:
# CHECK-NEXT: error: undefined symbol: __declspec(dllimport) _cfunc
# CHECK-NEXT: >>> referenced by {{.*}}.o:(_f2)

        .section        .text,"xr",one_only,_main
.globl _main
_main:
	call	__Z3fooi
	call	__Z3fooi
	call	__Z3barPKc

_f1:
	call	__Z3barPKc
.Lfunc_end1:

        .section        .text,"xr",one_only,_f2
.globl _f2
_f2:
	call	*__imp___Z3bazv
	call	_Z3fooi
	call	*__imp__cfunc
