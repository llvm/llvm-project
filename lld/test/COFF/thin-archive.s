# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc -o %t.main.obj %s

# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc -o %t.lib.obj \
# RUN:     %S/Inputs/mangled-symbol.s
# RUN: lld-link /lib /out:%t.lib %t.lib.obj
# RUN: lld-link /lib /llvmlibindex:no /out:%t_noindex.lib %t.lib.obj
# RUN: lld-link /lib /llvmlibthin /llvmlibindex /out:%t_thin.lib %t.lib.obj
# RUN: lld-link /lib /llvmlibthin /llvmlibindex:no \
# RUN:     /out:%t_thin_noindex.lib %t.lib.obj

# RUN: llvm-nm --print-armap %t.lib \
# RUN:   | FileCheck %s --check-prefix=SYMTAB
# RUN: llvm-nm --print-armap %t_noindex.lib \
# RUN:   | FileCheck %s --check-prefix=NO-SYMTAB
# RUN: llvm-nm --print-armap %t_thin.lib \
# RUN:   | FileCheck %s --check-prefix=SYMTAB
# RUN: llvm-nm --print-armap %t_thin_noindex.lib \
# RUN:   | FileCheck %s --check-prefix=NO-SYMTAB

# SYMTAB:        ?f@@YAHXZ in
# NO-SYMTAB-NOT: ?f@@YAHXZ in

# RUN: lld-link /entry:main %t.main.obj %t.lib /out:%t.exe 2>&1 | \
# RUN:     FileCheck --allow-empty %s
# RUN: lld-link /entry:main %t.main.obj %t_thin.lib /out:%t.exe 2>&1 | \
# RUN:     FileCheck --allow-empty %s
# RUN: lld-link /entry:main %t.main.obj /wholearchive:%t_thin.lib /out:%t.exe 2>&1 | \
# RUN:     FileCheck --allow-empty %s

# RUN: rm %t.lib.obj
# RUN: lld-link /entry:main %t.main.obj %t.lib /out:%t.exe 2>&1 | \
# RUN:     FileCheck --allow-empty %s
# RUN: env LLD_IN_TEST=1 not lld-link /entry:main %t.main.obj %t_thin.lib \
# RUN:     /out:%t.exe 2>&1 | FileCheck --check-prefix=NOOBJ %s
# RUN: env LLD_IN_TEST=1 not lld-link /entry:main %t.main.obj %t_thin.lib /out:%t.exe \
# RUN:     /demangle:no 2>&1 | FileCheck --check-prefix=NOOBJNODEMANGLE %s

# CHECK-NOT: error: could not get the buffer for the member defining
# NOOBJ: error: could not get the buffer for the member defining symbol int __cdecl f(void): {{.*}}.lib({{.*}}.lib.obj):
# NOOBJNODEMANGLE: error: could not get the buffer for the member defining symbol ?f@@YAHXZ: {{.*}}.lib({{.*}}.lib.obj):

	.text

	.def main
		.scl 2
		.type 32
	.endef
	.global main
main:
	call "?f@@YAHXZ"
	retq $0
