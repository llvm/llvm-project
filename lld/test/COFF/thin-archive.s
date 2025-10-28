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

# RUN: echo "/entry:main \"%t.main.obj\" /out:\"%t.exe\"" > %t.rsp

# RUN: lld-link @%t.rsp %t.lib /verbose 2>&1 | \
# RUN:     FileCheck %s --check-prefix=LOAD_NON_THIN
# RUN: lld-link @%t.rsp %t_thin.lib /verbose 2>&1 | \
# RUN:     FileCheck %s --check-prefix=LOAD_THIN_SYM
# RUN: lld-link @%t.rsp /wholearchive:%t_thin.lib /verbose 2>&1 | \
# RUN:     FileCheck %s --check-prefix=LOAD_THIN_WHOLE
# RUN: lld-link @%t.rsp /wholearchive %t_thin.lib /verbose 2>&1 | \
# RUN:     FileCheck %s --check-prefix=LOAD_THIN_WHOLE

# LOAD_NON_THIN:   Loaded {{.*}}.lib({{.*}}.obj) for int __cdecl f(void)
# LOAD_THIN_SYM:   Loaded {{.*}}.obj for int __cdecl f(void)
# LOAD_THIN_WHOLE: Loaded {{.*}}.obj for <whole-archive>

# RUN: rm %t.lib.obj
# RUN: lld-link @%t.rsp %t.lib 2>&1 | \
# RUN:     FileCheck %s --check-prefix=ERR --allow-empty
# RUN: env LLD_IN_TEST=1 not lld-link @%t.rsp %t_thin.lib 2>&1 | \
# RUN:     FileCheck %s --check-prefix=NOOBJ
# RUN: env LLD_IN_TEST=1 not lld-link @%t.rsp /wholearchive:%t_thin.lib 2>&1 | \
# RUN:     FileCheck %s --check-prefix=NOOBJWHOLE
# RUN: env LLD_IN_TEST=1 not lld-link @%t.rsp %t_thin.lib /demangle:no 2>&1 | \
# RUN:     FileCheck %s --check-prefix=NOOBJNODEMANGLE

# ERR-NOT: error: could not get the buffer for the member defining
# NOOBJ: error: could not get the buffer for the member defining symbol int __cdecl f(void): {{.*}}.lib({{.*}}.lib.obj):
# NOOBJWHOLE: error: {{.*}}.lib: could not get the buffer for a child of the archive: '{{.*}}.obj'
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
