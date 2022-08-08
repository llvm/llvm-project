// Check that we ignore XFG subsections in CodeView information.
// This test should be removed if we ever end up adding XFG support to lld.

# REQUIRES: x86
# RUN: llvm-mc -triple=i386-pc-win32 -filetype=obj -o %t.obj %s
# RUN: lld-link -verbose -safeseh:no -subsystem:console -debug -nodefaultlib -entry:foo -out:%t.exe -pdb:%t.pdb %t.obj 2>&1 | FileCheck %s

// XFGHashType should not generate a warning
# CHECK-NOT: ignoring unknown debug$S subsection kind 0xFF
// XFGHashVirtual should not generate a warning
# CHECK-NOT: ignoring unknown debug$S subsection kind 0x100

.text
_foo:
ret

.global _foo

.section .debug$S,"dr"
	.p2align	2
	.long	4                       # Debug section magic
	.long	0xF1 # Symbol subsection
	.long	.Ltmp6-.Ltmp5           # Subsection size
.Ltmp5:
	.short	.Ltmp8-.Ltmp7           # Record length
.Ltmp7:
	.short	4412                    # Record kind: S_COMPILE3
	.long	0                       # Flags and language
	.short	208                     # CPUType
	.short	9                       # Frontend version
	.short	0
	.short	0
	.short	0
	.short	9000                    # Backend version
	.short	0
	.short	0
	.short	0
	.asciz	"clang version SENTINEL" # Null-terminated compiler version string
	.p2align	2
.Ltmp8:
.Ltmp6:
	.long	0xFF # XFGHashType subsection
	.long	4           # Subsection size
	.long  0
	.long	0x100 # XFGHashVirtual subsection
	.long	4           # Subsection size
	.long  0
