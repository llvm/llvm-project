# REQUIRES: x86

# RUN: llvm-mc -triple=x86_64-windows-gnu %s -filetype=obj -o %t.obj

# RUN: lld-link %t.obj -out:%t.exe -debug:symtab -subsystem:console
# RUN: llvm-readobj --coff-exports %t.exe | FileCheck %s

# CHECK: Name: exportedFunc

## This assembly snippet has been reduced from what Clang generates from
## this C snippet, with -fsanitize=address. Normally, the .drectve
## section would be a regular section - but when compiled with
## -fsanitize=address, it becomes a comdat section.
##
# void exportedFunc(void) {}
# void mainCRTStartup(void) {}
# static __attribute__((section(".drectve"), used)) const char export_chkstk[] =
#     "-export:exportedFunc";

	.text
	.globl	exportedFunc
exportedFunc:
	retq

	.globl	mainCRTStartup
mainCRTStartup:
	retq

	.section	.drectve,"dr",one_only,export_chkstk
export_chkstk:
	.asciz	"-export:exportedFunc"
