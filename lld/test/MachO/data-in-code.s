# REQUIRES: x86

# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/bar.s -o %t/bar.o
# RUN: %lld -lSystem %t/foo.o %t/bar.o -o %t/main.exe
# RUN: llvm-otool -l %t/main.exe > %t/objdump
# RUN: llvm-otool -Gv %t/main.exe >> %t/objdump
# RUN: FileCheck %s < %t/objdump

# CHECK-LABEL:  sectname __text
# CHECK-NEXT:   segname __TEXT
# CHECK-NEXT:   addr
# CHECK-NEXT:   size
# CHECK-NEXT:   offset [[#%,TEXT:]]

# CHECK-LABEL:  cmd LC_DATA_IN_CODE
# CHECK-NEXT:   cmdsize 16
# CHECK-NEXT:   dataoff
# CHECK-NEXT:   datasize 24

# CHECK-LABEL:  Data in code table (3 entries)
# CHECK-NEXT:   offset length kind
# CHECK-NEXT:   [[#%x,TEXT + 28]] 24 JUMP_TABLE32
# CHECK-NEXT:   [[#%x,TEXT + 68]] 8 JUMP_TABLE32
# CHECK-NEXT:   [[#%x,TEXT + 84]] 12 JUMP_TABLE32

# RUN: %lld -lSystem %t/foo.o %t/bar.o -no_data_in_code_info -o %t/main.exe
# RUN: llvm-otool -l %t/main.exe | FileCheck --check-prefix=OMIT %s

# OMIT-NOT: LC_DATA_IN_CODE

# RUN: %lld -lSystem %t/foo.o %t/bar.o -no_data_in_code_info -data_in_code_info -o %t/main.exe
# RUN: llvm-otool -l %t/main.exe > %t/objdump
# RUN: llvm-otool -Gv %t/main.exe >> %t/objdump
# RUN: FileCheck %s < %t/objdump

#--- foo.s
.text
.section __TEXT,__StaticInit,regular,pure_instructions
.p2align 4, 0x90
_some_init_function:
retq
.p2align 2, 0x90
.data_region jt32
.long 0
.long 0
.end_data_region

.section __TEXT,__text,regular,pure_instructions
.globl _main
.p2align 4, 0x90
_main:
pushq	%rbp
movq	%rsp, %rbp
subq	$16, %rsp
movl	$0, -4(%rbp)
movb	$0, %al
callq	_bar
addq	$16, %rsp
popq	%rbp
retq
.p2align 2, 0x90
.data_region jt32
.long 0
.long 0
.long 0
.long 0
.long 0
.long 0
.end_data_region

#--- bar.s
.text
.globl _bar
.p2align 4
_bar:
retq
.p2align 2, 0x90
.data_region jt32
.long 0
.long 0
.long 0
.end_data_region
