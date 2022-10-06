# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/first.s -o %t/first.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/second.s -o %t/second.o

# RUN: %lld -lSystem -init_offsets -undefined dynamic_lookup  %t/first.o %t/second.o -o %t/out
# RUN: llvm-otool -lv %t/out | FileCheck --check-prefix=FLAGS --implicit-check-not=__mod_init_func %s
# RUN: llvm-otool -l %t/out > %t/dump.txt
# RUN: llvm-objdump --macho --print-imm-hex --section=__TEXT,__stubs %t/out >> %t/dump.txt
# RUN: llvm-objdump --macho --syms %t/out >> %t/dump.txt
# RUN: llvm-objcopy --dump-section=__TEXT,__init_offsets=%t/section.bin %t/out
# RUN: echo "__TEXT,__init_offsets contents:" >> %t/dump.txt
# RUN: od -An -txI %t/section.bin >> %t/dump.txt
# RUN: FileCheck --check-prefix=CONTENT %s < %t/dump.txt

## This test checks that:
## - __mod_init_func is replaced by __init_offsets.
## - __mod_init_func has type S_INIT_FUNC_OFFSETS.
## - initializers show up in the order their parent objects are specified on the
##   command line, and in the order they show up within __mod_init_func.
## - for undefined and dylib symbols, stubs are created, and the offsets point to those.
## - offsets are relative to __TEXT's address, they aren't an absolute virtual address.

# FLAGS:      sectname __init_offsets
# FLAGS-NEXT:  segname __TEXT
# FLAGS-NEXT:     addr
# FLAGS-NEXT:     size 0x0000000000000010
# FLAGS-NEXT:   offset
# FLAGS-NEXT:    align
# FLAGS-NEXT:   reloff 0
# FLAGS-NEXT:   nreloc 0
# FLAGS-NEXT:     type S_INIT_FUNC_OFFSETS

# CONTENT:      segname __TEXT
# CONTENT-NEXT: 0x[[#%x, TEXT:]]

# CONTENT:      Contents of (__TEXT,__stubs) section
# CONTENT-NEXT: [[#%x, ISNAN:]]: {{.*}} ## literal pool symbol address: ___isnan
# CONTENT-NEXT: [[#%x, UNDEF:]]: {{.*}} ## literal pool symbol address: _undefined

# CONTENT: SYMBOL TABLE:
# CONTENT: [[#%x, FIRST:]]  g F __TEXT,__text _first_init
# CONTENT: [[#%x, SECOND:]] g F __TEXT,__text _second_init

# CONTENT: __TEXT,__init_offsets contents:
# CONTENT: [[#%.8x, FIRST - TEXT]] [[#%.8x, ISNAN - TEXT]] [[#%.8x, UNDEF - TEXT]] [[#%.8x, SECOND - TEXT]]

#--- first.s
.globl _first_init, ___isnan, _main
.text
_first_init:
  ret
_main:
  ret

.section __DATA,__mod_init_func,mod_init_funcs
.quad _first_init
.quad ___isnan

.subsections_via_symbols

#--- second.s
.globl _second_init, _undefined
.text
_second_init:
  ret

.section __DATA,__mod_init_func,mod_init_funcs
.quad _undefined
.quad _second_init

.subsections_via_symbols
