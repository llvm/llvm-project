# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t.out
# RUN: llvm-objdump -s %t.out| FileCheck %s --check-prefix=BEFORE

# BEFORE:      Contents of section .foo:
# BEFORE-NEXT:  11223344 556677
# BEFORE:      Contents of section .init:
# BEFORE-NEXT:  1122

# RUN: echo "_foo4  " > %t_order.txt
# RUN: echo "  _foo3" >> %t_order.txt
# RUN: echo "_nönascii" >> %t_order.txt
# RUN: echo "_foo5" >> %t_order.txt
# RUN: echo -e " _foo2 \t\v\r\f" >> %t_order.txt
# RUN: echo "# a comment" >> %t_order.txt
# RUN: echo " " >> %t_order.txt
# RUN: echo "_foo4" >> %t_order.txt
# RUN: echo "_bar1" >> %t_order.txt
# RUN: echo "" >> %t_order.txt
# RUN: echo "_foo1" >> %t_order.txt
# RUN: echo "_init2" >> %t_order.txt
# RUN: echo "_init1" >> %t_order.txt

## Show that both --symbol-ordering-file and --symbol-ordering-file= work.
## Also show that lines beginning with '#', whitespace and empty lines, are ignored.
## Also show that "first one wins" when there are duplicate symbols in the order
## file (a warning will be emitted).
# RUN: ld.lld --symbol-ordering-file %t_order.txt %t.o -o %t2.out 2>&1 | \
# RUN:   FileCheck %s --check-prefix=WARN --implicit-check-not=warning:
# RUN: llvm-objdump -s %t2.out | FileCheck %s --check-prefix=AFTER
# RUN: ld.lld --symbol-ordering-file=%t_order.txt %t.o -o %t2.out
# RUN: llvm-objdump -s %t2.out | FileCheck %s --check-prefix=AFTER

# WARN: warning: {{.*}}.txt: duplicate ordered symbol

# AFTER:      Contents of section .foo:
# AFTER-NEXT:  44337755 662211
# AFTER:      Contents of section .init:
# AFTER-NEXT:  1122

## Show that a nonexistent symbol ordering file causes an error.
# RUN: not ld.lld --symbol-ordering-file=%t.nonexistent %t.o -o %t3.out 2>&1 | \
# RUN:   FileCheck -DMSG=%errc_ENOENT %s --check-prefix=ERR -DFILE=%t.nonexistent

# ERR: error: cannot open [[FILE]]: [[MSG]]

## Show that an empty ordering file can be handled (symbols remain in their
## normal order).
# RUN: touch %t.empty
# RUN: ld.lld --symbol-ordering-file=%t.empty %t.o -o %t4.out
# RUN: llvm-objdump -s %t4.out | FileCheck %s --check-prefix=BEFORE

.section .foo,"ax",@progbits,unique,1
_foo1:
 .byte 0x11

.section .foo,"ax",@progbits,unique,2
_foo2:
 .byte 0x22

.section .foo,"ax",@progbits,unique,3
_foo3:
 .byte 0x33

.section .foo,"ax",@progbits,unique,4
_foo4:
 .byte 0x44

.section .foo,"ax",@progbits,unique,5
_foo5:
 .byte 0x55
_bar1:
 .byte 0x66

.section .foo,"ax",@progbits,unique,6
"_nönascii":
.byte 0x77

.section .init,"ax",@progbits,unique,1
_init1:
 .byte 0x11

.section .init,"ax",@progbits,unique,2
_init2:
 .byte 0x22

.text
.global _start
_start:
    nop
