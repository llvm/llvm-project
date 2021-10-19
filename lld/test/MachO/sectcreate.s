# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: echo "-sectcreate 1.1" >%t1
# RUN: echo "-sectcreate 1.2" >%t2
# RUN: echo "-sectcreate 2" >%t3
# RUN: %lld \
# RUN:     -sectcreate SEG SEC1 %t1 \
# RUN:     -segcreate SEG SEC2 %t3 \
# RUN:     -sectcreate SEG SEC1 %t2 \
# RUN:     -o %t %t.o
# RUN: llvm-objdump -s %t | FileCheck %s

## -dead_strip does not strip -sectcreate sections,
## but also doesn't set S_ATTR_NO_DEAD_STRIP on them.
# RUN: %lld -dead_strip \
# RUN:     -sectcreate SEG SEC1 %t1 \
# RUN:     -segcreate SEG SEC2 %t3 \
# RUN:     -sectcreate SEG SEC1 %t2 \
# RUN:     -o %t %t.o
# RUN: llvm-objdump -s %t | FileCheck --check-prefix=STRIPPED %s
# RUN: llvm-readobj --sections %t | FileCheck --check-prefix=STRIPPEDSEC %s

# CHECK: Contents of section __TEXT,__text:
# CHECK: Contents of section __DATA,__data:
# CHECK: my string!.
# CHECK: Contents of section SEG,SEC1:
# CHECK: -sectcreate 1.1.
# CHECK: -sectcreate 1.2.
# CHECK: Contents of section SEG,SEC2:
# CHECK: -sectcreate 2.

# STRIPPED: Contents of section __TEXT,__text:
# STRIPPED-NOT: Contents of section __DATA,__data:
# STRIPPED-NOT: my string!.
# STRIPPED: Contents of section SEG,SEC1:
# STRIPPED: -sectcreate 1.1.
# STRIPPED: -sectcreate 1.2.
# STRIPPED: Contents of section SEG,SEC2:
# STRIPPED: -sectcreate 2.

# STRIPPEDSEC-NOT: NoDeadStrip

.text
.global _main
_main:
  mov $0, %eax
  ret

.data
.global my_string
my_string:
  .string "my string!"

.subsections_via_symbols
