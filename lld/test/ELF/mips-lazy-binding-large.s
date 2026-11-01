# REQUIRES: mips

## Imported functions follow symbols without GOT entries in .dynsym. Exercise
## the signed immediate boundary and the transition to a five-instruction stub.
# RUN: llvm-mc -filetype=obj -triple=mips -defsym FILLERS=32766 %s -o %t.signed.o
# RUN: ld.lld -shared %t.signed.o -o %t.signed.so
# RUN: llvm-objdump -s -j .MIPS.stubs %t.signed.so | FileCheck %s --check-prefix=SIGNED
# RUN: llvm-readelf --dyn-syms %t.signed.so | FileCheck %s --check-prefix=INDEX32767
# RUN: llvm-readobj --sections %t.signed.so | FileCheck %s --check-prefix=SMALL

# RUN: llvm-mc -filetype=obj -triple=mips -defsym FILLERS=32767 %s -o %t.unsigned.o
# RUN: ld.lld -shared %t.unsigned.o -o %t.unsigned.so
# RUN: llvm-objdump -s -j .MIPS.stubs %t.unsigned.so | FileCheck %s --check-prefix=UNSIGNED
# RUN: llvm-readelf --dyn-syms %t.unsigned.so | FileCheck %s --check-prefix=INDEX32768
# RUN: llvm-readobj --sections %t.unsigned.so | FileCheck %s --check-prefix=SMALL

# RUN: llvm-mc -filetype=obj -triple=mips -defsym FILLERS=65534 %s -o %t.limit.o
# RUN: ld.lld -shared %t.limit.o -o %t.limit.so
# RUN: llvm-objdump -s -j .MIPS.stubs %t.limit.so | FileCheck %s --check-prefix=LIMIT
# RUN: llvm-readelf --dyn-syms %t.limit.so | FileCheck %s --check-prefix=INDEX65535
# RUN: llvm-readobj --sections %t.limit.so | FileCheck %s --check-prefix=SMALL

# RUN: llvm-mc -filetype=obj -triple=mips -defsym FILLERS=65535 %s -o %t.big.o
# RUN: ld.lld -shared %t.big.o -o %t.big.so
# RUN: llvm-objdump -s -j .MIPS.stubs %t.big.so | FileCheck %s --check-prefix=BIG
# RUN: llvm-readelf --dyn-syms %t.big.so | FileCheck %s --check-prefix=INDEX65536
# RUN: llvm-readobj --sections %t.big.so | FileCheck %s --check-prefix=BIGSIZE
# RUN: llvm-readobj -A %t.big.so | FileCheck %s --check-prefix=GOT

# SIGNED:      Contents of section .MIPS.stubs:
# SIGNED-NEXT: {{[0-9a-f]+}} 8f998010 03e07825 0320f809 24187fff
# UNSIGNED:      Contents of section .MIPS.stubs:
# UNSIGNED-NEXT: {{[0-9a-f]+}} 8f998010 03e07825 0320f809 34188000
# LIMIT:      Contents of section .MIPS.stubs:
# LIMIT-NEXT: {{[0-9a-f]+}} 8f998010 03e07825 0320f809 3418ffff
# BIG:      Contents of section .MIPS.stubs:
# BIG-NEXT: {{[0-9a-f]+}} 8f998010 03e07825 3c180001 0320f809
# BIG-NEXT: {{[0-9a-f]+}} 37180000

# INDEX32767: 32767: {{[0-9a-f]+}} 0 FUNC GLOBAL DEFAULT UND target
# INDEX32768: 32768: {{[0-9a-f]+}} 0 FUNC GLOBAL DEFAULT UND target
# INDEX65535: 65535: {{[0-9a-f]+}} 0 FUNC GLOBAL DEFAULT UND target
# INDEX65536: 65536: {{[0-9a-f]+}} 0 FUNC GLOBAL DEFAULT UND target

# SMALL:      Name: .MIPS.stubs
# SMALL:      Address:
# SMALL-NEXT: Offset:
# SMALL-NEXT: Size: 16
# BIGSIZE:      Name: .MIPS.stubs
# BIGSIZE:      Address:
# BIGSIZE-NEXT: Offset:
# BIGSIZE-NEXT: Size: 20

# GOT:      Global entries [
# GOT:        Initial: 0x[[STUB:[1-9A-F][0-9A-F]*]]
# GOT-NEXT:   Value: 0x[[STUB]]
# GOT-NEXT:   Type: Function
# GOT-NEXT:   Section: Undefined
# GOT-NEXT:   Name: target

.text
.globl target
.type target,@function
lw $25, %call16(target)($gp)

.macro export
.globl filler\@
filler\@:
.endm
.rept FILLERS
export
.endr
