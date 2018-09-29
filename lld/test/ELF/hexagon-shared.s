# REQUIRES: hexagon
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf %s -o %t
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf %S/Inputs/hexagon-shared.s -o %t2.o
# RUN: ld.lld -shared %t2.o -soname %t3.so -o %t3.so
# RUN: ld.lld -shared %t %t3.so -soname %t4.so -o %t4.so
# RUN: llvm-objdump -d -j .plt %t4.so | FileCheck %s

.global foo
foo:
call ##bar

# CHECK: { immext(#65472
# CHECK: r28 = add(pc,##65520) }
# CHECK: { r14 -= add(r28,#16)
# CHECK: r15 = memw(r28+#8)
# CHECK: r28 = memw(r28+#4) }
# CHECK: { r14 = asr(r14,#2)
# CHECK: jumpr r28 }
# CHECK: { trap0(#219) }
# CHECK: immext(#65472)
# CHECK: r14 = add(pc,##65488) }
# CHECK: r28 = memw(r14+#0) }
# CHECK: jumpr r28 }
