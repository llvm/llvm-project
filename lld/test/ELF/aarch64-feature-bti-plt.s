# REQUIRES: aarch64

# RUN: rm -rf %t && split-file %s %t

# RUN: llvm-mc --triple=aarch64 --filetype=obj -o %t.o %t/a.s
# RUN: ld.lld --shared -T %t/largegap.lds -z force-bti %t.o -o %t.elf
# RUN: llvm-objdump -d %t.elf | FileCheck %s

#--- largegap.lds
SECTIONS {
  .plt : { *(.plt) }
  .text.near 0x1000 : AT(0x1000) { *(.text.near) }
  .text.far 0xf0000000 : AT(0xf0000000) { *(.text.far) }
}

#--- a.s
# CHECK:        <.plt>:
# CHECK-NEXT:     bti     c

## foo@plt is targeted by a range extension thunk with an indirect branch.
## Add a bti c instruction.
# CHECK:        <foo@plt>:
# CHECK-NEXT:     bti     c

## biz is not targeted by a thunk using an indirect branch, so no need for bti c.
# CHECK:        <biz@plt>:
# CHECK-NEXT:     adrp    x16, {{.*}} <func>

# CHECK:         <bar>:
# CHECK-NEXT:      bl   {{.*}} <foo@plt>
# CHECK-NEXT:      bl   {{.*}} <biz@plt>

# CHECK:         <func>:
# CHECK-NEXT:      bl   {{.*}} <__AArch64ADRPThunk_foo>

# CHECK:         <__AArch64ADRPThunk_foo>:
# CHECK-NEXT:      adrp    x16, 0x0 <foo>
# CHECK-NEXT:      add     x16, x16, {{.*}}
# CHECK-NEXT:      br      x16

        .global foo
        .global biz
        .section .text.near, "ax", %progbits
bar:
        .type bar, %function
        bl foo
        bl biz
        ret

        .section .text.far, "ax", %progbits
func:
        .type func, %function
        bl foo
        ret
