## Check ADRP page offsets -4GiB, -4GiB-4KiB, +4GiB-4KiB and +4GiB.
## New program headers and alignment place foo at 0x200200000.

# DEFINE: %{link} = %clang %cflags -no-pie -fuse-ld=lld %s -Wl,-q
# RUN: %{link} -Wl,-Ttext=0x1001ffff8 -o %t.bv
# RUN: %{link} -Wl,-Ttext=0x1001feff8 -o %t.biv
# RUN: %{link} -Wl,-Ttext=0x3001feff8 -o %t.fv
# RUN: %{link} -Wl,-Ttext=0x3001ffff8 -o %t.fiv

# DEFINE: %{bolt} = llvm-bolt --funcs=foo --custom-allocation-vma=0x200000000
# RUN: %{bolt} %t.bv -o %t.bv.bolt
# RUN: %{bolt} %t.biv -o %t.biv.bolt
# RUN: %{bolt} %t.fv -o %t.fv.bolt
# RUN: %{bolt} %t.fiv -o %t.fiv.bolt

# RUN: llvm-objdump -d %t.bv.bolt | FileCheck %s --check-prefixes=CHECK,BACKWARD-VALID,VALID
# RUN: llvm-objdump -d %t.biv.bolt | FileCheck %s --check-prefixes=CHECK,BACKWARD-INVALID,INVALID
# RUN: llvm-objdump -d %t.fv.bolt | FileCheck %s --check-prefixes=CHECK,FORWARD-VALID,VALID
# RUN: llvm-objdump -d %t.fiv.bolt | FileCheck %s --check-prefixes=CHECK,FORWARD-INVALID,INVALID

# BACKWARD-INVALID-LABEL: 00000001001ff000 <bar>:
# FORWARD-INVALID-LABEL:  0000000300200000 <bar>:

# CHECK-LABEL:            <foo>:
# CHECK:                  bl 0x200200008 <foo+0x8>

# BACKWARD-VALID:         200200008: {{.*}} adrp x16, 0x100200000 <bar>
# FORWARD-VALID:          200200008: {{.*}} adrp x16, 0x3001ff000 <bar>
# VALID-NEXT:             add x16, x16, #0x0{{$}}

# INVALID:                200200008: {{.*}} movz x16, #0x0, lsl #48{{$}}
# BACKWARD-INVALID-NEXT:  movk x16, #0x1, lsl #32{{$}}
# BACKWARD-INVALID-NEXT:  movk x16, #0x1f, lsl #16{{$}}
# BACKWARD-INVALID-NEXT:  movk x16, #0xf000{{$}}
# FORWARD-INVALID-NEXT:   movk x16, #0x3, lsl #32{{$}}
# FORWARD-INVALID-NEXT:   movk x16, #0x20, lsl #16{{$}}
# FORWARD-INVALID-NEXT:   movk x16, #0x0{{$}}

# CHECK-NEXT:             br x16{{$}}

    .text
    .globl foo
    .type foo, %function
foo:
    bl bar
    ret
    .size foo, .-foo

    .globl bar
    .type bar, %function
bar:
    ret
    .size bar, .-bar
