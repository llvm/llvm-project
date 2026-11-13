## Check B at -128MiB, -128MiB-4, +128MiB-4 and +128MiB.
## The cold target follows the new program headers at 0x10000200.

# DEFINE: %{link} = %clang %cflags -no-pie -fuse-ld=lld -Wa,-L %s
# RUN: %{link} -Wl,-Ttext=0x180001f4 -o %t.bv
# RUN: %{link} -Wl,-Ttext=0x180001f8 -o %t.biv
# RUN: %{link} -Wl,-Ttext=0x080001f8 -o %t.fv
# RUN: %{link} -Wl,-Ttext=0x080001f4 -o %t.fiv
# RUN: link_fdata --no-lbr %s %t.bv %t.fdata

# DEFINE: %{bolt} = llvm-bolt --relocs=0 --data=%t.fdata --split-functions \
# DEFINE:   --split-all-cold --keep-nops --custom-allocation-vma=0x10000000
# RUN: %{bolt} %t.bv -o %t.bv.bolt
# RUN: %{bolt} %t.biv -o %t.biv.bolt
# RUN: %{bolt} %t.fv -o %t.fv.bolt
# RUN: %{bolt} %t.fiv -o %t.fiv.bolt

# RUN: llvm-objdump -d %t.bv.bolt | FileCheck %s --check-prefixes=CHECK,BACKWARD-VALID
# RUN: llvm-objdump -d %t.biv.bolt | FileCheck %s --check-prefixes=CHECK,BACKWARD-INVALID,INVALID
# RUN: llvm-objdump -d %t.fv.bolt | FileCheck %s --check-prefixes=CHECK,FORWARD-VALID
# RUN: llvm-objdump -d %t.fiv.bolt | FileCheck %s --check-prefixes=CHECK,FORWARD-INVALID,INVALID

# CHECK-LABEL:            <_start>:

# BACKWARD-VALID:         b.eq 0x18000200 <_start+0xc>
# BACKWARD-VALID:         18000200: {{.*}} b 0x10000200 <_start.cold.0>
# FORWARD-VALID:          b.eq 0x8000204 <_start+0xc>
# FORWARD-VALID:          8000204: {{.*}} b 0x10000200 <_start.cold.0>

# BACKWARD-INVALID:       b.eq 0x18000204 <_start+0xc>
# BACKWARD-INVALID:       18000204: {{.*}} adrp x16, 0x10000000
# FORWARD-INVALID:        b.eq 0x8000200 <_start+0xc>
# FORWARD-INVALID:        8000200: {{.*}} adrp x16, 0x10000000
# INVALID-NEXT:           add x16, x16, #0x200{{$}}
# INVALID-NEXT:           br x16{{$}}

# CHECK:                  0000000010000200 <_start.cold.0>:

    .text
    .globl _start
    .type _start, %function
_start:
# FDATA: 1 _start #_start# 10
    cmp x0, #0
    b.eq .Lcold
.Lhot:
# FDATA: 1 _start #.Lhot# 10
    ret
.Lcold:
# FDATA: 1 _start #.Lcold# 0
    .rept 4
    nop
    .endr
    ret
    .size _start, .-_start
