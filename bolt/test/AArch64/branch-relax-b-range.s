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
# RUN: not %{bolt} %t.biv -o %t.biv.bolt 2>&1 | FileCheck %s --check-prefix=OUT-OF-RANGE
# RUN: %{bolt} %t.fv -o %t.fv.bolt
# RUN: not %{bolt} %t.fiv -o %t.fiv.bolt 2>&1 | FileCheck %s --check-prefix=OUT-OF-RANGE

# RUN: llvm-objdump -d %t.bv.bolt | FileCheck %s --check-prefixes=CHECK,BACKWARD-VALID
# RUN: llvm-objdump -d %t.fv.bolt | FileCheck %s --check-prefixes=CHECK,FORWARD-VALID

# CHECK-LABEL:            <_start>:

# BACKWARD-VALID:         b.eq 0x18000200 <_start+0xc>
# BACKWARD-VALID:         18000200: {{.*}} b 0x10000200 <_start.cold.0>
# FORWARD-VALID:          b.eq 0x8000204 <_start+0xc>
# FORWARD-VALID:          8000204: {{.*}} b 0x10000200 <_start.cold.0>

# OUT-OF-RANGE: BOLT-ERROR: Unable to relax non-call branch beyond 128MiB

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
