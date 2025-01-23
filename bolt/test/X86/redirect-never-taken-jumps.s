## Test redirect-never-taken-jumps

# RUN: llvm-mc --filetype=obj --triple x86_64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata1 FDATA1
# RUN: link_fdata %s %t.o %t.fdata2 FDATA2
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q

# RUN: llvm-bolt %t.exe -o %t.bolt --reorder-blocks=none --split-functions=1 \
# RUN:         --redirect-never-taken-jumps --print-redirect-never-taken --data=%t.fdata1 \
# RUN:     2>&1 | FileCheck --check-prefix=CHECK_REGULAR %s

# RUN: llvm-bolt %t.exe -o %t.bolt --reorder-blocks=none --split-functions=1 \
# RUN:         --redirect-never-taken-jumps --data=%t.fdata2 \
# RUN:     2>&1 | FileCheck --check-prefix=CHECK_CONSERVATIVE_DEFAULT %s

# RUN: llvm-bolt %t.exe -o %t.bolt --reorder-blocks=none --split-functions=1 \
# RUN:         --redirect-never-taken-jumps --conservative-never-taken-threshold=2.0 --data=%t.fdata2 \
# RUN:     2>&1 | FileCheck --check-prefix=CHECK_CONSERVATIVE_THRESHOLD %s

# RUN: llvm-bolt %t.exe -o %t.bolt --reorder-blocks=none --split-functions=1 \
# RUN:         --redirect-never-taken-jumps --aggressive-never-taken --data=%t.fdata2 \
# RUN:     2>&1 | FileCheck --check-prefix=CHECK_AGGRRESSIVE %s

# CHECK_REGULAR: redirection of never-taken jumps saved 4 bytes hot section code size and 12 bytes total code size
# CHECK_REGULAR: .LBB00 (1 instructions, align : 1)
# CHECK_REGULAR: Successors: .LBB2 (mispreds: 0, count: 0), .Ltmp2 (mispreds: 0, count: 20)
# CHECK_REGULAR: .Ltmp2 (1 instructions, align : 1)
# CHECK_REGULAR: Successors: .LBB2 (mispreds: 0, count: 20)
# CHECK_REGULAR: .LBB2 (1 instructions, align : 1)
# CHECK_REGULAR: Successors: .Ltmp0 (mispreds: 0, count: 0), .LFT0 (mispreds: 0, count: 20)
# CHECK_REGULAR: .Ltmp3 (1 instructions, align : 1)
# CHECK_REGULAR: HOT-COLD SPLIT POINT
# CHECK_REGULAR: .Ltmp0 (2 instructions, align : 1)
# CHECK_REGULAR: Successors: .LBB1 (mispreds: 0, count: 0), .LBB00 (mispreds: 0, count: 0)
# CHECk_REGULAR: .Ltmp4 (1 instructions, align : 1)
# CHECK_REGULAR: Successors: .LBB1 (mispreds: 0, count: 0)
# CHECk_REGULAR: .LBB1 (2 instructions, align : 1)
# CHECk_REGULAR: Successors: .LBB0 (mispreds: 0, count: 0), .Ltmp2 (mispreds: 0, count: 0)
# CHECk_REGULAR: .Ltmp1 (1 instructions, align : 1)
# CHECK_REGULAR: Successors: .Ltmp4 (mispreds: 0, count: 0), .LBB0 (mispreds: 0, count: 0)
# CHECk_REGULAR: .LBB0 (1 instructions, align : 1)
# CHECK_REGULAR: Successors: .Ltmp3 (mispreds: 0, count: 0)

# CHECK_CONSERVATIVE_DEFAULT: redirection of never-taken jumps saved 4 bytes hot section code size and 8 bytes total code size
# CHECK_CONSERVATIVE_THRESHOLD: redirection of never-taken jumps saved 4 bytes hot section code size and 12 bytes total code size
# CHECK_AGGRRESSIVE: redirection of never-taken jumps saved 4 bytes hot section code size and 12 bytes total code size


        .globl  main
        .type   main, @function
main:
LBB00:
        ja Ltmp0
# FDATA1: 1 main #LBB00# 1 main #Ltmp0# 0 0
# FDATA1: 1 main #LBB00# 1 main #Ltmp2# 0 20
# FDATA2: 1 main #LBB00# 1 main #Ltmp0# 0 0
# FDATA2: 1 main #LBB00# 1 main #Ltmp2# 0 10
Ltmp2:
        cmpl $0,%eax
Ltmp2Jcc:
        ja Ltmp0
# FDATA1: 1 main #Ltmp2Jcc# 1 main #Ltmp0# 0 0
# FDATA1: 1 main #Ltmp2Jcc# 1 main #LFT0# 0 20
# FDATA2: 1 main #Ltmp2Jcc# 1 main #Ltmp0# 0 0
# FDATA2: 1 main #Ltmp2Jcc# 1 main #LFT0# 0 20
LFT0:
        ja Ltmp1
# FDATA1: 1 main #LFT0# 1 main #Ltmp1# 0 0
# FDATA1: 1 main #LFT0# 1 main #Ltmp3# 0 20
# FDATA2: 1 main #LFT0# 1 main #Ltmp1# 0 0
# FDATA2: 1 main #LFT0# 1 main #Ltmp3# 0 20
Ltmp3:
        ret
Ltmp0:
        jae Ltmp2
        jmp LBB00
Ltmp4:
        cmpl $0,%eax
        jae Ltmp2
        jmp Ltmp3
Ltmp1:
        je Ltmp4
        jmp Ltmp3
.LLmain_end:
        .size   main, .LLmain_end-main
