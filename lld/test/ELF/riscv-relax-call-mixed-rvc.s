# REQUIRES: riscv
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=-c,+relax b.s -o b.o

# RUN: ld.lld a.o b.o --shared -o a -Ttext=0x10000
# RUN: llvm-objdump -d --no-show-raw-insn -M no-aliases a | FileCheck %s

## This needs to be a *uncompressed* jal instruction since it came from the
## source file which does not enable C
# CHECK-LABEL: <foo>:
# CHECK-NEXT:    10000: jal zero, {{.*}} <foo>
# CHECK-NEXT:    10004: sub zero, zero, zero

# w/ C
#--- a.s
	.text
	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0_zifencei2p0"

# w/o C
#--- b.s
	.text
	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_zicsr2p0_zifencei2p0"
	.p2align	5
	.type	foo,@function
foo:
    tail    foo
    # Pick a non-canonical nop to ensure test output can't be confused
    # with riscv_align padding
    sub zero, zero, zero
