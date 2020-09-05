# REQUIRES: riscv

# RUN: llvm-mc -filetype=obj -triple=riscv64-unknown-elf -mattr=-relax %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readelf --arch-specific %t | FileCheck %s
# RUN: ld.lld %t.o %t.o -o %t2
# RUN: llvm-readelf --arch-specific %t2 | FileCheck %s

# CHECK:      BuildAttributes {
# CHECK-NEXT:   FormatVersion: 0x41
# CHECK-NEXT:   Section 1 {
# CHECK-NEXT:     SectionLength: 52
# CHECK-NEXT:     Vendor: riscv
# CHECK-NEXT:     Tag: Tag_File (0x1)
# CHECK-NEXT:     Size: 42
# CHECK-NEXT:     FileAttributes {
# CHECK-NEXT:       Attribute {
# CHECK-NEXT:         Tag: 4
# CHECK-NEXT:         Value: 16
# CHECK-NEXT:         TagName: stack_align
# CHECK-NEXT:         Description: Stack alignment is 16-bytes
# CHECK-NEXT:       }
# CHECK-NEXT:       Attribute {
# CHECK-NEXT:         Tag: 5
# CHECK-NEXT:         TagName: arch
# CHECK-NEXT:         Value: rv64i2p0_m2p0_a2p0_f2p0_d2p0_c2p0
# CHECK-NEXT:       }
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }

.attribute 4, 16
.attribute 5, "rv64i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"
