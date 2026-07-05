# REQUIRES: hexagon

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf a.s -o a.o
# RUN: ld.lld -e 0 a.o -o out 2>&1 | count 0
# RUN: llvm-readelf -S -l --arch-specific out | FileCheck %s --check-prefixes=HDR,CHECK
# RUN: ld.lld -e 0 a.o a.o -o out1 2>&1 | count 0
# RUN: llvm-readobj --arch-specific out1 | FileCheck %s
# RUN: ld.lld -r a.o a.o -o out1 2>&1 | count 0
# RUN: llvm-readobj --arch-specific out1 | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf b.s -o b.o
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf c.s -o c.o
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf d.s -o d.o
# RUN: ld.lld a.o b.o c.o -o out2
# RUN: llvm-readobj --arch-specific out2 | FileCheck %s --check-prefix=CHECK2
# RUN: ld.lld a.o b.o c.o d.o -o out3
# RUN: llvm-readobj --arch-specific out3 | FileCheck %s --check-prefix=CHECK3

# HDR:      Name                Type               Address  Off    Size   ES Flg Lk Inf Al
# HDR:      .hexagon.attributes HEXAGON_ATTRIBUTES 00000000 {{.*}} {{.*}} 00     0   0  1{{$}}

# HDR:      Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# HDR:      LOAD           {{.*}}
# HDR-NEXT: GNU_STACK      {{.*}}

# CHECK:      BuildAttributes {
# CHECK-NEXT:   FormatVersion: 0x41
# CHECK-NEXT:   Section 1 {
# CHECK-NEXT:     SectionLength: 19
# CHECK-NEXT:     Vendor: hexagon
# CHECK-NEXT:     Tag: Tag_File (0x1)
# CHECK-NEXT:     Size: 7
# CHECK-NEXT:     FileAttributes {
# CHECK-NEXT:       Attribute {
# CHECK-NEXT:         Tag: 4
# CHECK-NEXT:         TagName: arch
# CHECK-NEXT:         Value: 68{{$}}
# CHECK-NEXT:       }
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK2:      BuildAttributes {
# CHECK2-NEXT:   FormatVersion: 0x41
# CHECK2-NEXT:   Section 1 {
# CHECK2-NEXT:     SectionLength: 21
# CHECK2-NEXT:     Vendor: hexagon
# CHECK2-NEXT:     Tag: Tag_File (0x1)
# CHECK2-NEXT:     Size: 9
# CHECK2-NEXT:     FileAttributes {
# CHECK2-NEXT:       Attribute {
# CHECK2-NEXT:         Tag: 4
# CHECK2-NEXT:         TagName: arch
# CHECK2-NEXT:         Value: 68{{$}}
# CHECK2-NEXT:       }
# CHECK2-NEXT:       Attribute {
# CHECK2-NEXT:         Tag: 5
# CHECK2-NEXT:         TagName: hvx_arch
# CHECK2-NEXT:         Value: 68{{$}}
# CHECK2-NEXT:       }
# CHECK2-NEXT:     }
# CHECK2-NEXT:   }
# CHECK2-NEXT: }

# CHECK3:      BuildAttributes {
# CHECK3-NEXT:   FormatVersion: 0x41
# CHECK3-NEXT:   Section 1 {
# CHECK3-NEXT:     SectionLength: 25
# CHECK3-NEXT:     Vendor: hexagon
# CHECK3-NEXT:     Tag: Tag_File (0x1)
# CHECK3-NEXT:     Size: 13
# CHECK3-NEXT:     FileAttributes {
# CHECK3-NEXT:       Attribute {
# CHECK3-NEXT:         Tag: 7
# CHECK3-NEXT:         TagName: hvx_qfloat
# CHECK3-NEXT:         Value: 68{{$}}
# CHECK3-NEXT:       }
# CHECK3-NEXT:       Attribute {
# CHECK3-NEXT:         Tag: 9
# CHECK3-NEXT:         TagName: audio
# CHECK3-NEXT:         Value: 68{{$}}
# CHECK3-NEXT:       }
# CHECK3-NEXT:       Attribute {
# CHECK3-NEXT:         Tag: 4
# CHECK3-NEXT:         TagName: arch
# CHECK3-NEXT:         Value: 68{{$}}
# CHECK3-NEXT:       }
# CHECK3-NEXT:       Attribute {
# CHECK3-NEXT:         Tag: 5
# CHECK3-NEXT:         TagName: hvx_arch
# CHECK3-NEXT:         Value: 68{{$}}
# CHECK3-NEXT:       }
# CHECK3-NEXT:     }
# CHECK3-NEXT:   }
# CHECK3-NEXT: }

#--- a.s
.section .hexagon.attributes,"",@0x70000003
.byte 0x41
.long .Lend-.hexagon.attributes-1
.asciz "hexagon"
.Lbegin:
.byte 1
.long .Lend-.Lbegin
.byte 4
.byte 68
.Lend:

#--- b.s
.section .hexagon.attributes,"",@0x70000003
.byte 0x41
.long .Lend1-.hexagon.attributes-1
.asciz "hexagon"
.Lbegin1:
.byte 1
.long .Lend1-.Lbegin1
.byte 4
.byte 68
.Lend1:

#--- c.s
.section .hexagon.attributes,"",@0x70000003
.byte 0x41
.long .Lend2-.hexagon.attributes-1
.asciz "hexagon"
.Lbegin2:
.byte 1
.long .Lend2-.Lbegin2
.byte 4
.byte 68
.byte 5
.byte 68
.Lend2:

#--- d.s
.section .hexagon.attributes,"",@0x70000003
.byte 0x41
.long .Lend3-.hexagon.attributes-1
.asciz "hexagon"
.Lbegin3:
.byte 1
.long .Lend3-.Lbegin3
.byte 4
.byte 68
.byte 7
.byte 68
.byte 9
.byte 68
.Lend3:
