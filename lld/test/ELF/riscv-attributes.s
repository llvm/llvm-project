# REQUIRES: riscv

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=riscv64 a.s -o a.o
# RUN: ld.lld -e 0 a.o -o out 2>&1 | count 0
# RUN: llvm-readelf -S -l --arch-specific out | FileCheck %s --check-prefixes=HDR,CHECK
# RUN: ld.lld -e 0 a.o a.o -o out1 2>&1 | count 0
# RUN: llvm-readobj --arch-specific out1 | FileCheck %s
# RUN: ld.lld -r a.o a.o -o out1 2>&1 | count 0
# RUN: llvm-readobj --arch-specific out1 | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=riscv64 b.s -o b.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 c.s -o c.o
# RUN: ld.lld a.o b.o c.o -o out2
# RUN: llvm-readobj --arch-specific out2 | FileCheck %s --check-prefix=CHECK2

# RUN: llvm-mc -filetype=obj -triple=riscv64 unrecognized_ext1.s -o unrecognized_ext1.o
# RUN: ld.lld -e 0 unrecognized_ext1.o -o unrecognized_ext1 2>&1 | count 0
# RUN: llvm-readobj --arch-specific unrecognized_ext1 | FileCheck %s --check-prefix=UNRECOGNIZED_EXT1

# RUN: llvm-mc -filetype=obj -triple=riscv64 unrecognized_ext2.s -o unrecognized_ext2.o
# RUN: ld.lld -e 0 unrecognized_ext2.o -o unrecognized_ext2 2>&1 | count 0
# RUN: llvm-readobj --arch-specific unrecognized_ext2 | FileCheck %s --check-prefix=UNRECOGNIZED_EXT2

# RUN: llvm-mc -filetype=obj -triple=riscv64 unrecognized_version.s -o unrecognized_version.o
# RUN: ld.lld -e 0 unrecognized_version.o -o unrecognized_version 2>&1 | count 0
# RUN: llvm-readobj --arch-specific unrecognized_version | FileCheck %s --check-prefix=UNRECOGNIZED_VERSION

# RUN: llvm-mc -filetype=obj -triple=riscv64 merge_version_test_input.s -o merge_version_test_input.o
# RUN: ld.lld -e 0 unrecognized_version.o merge_version_test_input.o -o out3 2>&1 | count 0
# RUN: llvm-readobj --arch-specific out3 | FileCheck %s --check-prefix=CHECK3

# RUN: llvm-mc -filetype=obj -triple=riscv64 invalid_arch1.s -o invalid_arch1.o
# RUN: not ld.lld invalid_arch1.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=INVALID_ARCH1 --implicit-check-not=error:
# INVALID_ARCH1: error: invalid_arch1.o:(.riscv.attributes): rv64i2: extension lacks version in expected format

## A zero value attribute is not printed.
# RUN: llvm-mc -filetype=obj -triple=riscv64 unaligned_access_0.s -o unaligned_access_0.o
# RUN: ld.lld -e 0 --fatal-warnings a.o unaligned_access_0.o -o unaligned_access_0
# RUN: llvm-readobj -A unaligned_access_0 | FileCheck /dev/null --implicit-check-not='TagName: unaligned_access'

## Differing stack_align values lead to an error.
# RUN: llvm-mc -filetype=obj -triple=riscv64 diff_stack_align.s -o diff_stack_align.o
# RUN: not ld.lld a.o b.o c.o diff_stack_align.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=STACK_ALIGN --implicit-check-not=error:
# STACK_ALIGN: error: diff_stack_align.o:(.riscv.attributes) has stack_align=32 but a.o:(.riscv.attributes) has stack_align=16

## RISC-V tag merging for atomic_abi values A6C and A7 lead to an error.
# RUN: llvm-mc -filetype=obj -triple=riscv64  atomic_abi_A6C.s -o atomic_abi_A6C.o
# RUN: llvm-mc -filetype=obj -triple=riscv64  atomic_abi_A7.s -o atomic_abi_A7.o
# RUN: not ld.lld atomic_abi_A6C.o atomic_abi_A7.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ATOMIC_ABI_ERROR --implicit-check-not=error:
# ATOMIC_ABI_ERROR: error: atomic abi mismatch for .riscv.attributes
# ATOMIC_ABI_ERROR-NEXT: >>> atomic_abi_A6C.o:(.riscv.attributes): atomic_abi=1
# ATOMIC_ABI_ERROR-NEXT: >>> atomic_abi_A7.o:(.riscv.attributes): atomic_abi=3

## RISC-V tag merging for atomic_abi values A6C and invalid lead to an error.
# RUN: llvm-mc -filetype=obj -triple=riscv64  atomic_abi_invalid.s -o atomic_abi_invalid.o
# RUN: not ld.lld atomic_abi_A6C.o atomic_abi_invalid.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ATOMIC_ABI_INVALID --implicit-check-not=error:
# ATOMIC_ABI_INVALID: error: unknown atomic abi for .riscv.attributes
# ATOMIC_ABI_INVALID-NEXT: >>> atomic_abi_invalid.o:(.riscv.attributes): atomic_abi=42

# RUN: llvm-mc -filetype=obj -triple=riscv64  atomic_abi_A6S.s -o atomic_abi_A6S.o
# RUN: ld.lld atomic_abi_A6S.o atomic_abi_A6C.o -o atomic_abi_A6C_A6S
# RUN: llvm-readobj -A atomic_abi_A6C_A6S | FileCheck %s --check-prefix=A6C_A6S

# RUN: ld.lld atomic_abi_A6S.o atomic_abi_A7.o -o atomic_abi_A6S_A7
# RUN: llvm-readobj -A atomic_abi_A6S_A7 | FileCheck %s --check-prefix=A6S_A7

# RUN: llvm-mc -filetype=obj -triple=riscv64  atomic_abi_unknown.s -o atomic_abi_unknown.o
# RUN: ld.lld atomic_abi_unknown.o atomic_abi_A6C.o -o atomic_abi_A6C_unknown
# RUN: llvm-readobj -A atomic_abi_A6C_unknown | FileCheck %s --check-prefixes=UNKNOWN_A6C

# RUN: ld.lld atomic_abi_unknown.o diff_stack_align.o -o atomic_abi_none_unknown
# RUN: llvm-readobj -A atomic_abi_none_unknown | FileCheck %s --check-prefixes=UNKNOWN_NONE

# RUN: ld.lld diff_stack_align.o atomic_abi_A6C.o -o atomic_abi_A6C_none
# RUN: llvm-readobj -A atomic_abi_A6C_none | FileCheck %s --check-prefixes=NONE_A6C

# RUN: ld.lld atomic_abi_unknown.o atomic_abi_A6S.o -o atomic_abi_A6S_unknown
# RUN: llvm-readobj -A atomic_abi_A6S_unknown | FileCheck %s --check-prefix=UNKNOWN_A6S

# RUN: ld.lld atomic_abi_unknown.o atomic_abi_A7.o -o atomic_abi_A7_unknown
# RUN: llvm-readobj -A atomic_abi_A7_unknown | FileCheck %s --check-prefix=UNKNOWN_A7

# RUN: ld.lld diff_stack_align.o atomic_abi_A7.o -o atomic_abi_A7_none
# RUN: llvm-readobj -A atomic_abi_A7_none | FileCheck %s --check-prefix=NONE_A7

## The deprecated priv_spec is not handled as GNU ld does.
## Differing priv_spec attributes lead to an absent attribute.
# RUN: llvm-mc -filetype=obj -triple=riscv64 diff_priv_spec.s -o diff_priv_spec.o
# RUN: ld.lld -e 0 --fatal-warnings a.o b.o c.o diff_priv_spec.o -o diff_priv_spec
# RUN: llvm-readobj -A diff_priv_spec | FileCheck /dev/null --implicit-check-not='TagName: priv_spec'

## Unknown tags currently lead to warnings.
# RUN: llvm-mc -filetype=obj -triple=riscv64 unknown13.s -o unknown13.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 unknown13a.s -o unknown13a.o
# RUN: ld.lld -e 0 unknown13.o unknown13.o unknown13a.o -o unknown13 2>&1 | FileCheck %s --check-prefix=UNKNOWN13 --implicit-check-not=warning:
# UNKNOWN13-COUNT-2: warning: unknown13.o:(.riscv.attributes): invalid tag 0xd at offset 0x10
# UNKNOWN13:         warning: unknown13a.o:(.riscv.attributes): invalid tag 0xd at offset 0x10

# RUN: llvm-mc -filetype=obj -triple=riscv64 unknown22.s -o unknown22.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 unknown22a.s -o unknown22a.o
# RUN: ld.lld -e 0 unknown22.o unknown22.o unknown22a.o -o unknown22 2>&1 | FileCheck %s --check-prefix=UNKNOWN22 --implicit-check-not=warning:
# UNKNOWN22-COUNT-2: warning: unknown22.o:(.riscv.attributes): invalid tag 0x16 at offset 0x10
# UNKNOWN22:         warning: unknown22a.o:(.riscv.attributes): invalid tag 0x16 at offset 0x10

# HDR:      Name              Type             Address          Off    Size   ES Flg Lk Inf Al
# HDR:      .riscv.attributes RISCV_ATTRIBUTES 0000000000000000 000158 00005a 00      0   0  1{{$}}

# HDR:      Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# HDR:      LOAD           0x000000 0x0000000000010000 0x0000000000010000 0x000158 0x000158 R   0x1000
# HDR-NEXT: GNU_STACK      0x000000 0x0000000000000000 0x0000000000000000 0x000000 0x000000 RW  0
# HDR-NEXT: ATTRIBUTES     0x000158 0x0000000000000000 0x0000000000000000 0x00005a 0x00005a R   0x1{{$}}

# CHECK:      BuildAttributes {
# CHECK-NEXT:   FormatVersion: 0x41
# CHECK-NEXT:   Section 1 {
# CHECK-NEXT:     SectionLength: 89
# CHECK-NEXT:     Vendor: riscv
# CHECK-NEXT:     Tag: Tag_File (0x1)
# CHECK-NEXT:     Size: 79
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
# CHECK-NEXT:         Value: rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0_zmmul1p0_zaamo1p0_zalrsc1p0{{$}}
# CHECK-NEXT:       }
# CHECK-NEXT:     }
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK2:      BuildAttributes {
# CHECK2-NEXT:   FormatVersion: 0x41
# CHECK2-NEXT:   Section 1 {
# CHECK2-NEXT:     SectionLength: 132
# CHECK2-NEXT:     Vendor: riscv
# CHECK2-NEXT:     Tag: Tag_File (0x1)
# CHECK2-NEXT:     Size: 122
# CHECK2-NEXT:     FileAttributes {
# CHECK2-NEXT:       Attribute {
# CHECK2-NEXT:         Tag: 4
# CHECK2-NEXT:         Value: 16
# CHECK2-NEXT:         TagName: stack_align
# CHECK2-NEXT:         Description: Stack alignment is 16-bytes
# CHECK2-NEXT:       }
# CHECK2-NEXT:       Attribute {
# CHECK2-NEXT:         Tag: 6
# CHECK2-NEXT:         Value: 1
# CHECK2-NEXT:         TagName: unaligned_access
# CHECK2-NEXT:         Description: Unaligned access
# CHECK2-NEXT:       }
# CHECK2-NEXT:       Attribute {
# CHECK2-NEXT:         Tag: 8
# CHECK2-NEXT:         TagName: priv_spec
# CHECK2-NEXT:         Value: 2
# CHECK2-NEXT:       }
# CHECK2-NEXT:       Attribute {
# CHECK2-NEXT:         Tag: 10
# CHECK2-NEXT:         TagName: priv_spec_minor
# CHECK2-NEXT:         Value: 2
# CHECK2-NEXT:       }
# CHECK2-NEXT:       Attribute {
# CHECK2-NEXT:         Tag: 5
# CHECK2-NEXT:         TagName: arch
# CHECK2-NEXT:         Value: rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zkt1p0_zve32f1p0_zve32x1p0_zvl32b1p0{{$}}
# CHECK2-NEXT:       }
# CHECK2-NEXT:     }
# CHECK2-NEXT:   }
# CHECK2-NEXT: }

# CHECK3:      BuildAttributes {
# CHECK3-NEXT:   FormatVersion: 0x41
# CHECK3-NEXT:   Section 1 {
# CHECK3-NEXT:     SectionLength: 26
# CHECK3-NEXT:     Vendor: riscv
# CHECK3-NEXT:     Tag: Tag_File (0x1)
# CHECK3-NEXT:     Size: 16
# CHECK3-NEXT:     FileAttributes {
# CHECK3-NEXT:       Attribute {
# CHECK3-NEXT:         Tag: 5
# CHECK3-NEXT:         TagName: arch
# CHECK3-NEXT:         Value: rv64i99p0{{$}}
# CHECK3-NEXT:       }
# CHECK3-NEXT:     }
# CHECK3-NEXT:   }
# CHECK3-NEXT: }

#--- a.s
.attribute stack_align, 16
.attribute arch, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zmmul1p0"
.attribute unaligned_access, 0

#--- b.s
.attribute stack_align, 16
.attribute arch, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zmmul1p0"
.attribute priv_spec, 2
.attribute priv_spec_minor, 2

#--- c.s
.attribute stack_align, 16
.attribute arch, "rv64i2p1_f2p2_zkt1p0_zve32f1p0_zve32x1p0_zvl32b1p0"
.attribute unaligned_access, 1
.attribute priv_spec, 2
.attribute priv_spec_minor, 2

#--- unrecognized_ext1.s
# UNRECOGNIZED_EXT1:      BuildAttributes {
# UNRECOGNIZED_EXT1-NEXT:   FormatVersion: 0x41
# UNRECOGNIZED_EXT1-NEXT:   Section 1 {
# UNRECOGNIZED_EXT1-NEXT:     SectionLength: 30
# UNRECOGNIZED_EXT1-NEXT:     Vendor: riscv
# UNRECOGNIZED_EXT1-NEXT:     Tag: Tag_File (0x1)
# UNRECOGNIZED_EXT1-NEXT:     Size: 20
# UNRECOGNIZED_EXT1-NEXT:     FileAttributes {
# UNRECOGNIZED_EXT1-NEXT:       Attribute {
# UNRECOGNIZED_EXT1-NEXT:         Tag: 5
# UNRECOGNIZED_EXT1-NEXT:         TagName: arch
# UNRECOGNIZED_EXT1-NEXT:         Value: rv64i2p1_y2p0{{$}}
# UNRECOGNIZED_EXT1-NEXT:       }
# UNRECOGNIZED_EXT1-NEXT:     }
# UNRECOGNIZED_EXT1-NEXT:   }
# UNRECOGNIZED_EXT1-NEXT: }
.section .riscv.attributes,"",@0x70000003
.byte 0x41
.long .Lend-.riscv.attributes-1
.asciz "riscv"  # vendor
.Lbegin:
.byte 1  # Tag_File
.long .Lend-.Lbegin
.byte 5  # Tag_RISCV_arch
.asciz "rv64i2p1_y2p0"
.Lend:

#--- unrecognized_ext2.s
# UNRECOGNIZED_EXT2:      BuildAttributes {
# UNRECOGNIZED_EXT2-NEXT:   FormatVersion: 0x41
# UNRECOGNIZED_EXT2-NEXT:   Section 1 {
# UNRECOGNIZED_EXT2-NEXT:     SectionLength: 36
# UNRECOGNIZED_EXT2-NEXT:     Vendor: riscv
# UNRECOGNIZED_EXT2-NEXT:     Tag: Tag_File (0x1)
# UNRECOGNIZED_EXT2-NEXT:     Size: 26
# UNRECOGNIZED_EXT2-NEXT:     FileAttributes {
# UNRECOGNIZED_EXT2-NEXT:       Attribute {
# UNRECOGNIZED_EXT2-NEXT:         Tag: 5
# UNRECOGNIZED_EXT2-NEXT:         TagName: arch
# UNRECOGNIZED_EXT2-NEXT:         Value: rv64i2p1_zmadeup1p0{{$}}
# UNRECOGNIZED_EXT2-NEXT:       }
# UNRECOGNIZED_EXT2-NEXT:     }
# UNRECOGNIZED_EXT2-NEXT:   }
# UNRECOGNIZED_EXT2-NEXT: }
.section .riscv.attributes,"",@0x70000003
.byte 0x41
.long .Lend-.riscv.attributes-1
.asciz "riscv"  # vendor
.Lbegin:
.byte 1  # Tag_File
.long .Lend-.Lbegin
.byte 5  # Tag_RISCV_arch
.asciz "rv64i2p1_zmadeup1p0"
.Lend:

#--- unrecognized_version.s
# UNRECOGNIZED_VERSION:      BuildAttributes {
# UNRECOGNIZED_VERSION-NEXT:   FormatVersion: 0x41
# UNRECOGNIZED_VERSION-NEXT:   Section 1 {
# UNRECOGNIZED_VERSION-NEXT:     SectionLength: 26
# UNRECOGNIZED_VERSION-NEXT:     Vendor: riscv
# UNRECOGNIZED_VERSION-NEXT:     Tag: Tag_File (0x1)
# UNRECOGNIZED_VERSION-NEXT:     Size: 16
# UNRECOGNIZED_VERSION-NEXT:     FileAttributes {
# UNRECOGNIZED_VERSION-NEXT:       Attribute {
# UNRECOGNIZED_VERSION-NEXT:         Tag: 5
# UNRECOGNIZED_VERSION-NEXT:         TagName: arch
# UNRECOGNIZED_VERSION-NEXT:         Value: rv64i99p0
# UNRECOGNIZED_VERSION-NEXT:       }
# UNRECOGNIZED_VERSION-NEXT:     }
# UNRECOGNIZED_VERSION-NEXT:   }
# UNRECOGNIZED_VERSION-NEXT: }
.section .riscv.attributes,"",@0x70000003
.byte 0x41
.long .Lend-.riscv.attributes-1
.asciz "riscv"  # vendor
.Lbegin:
.byte 1  # Tag_File
.long .Lend-.Lbegin
.byte 5  # Tag_RISCV_arch
.asciz "rv64i99p0"
.Lend:

#--- merge_version_test_input.s
.section .riscv.attributes,"",@0x70000003
.byte 0x41
.long .Lend-.riscv.attributes-1
.asciz "riscv"  # vendor
.Lbegin:
.byte 1  # Tag_File
.long .Lend-.Lbegin
.byte 5  # Tag_RISCV_arch
.asciz "rv64i2p1"
.Lend:

#--- invalid_arch1.s
.section .riscv.attributes,"",@0x70000003
.byte 0x41
.long .Lend-.riscv.attributes-1
.asciz "riscv"  # vendor
.Lbegin:
.byte 1  # Tag_File
.long .Lend-.Lbegin
.byte 5  # Tag_RISCV_arch
.asciz "rv64i2"
.Lend:

#--- unaligned_access_0.s
.attribute unaligned_access, 0

#--- diff_stack_align.s
.attribute stack_align, 32

#--- diff_priv_spec.s
.attribute priv_spec, 3
.attribute priv_spec_minor, 3

#--- atomic_abi_unknown.s
.attribute atomic_abi, 0

#--- atomic_abi_A6C.s
.attribute atomic_abi, 1

#--- atomic_abi_A6S.s
.attribute atomic_abi, 2

#--- atomic_abi_A7.s
.attribute atomic_abi, 3

#--- atomic_abi_invalid.s
.attribute atomic_abi, 42

#      UNKNOWN_NONE: BuildAttributes {
# UNKNOWN_NONE-NEXT:   FormatVersion: 0x41
# UNKNOWN_NONE-NEXT:   Section 1 {
# UNKNOWN_NONE-NEXT:     SectionLength: 17
# UNKNOWN_NONE-NEXT:     Vendor: riscv
# UNKNOWN_NONE-NEXT:     Tag: Tag_File (0x1)
# UNKNOWN_NONE-NEXT:     Size: 7
# UNKNOWN_NONE-NEXT:     FileAttributes {
# UNKNOWN_NONE-NEXT:       Attribute {
# UNKNOWN_NONE-NEXT:         Tag: 4
# UNKNOWN_NONE-NEXT:         Value: 32
# UNKNOWN_NONE-NEXT:         TagName: stack_align
# UNKNOWN_NONE-NEXT:         Description: Stack alignment is 32-bytes
# UNKNOWN_NONE-NEXT:       }
# UNKNOWN_NONE-NEXT:     }
# UNKNOWN_NONE-NEXT:   }
# UNKNOWN_NONE-NEXT: }

#      NONE_A6C: BuildAttributes {
# NONE_A6C-NEXT:   FormatVersion: 0x41
# NONE_A6C-NEXT:   Section 1 {
# NONE_A6C-NEXT:     SectionLength: 19
# NONE_A6C-NEXT:     Vendor: riscv
# NONE_A6C-NEXT:     Tag: Tag_File (0x1)
# NONE_A6C-NEXT:     Size: 9
# NONE_A6C-NEXT:     FileAttributes {
# NONE_A6C-NEXT:       Attribute {
# NONE_A6C-NEXT:         Tag: 14
# NONE_A6C-NEXT:         Value: 1
# NONE_A6C-NEXT:         TagName: atomic_abi
# NONE_A6C-NEXT:         Description: Atomic ABI is 1
# NONE_A6C-NEXT:       }
# NONE_A6C-NEXT:       Attribute {
# NONE_A6C-NEXT:         Tag: 4
# NONE_A6C-NEXT:         Value: 32
# NONE_A6C-NEXT:         TagName: stack_align
# NONE_A6C-NEXT:         Description: Stack alignment is 32-bytes
# NONE_A6C-NEXT:       }
# NONE_A6C-NEXT:     }
# NONE_A6C-NEXT:   }
# NONE_A6C-NEXT: }

#      UNKNOWN_A6C: BuildAttributes {
# UNKNOWN_A6C-NEXT:   FormatVersion: 0x41
# UNKNOWN_A6C-NEXT:   Section 1 {
# UNKNOWN_A6C-NEXT:     SectionLength: 17
# UNKNOWN_A6C-NEXT:     Vendor: riscv
# UNKNOWN_A6C-NEXT:     Tag: Tag_File (0x1)
# UNKNOWN_A6C-NEXT:     Size: 7
# UNKNOWN_A6C-NEXT:     FileAttributes {
# UNKNOWN_A6C-NEXT:       Attribute {
# UNKNOWN_A6C-NEXT:         Tag: 14
# UNKNOWN_A6C-NEXT:         Value: 1
# UNKNOWN_A6C-NEXT:         TagName: atomic_abi
# UNKNOWN_A6C-NEXT:         Description: Atomic ABI is 1
# UNKNOWN_A6C-NEXT:       }
# UNKNOWN_A6C-NEXT:     }
# UNKNOWN_A6C-NEXT:   }
# UNKNOWN_A6C-NEXT: }

#      UNKNOWN_A6S: BuildAttributes {
# UNKNOWN_A6S-NEXT:   FormatVersion: 0x41
# UNKNOWN_A6S-NEXT:   Section 1 {
# UNKNOWN_A6S-NEXT:     SectionLength:
# UNKNOWN_A6S-NEXT:     Vendor: riscv
# UNKNOWN_A6S-NEXT:     Tag: Tag_File (0x1)
# UNKNOWN_A6S-NEXT:     Size: 7
# UNKNOWN_A6S-NEXT:     FileAttributes {
# UNKNOWN_A6S-NEXT:       Attribute {
# UNKNOWN_A6S-NEXT:         Tag: 14
# UNKNOWN_A6S-NEXT:         Value: 2
# UNKNOWN_A6S-NEXT:         TagName: atomic_abi
# UNKNOWN_A6S-NEXT:         Description: Atomic ABI is 2
# UNKNOWN_A6S-NEXT:       }
# UNKNOWN_A6S-NEXT:     }
# UNKNOWN_A6S-NEXT:   }
# UNKNOWN_A6S-NEXT: }

#      NONE_A7: BuildAttributes {
# NONE_A7-NEXT:   FormatVersion: 0x41
# NONE_A7-NEXT:   Section 1 {
# NONE_A7-NEXT:     SectionLength: 19
# NONE_A7-NEXT:     Vendor: riscv
# NONE_A7-NEXT:     Tag: Tag_File (0x1)
# NONE_A7-NEXT:     Size: 9
# NONE_A7-NEXT:     FileAttributes {
# NONE_A7-NEXT:       Attribute {
# NONE_A7-NEXT:         Tag: 14
# NONE_A7-NEXT:         Value: 3
# NONE_A7-NEXT:         TagName: atomic_abi
# NONE_A7-NEXT:         Description: Atomic ABI is 3
# NONE_A7-NEXT:       }
# NONE_A7-NEXT:       Attribute {
# NONE_A7-NEXT:         Tag: 4
# NONE_A7-NEXT:         Value: 32
# NONE_A7-NEXT:         TagName: stack_align
# NONE_A7-NEXT:         Description: Stack alignment is 32-bytes
# NONE_A7-NEXT:       }
# NONE_A7-NEXT:     }
# NONE_A7-NEXT:   }
# NONE_A7-NEXT: }


#      UNKNOWN_A7: BuildAttributes {
# UNKNOWN_A7-NEXT:   FormatVersion: 0x41
# UNKNOWN_A7-NEXT:   Section 1 {
# UNKNOWN_A7-NEXT:     SectionLength: 17
# UNKNOWN_A7-NEXT:     Vendor: riscv
# UNKNOWN_A7-NEXT:     Tag: Tag_File (0x1)
# UNKNOWN_A7-NEXT:     Size: 7
# UNKNOWN_A7-NEXT:     FileAttributes {
# UNKNOWN_A7-NEXT:       Attribute {
# UNKNOWN_A7-NEXT:         Tag: 14
# UNKNOWN_A7-NEXT:         Value: 3
# UNKNOWN_A7-NEXT:         TagName: atomic_abi
# UNKNOWN_A7-NEXT:         Description: Atomic ABI is 3
# UNKNOWN_A7-NEXT:       }
# UNKNOWN_A7-NEXT:     }
# UNKNOWN_A7-NEXT:   }
# UNKNOWN_A7-NEXT: }

#      A6C_A6S: BuildAttributes {
# A6C_A6S-NEXT:   FormatVersion: 0x41
# A6C_A6S-NEXT:   Section 1 {
# A6C_A6S-NEXT:     SectionLength: 17
# A6C_A6S-NEXT:     Vendor: riscv
# A6C_A6S-NEXT:     Tag: Tag_File (0x1)
# A6C_A6S-NEXT:     Size: 7
# A6C_A6S-NEXT:     FileAttributes {
# A6C_A6S-NEXT:       Attribute {
# A6C_A6S-NEXT:         Tag: 14
# A6C_A6S-NEXT:         Value: 1
# A6C_A6S-NEXT:         TagName: atomic_abi
# A6C_A6S-NEXT:         Description: Atomic ABI is 1
# A6C_A6S-NEXT:       }
# A6C_A6S-NEXT:     }
# A6C_A6S-NEXT:   }
# A6C_A6S-NEXT: }

#      A6S_A7: BuildAttributes {
# A6S_A7-NEXT:   FormatVersion: 0x41
# A6S_A7-NEXT:   Section 1 {
# A6S_A7-NEXT:     SectionLength: 17
# A6S_A7-NEXT:     Vendor: riscv
# A6S_A7-NEXT:     Tag: Tag_File (0x1)
# A6S_A7-NEXT:     Size: 7
# A6S_A7-NEXT:     FileAttributes {
# A6S_A7-NEXT:       Attribute {
# A6S_A7-NEXT:         Tag: 14
# A6S_A7-NEXT:         Value: 3
# A6S_A7-NEXT:         TagName: atomic_abi
# A6S_A7-NEXT:         Description: Atomic ABI is 3
# A6S_A7-NEXT:       }
# A6S_A7-NEXT:     }
# A6S_A7-NEXT:   }
# A6S_A7-NEXT: }

#--- unknown13.s
.attribute 13, "0"
#--- unknown13a.s
.attribute 13, "1"

#--- unknown22.s
.attribute 22, 1
#--- unknown22a.s
.attribute 22, 2
