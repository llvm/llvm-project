# REQUIRES: riscv

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=riscv64 a.s -o a.o
# RUN: ld.lld -e 0 a.o -o out 2>&1 | count 0
# RUN: llvm-readobj --arch-specific out | FileCheck %s
# RUN: ld.lld -e 0 a.o a.o -o out1 2>&1 | count 0
# RUN: llvm-readobj --arch-specific out1 | FileCheck %s
# RUN: ld.lld -r a.o a.o -o out1 2>&1 | count 0
# RUN: llvm-readobj --arch-specific out1 | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=riscv64 b.s -o b.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 c.s -o c.o
# RUN: ld.lld a.o b.o c.o -o out2
# RUN: llvm-readobj --arch-specific out2 | FileCheck %s --check-prefix=CHECK2

# RUN: llvm-mc -filetype=obj -triple=riscv64 unrecognized_ext1.s -o unrecognized_ext1.o
# RUN: not ld.lld unrecognized_ext1.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNRECOGNIZED_EXT1 --implicit-check-not=error:
# UNRECOGNIZED_EXT1: error: unrecognized_ext1.o:(.riscv.attributes): rv64i2p0_y2p0: invalid standard user-level extension 'y'

# RUN: llvm-mc -filetype=obj -triple=riscv64 unrecognized_ext2.s -o unrecognized_ext2.o
# RUN: not ld.lld unrecognized_ext2.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNRECOGNIZED_EXT2 --implicit-check-not=error:
# UNRECOGNIZED_EXT2: error: unrecognized_ext2.o:(.riscv.attributes): rv64i2p0_zmadeup1p0: unsupported version number 1.0 for extension 'zmadeup'

# RUN: llvm-mc -filetype=obj -triple=riscv64 unrecognized_version.s -o unrecognized_version.o
# RUN: not ld.lld unrecognized_version.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNRECOGNIZED_VERSION --implicit-check-not=error:
# UNRECOGNIZED_VERSION: error: unrecognized_version.o:(.riscv.attributes): rv64i99p0: unsupported version number 99.0 for extension 'i'

# RUN: llvm-mc -filetype=obj -triple=riscv64 invalid_arch1.s -o invalid_arch1.o
# RUN: ld.lld -e 0 invalid_arch1.o -o invalid_arch1
# RUN: llvm-readobj --arch-specific invalid_arch1 | FileCheck %s --check-prefix=INVALID_ARCH1

## A zero value attribute is not printed.
# RUN: llvm-mc -filetype=obj -triple=riscv64 unaligned_access_0.s -o unaligned_access_0.o
# RUN: ld.lld -e 0 --fatal-warnings a.o unaligned_access_0.o -o unaligned_access_0
# RUN: llvm-readobj -A unaligned_access_0 | FileCheck /dev/null --implicit-check-not='TagName: unaligned_access'

## Differing stack_align values lead to an error.
# RUN: llvm-mc -filetype=obj -triple=riscv64 diff_stack_align.s -o diff_stack_align.o
# RUN: not ld.lld a.o b.o c.o diff_stack_align.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=STACK_ALIGN --implicit-check-not=error:
# STACK_ALIGN: error: diff_stack_align.o:(.riscv.attributes) has stack_align=32 but a.o:(.riscv.attributes) has stack_align=16

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

# CHECK2:      BuildAttributes {
# CHECK2-NEXT:   FormatVersion: 0x41
# CHECK2-NEXT:   Section 1 {
# CHECK2-NEXT:     SectionLength: 95
# CHECK2-NEXT:     Vendor: riscv
# CHECK2-NEXT:     Tag: Tag_File (0x1)
# CHECK2-NEXT:     Size: 85
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
# CHECK2-NEXT:         Value: rv64i2p0_m2p0_a2p0_f2p0_d2p0_c2p0_zkt1p0_zve32f1p0_zve32x1p0_zvl32b1p0
# CHECK2-NEXT:       }
# CHECK2-NEXT:     }
# CHECK2-NEXT:   }
# CHECK2-NEXT: }

#--- a.s
.attribute stack_align, 16
.attribute arch, "rv64i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"
.attribute unaligned_access, 0

#--- b.s
.attribute stack_align, 16
.attribute arch, "rv64i2p0_m2p0_a2p0_f2p0_d2p0_c2p0"
.attribute priv_spec, 2
.attribute priv_spec_minor, 2

#--- c.s
.attribute stack_align, 16
.attribute arch, "rv64i2p0_f2p0_zkt1p0_zve32f1p0_zve32x1p0_zvl32b1p0"
.attribute unaligned_access, 1
.attribute priv_spec, 2
.attribute priv_spec_minor, 2

#--- unrecognized_ext1.s
.section .riscv.attributes,"",@0x70000003
.byte 0x41
.long .Lend-.riscv.attributes-1
.asciz "riscv"  # vendor
.Lbegin:
.byte 1  # Tag_File
.long .Lend-.Lbegin
.byte 5  # Tag_RISCV_arch
.asciz "rv64i2p0_y2p0"
.Lend:

#--- unrecognized_ext2.s
.section .riscv.attributes,"",@0x70000003
.byte 0x41
.long .Lend-.riscv.attributes-1
.asciz "riscv"  # vendor
.Lbegin:
.byte 1  # Tag_File
.long .Lend-.Lbegin
.byte 5  # Tag_RISCV_arch
.asciz "rv64i2p0_zmadeup1p0"
.Lend:

#--- unrecognized_version.s
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

#--- invalid_arch1.s
# INVALID_ARCH1:      BuildAttributes {
# INVALID_ARCH1-NEXT:   FormatVersion: 0x41
# INVALID_ARCH1-NEXT:   Section 1 {
# INVALID_ARCH1-NEXT:     SectionLength: 25
# INVALID_ARCH1-NEXT:     Vendor: riscv
# INVALID_ARCH1-NEXT:     Tag: Tag_File (0x1)
# INVALID_ARCH1-NEXT:     Size: 15
# INVALID_ARCH1-NEXT:     FileAttributes {
# INVALID_ARCH1-NEXT:       Attribute {
# INVALID_ARCH1-NEXT:         Tag: 5
# INVALID_ARCH1-NEXT:         TagName: arch
# INVALID_ARCH1-NEXT:         Value: rv64i2p0
# INVALID_ARCH1-NEXT:       }
# INVALID_ARCH1-NEXT:     }
# INVALID_ARCH1-NEXT:   }
# INVALID_ARCH1-NEXT: }
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

#--- unknown13.s
.attribute 13, "0"
#--- unknown13a.s
.attribute 13, "1"

#--- unknown22.s
.attribute 22, 1
#--- unknown22a.s
.attribute 22, 2
