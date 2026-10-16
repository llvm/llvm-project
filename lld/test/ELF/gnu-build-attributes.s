# REQUIRES: x86

## Check that .gnu.build.attributes.* sections are concatenated into a single
## .gnu.build.attributes section.

# RUN: llvm-mc -filetype=obj -triple=x86_64-linux-gnu %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readobj -n %t | FileCheck %s

# CHECK:      NoteSections [
# CHECK-NEXT:   NoteSection {
# CHECK-NEXT:     Name: .gnu.build.attributes
# CHECK-NEXT:     Offset: 0x120
# CHECK-NEXT:     Size: 0x28
# CHECK-NEXT:     Notes [
# CHECK-NEXT:       {
# CHECK-NEXT:         Owner: GA${{.*}}:a1
# CHECK-NEXT:         Data size: 0x0
# CHECK-NEXT:         Type: OPEN
# CHECK-NEXT:       }
# CHECK-NEXT:       {
# CHECK-NEXT:         Owner: GA${{.*}}:b1
# CHECK-NEXT:         Data size: 0x0
# CHECK-NEXT:         Type: OPEN
# CHECK-NEXT:       }
# CHECK-NEXT:     ]
# CHECK-NEXT:   }
# CHECK-NEXT: ]

.section ".gnu.build.attributes.text.foo", "", @note
.balign 4
.long	8
.long	0
.long	0x100
.asciz	"GA$\x03:a1"

.section ".gnu.build.attributes.text.bar", "", @note
.balign 4
.long	8
.long	0
.long	0x100
.asciz	"GA$\x03:b1"
