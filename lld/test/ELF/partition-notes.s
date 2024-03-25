// Test that notes (both from object files and synthetic) are duplicated into
// each partition.

// REQUIRES: x86

// RUN: llvm-mc %s -o %t.o -filetype=obj --triple=x86_64-unknown-linux
// RUN: ld.lld %t.o -o %t --shared --gc-sections --build-id=sha1

// RUN: llvm-objcopy --extract-main-partition %t %t0
// RUN: llvm-objcopy --extract-partition=part1 %t %t1

// RUN: llvm-readobj --all %t0 | FileCheck %s
// RUN: llvm-readobj --all %t1 | FileCheck %s

// CHECK:        Type: PT_NOTE
// CHECK-NEXT:   Offset: 0x{{0*}}[[NOTE_OFFSET:[^ ]*]]

// CHECK:      Notes [
// CHECK-NEXT:   NoteSection {
// CHECK-NEXT:     Name: .note.obj
// CHECK-NEXT:     Offset: 0x{{0*}}[[NOTE_OFFSET]]
// CHECK-NEXT:     Size:
// CHECK-NEXT:     Note {
// CHECK-NEXT:       Owner: foo
// CHECK-NEXT:       Data size: 0x4
// CHECK-NEXT:       Type: NT_VERSION (version)
// CHECK-NEXT:       Description data (
// CHECK-NEXT:         0000: 62617200                             |bar.|
// CHECK-NEXT:       )
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   NoteSection {
// CHECK-NEXT:     Name: .note.gnu.build-id
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size:
// CHECK-NEXT:     Note {
// CHECK-NEXT:       Owner: GNU
// CHECK-NEXT:       Data size:
// CHECK-NEXT:       Type: NT_GNU_BUILD_ID (unique build ID bitstring)
// CHECK-NEXT:       Build ID: d5101cb9d03b7e836ba9b64f5768a0b31980920f{{$}}
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: ]

.section .llvm_sympart,"",@llvm_sympart
.asciz "part1"
.quad p1

.section .data.p0,"aw",@progbits
.globl p0
p0:

.section .data.p1,"aw",@progbits
.globl p1
p1:

.section .note.obj,"a",@note
.align 4
.long 2f-1f
.long 3f-2f
.long 1
1: .asciz "foo"
2: .asciz "bar"
3:
