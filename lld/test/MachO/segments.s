# REQUIRES: x86, shell
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: %lld -o %t %t.o
# RUN: (llvm-readobj --macho-segment %t; echo "Total file size"; wc -c %t) | FileCheck %s

## These two segments must always be present at the start of an executable.
# CHECK-NOT:  Segment {
# CHECK:      Segment {
# CHECK-NEXT:   Cmd: LC_SEGMENT_64
# CHECK-NEXT:   Name: __PAGEZERO
# CHECK-NEXT:   Size: 72
# CHECK-NEXT:   vmaddr: 0x0
# CHECK-NEXT:   vmsize: 0x100000000
# CHECK-NEXT:   fileoff: 0
# CHECK-NEXT:   filesize: 0
## The kernel won't execute a binary with the wrong protections for __PAGEZERO.
# CHECK-NEXT:   maxprot: ---
# CHECK-NEXT:   initprot: ---
# CHECK-NEXT:   nsects: 0
# CHECK-NEXT:   flags: 0x0
# CHECK-NEXT: }
# CHECK-NEXT: Segment {
# CHECK-NEXT:   Cmd: LC_SEGMENT_64
# CHECK-NEXT:   Name: __TEXT
# CHECK-NEXT:   Size: 152
# CHECK-NEXT:   vmaddr: 0x100000000
# CHECK-NEXT:   vmsize:
## dyld3 assumes that the __TEXT segment starts from the file header
# CHECK-NEXT:   fileoff: 0
# CHECK-NEXT:   filesize:
# CHECK-NEXT:   maxprot: rwx
# CHECK-NEXT:   initprot: r-x
# CHECK-NEXT:   nsects: 1
# CHECK-NEXT:   flags: 0x0
# CHECK-NEXT: }

## Check that we handle max-length names correctly.
# CHECK:      Cmd: LC_SEGMENT_64
# CHECK-NEXT: Name: maxlen_16ch_name

## This segment must always be present at the end of an executable, and cover
## its last byte.
# CHECK:      Name: __LINKEDIT
# CHECK-NEXT: Size:
# CHECK-NEXT: vmaddr:
# CHECK-NEXT: vmsize:
# CHECK-NEXT: fileoff: [[#%u, LINKEDIT_OFF:]]
# CHECK-NEXT: filesize: [[#%u, LINKEDIT_SIZE:]]
# CHECK-NEXT: maxprot: rwx
# CHECK-NEXT: initprot: r--
# CHECK-NOT:  Cmd: LC_SEGMENT_64

# CHECK-LABEL: Total file size
# CHECK-NEXT:  [[#%u, LINKEDIT_OFF + LINKEDIT_SIZE]]

.text
.global _main
_main:
  mov $0, %rax
  ret

.section maxlen_16ch_name,foo
