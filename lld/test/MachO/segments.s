# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: lld -flavor darwinnew -o %t %t.o
# RUN: llvm-readobj --macho-segment %t | FileCheck %s

## These two segments must always be present at the start of an executable.
# CHECK-NOT:  Segment {
# CHECK:      Segment {
# CHECK:        Cmd: LC_SEGMENT_64
# CHECK:        Name: __PAGEZERO
# CHECK:        Size: 72
# CHECK:        vmaddr: 0x0
# CHECK:        vmsize: 0x100000000
# CHECK:        fileoff: 0
# CHECK:        filesize: 0
## The kernel won't execute a binary with the wrong protections for __PAGEZERO.
# CHECK:        maxprot: ---
# CHECK:        initprot: ---
# CHECK:        nsects: 0
# CHECK:        flags: 0x0
# CHECK:      }
# CHECK:      Segment {
# CHECK:        Cmd: LC_SEGMENT_64
# CHECK:        Name: __TEXT
# CHECK:        Size: 152
# CHECK:        vmaddr: 0x100000000
# CHECK:        vmsize:
## dyld3 assumes that the __TEXT segment starts from the file header
# CHECK:        fileoff: 0
# CHECK:        filesize:
# CHECK:        maxprot: rwx
# CHECK:        initprot: r-x
# CHECK:        nsects: 1
# CHECK:        flags: 0x0
# CHECK:      }

## Check that we handle max-length names correctly.
# CHECK:      Cmd: LC_SEGMENT_64
# CHECK-NEXT: Name: maxlen_16ch_name

## This segment must always be present at the end of an executable.
# CHECK:      Name: __LINKEDIT
# CHECK:      maxprot: rwx
# CHECK:      initprot: r--
# CHECK-NOT:  Cmd: LC_SEGMENT_64

.text
.global _main
_main:
  mov $0, %rax
  ret

.section maxlen_16ch_name,foo
