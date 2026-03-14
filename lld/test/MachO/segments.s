# REQUIRES: x86, aarch64
# RUN: rm -rf %t; mkdir -p %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t/x86_64.o
# RUN: llvm-mc -filetype=obj -triple=arm64_32-apple-watchos %s -o %t/arm64-32.o
# RUN: %lld -o %t/x86_64 %t/x86_64.o
# RUN: %lld-watchos -o %t/arm64_32 %t/arm64-32.o

# RUN: llvm-readobj --macho-segment %t/x86_64 > %t/x86_64.out
# RUN: echo "Total file size" >> %t/x86_64.out
# RUN: wc -c %t/x86_64 >> %t/x86_64.out
# RUN: FileCheck %s -DSUFFIX=_64 -DPAGEZERO_SIZE=0x100000000 -DTEXT_ADDR=0x100000000 < %t/x86_64.out

# RUN: llvm-readobj --macho-segment %t/arm64_32 > %t/arm64-32.out
# RUN: echo "Total file size" >> %t/arm64-32.out
# RUN: wc -c %t/arm64_32 >> %t/arm64-32.out
# RUN: FileCheck %s -DSUFFIX= -DPAGEZERO_SIZE=0x4000 -DTEXT_ADDR=0x4000 < %t/arm64-32.out

## These two segments must always be present at the start of an executable.
# CHECK-NOT:  Segment {
# CHECK:      Segment {
# CHECK-NEXT:   Cmd: LC_SEGMENT[[SUFFIX]]{{$}}
# CHECK-NEXT:   Name: __PAGEZERO
# CHECK-NEXT:   Size:
# CHECK-NEXT:   vmaddr: 0x0
# CHECK-NEXT:   vmsize: [[PAGEZERO_SIZE]]
# CHECK-NEXT:   fileoff: 0
# CHECK-NEXT:   filesize: 0
## The kernel won't execute a binary with the wrong protections for __PAGEZERO.
# CHECK-NEXT:   maxprot: ---
# CHECK-NEXT:   initprot: ---
# CHECK-NEXT:   nsects: 0
# CHECK-NEXT:   flags: 0x0
# CHECK-NEXT: }
# CHECK-NEXT: Segment {
# CHECK-NEXT:   Cmd: LC_SEGMENT[[SUFFIX]]{{$}}
# CHECK-NEXT:   Name: __TEXT
# CHECK-NEXT:   Size:
# CHECK-NEXT:   vmaddr: [[TEXT_ADDR]]
# CHECK-NEXT:   vmsize:
## dyld3 assumes that the __TEXT segment starts from the file header
# CHECK-NEXT:   fileoff: 0
# CHECK-NEXT:   filesize:
# CHECK-NEXT:   maxprot: r-x
# CHECK-NEXT:   initprot: r-x
# CHECK-NEXT:   nsects: 1
# CHECK-NEXT:   flags: 0x0
# CHECK-NEXT: }

## Check that we handle max-length names correctly.
# CHECK:      Cmd: LC_SEGMENT[[SUFFIX]]{{$}}
# CHECK-NEXT: Name: maxlen_16ch_name

## This segment must always be present at the end of an executable, and cover
## its last byte.
# CHECK:      Name: __LINKEDIT
# CHECK-NEXT: Size:
# CHECK-NEXT: vmaddr:
# CHECK-NEXT: vmsize:
# CHECK-NEXT: fileoff: [[#%u, LINKEDIT_OFF:]]
# CHECK-NEXT: filesize: [[#%u, LINKEDIT_SIZE:]]
# CHECK-NEXT: maxprot: r--
# CHECK-NEXT: initprot: r--
# CHECK-NOT:  Cmd: LC_SEGMENT[[SUFFIX]]{{$}}

# CHECK-LABEL: Total file size
# CHECK-NEXT:  [[#%u, LINKEDIT_OFF + LINKEDIT_SIZE]]

.text
.global _main
_main:
  ret

.section maxlen_16ch_name,foo
