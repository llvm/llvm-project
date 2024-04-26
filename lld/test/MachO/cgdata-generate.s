# REQUIRES: aarch64

# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype obj -triple arm64-apple-darwin %t/merge-1.s -o %t/merge-1.o
# RUN: llvm-mc -filetype obj -triple arm64-apple-darwin %t/merge-2.s -o %t/merge-2.o
# RUN: llvm-mc -filetype obj -triple arm64-apple-darwin %t/main.s -o %t/main.o

# This checks if the codegen data from the linker is identical to the merged codegen data
# from each object file, which is obtained using the llvm-cgdata tool.
# RUN: %no-arg-lld -dylib -arch arm64 -platform_version ios 14.0 15.0 -o %t/out \
# RUN: %t/merge-1.o %t/merge-2.o %t/main.o --codegen-data-generate-path=%t/out-cgdata
# RUN: llvm-cgdata merge %t/merge-1.o %t/merge-2.o %t/main.o -o %t/merge-cgdata
# RUN: diff %t/out-cgdata %t/merge-cgdata

# Merge order doesn't matter. `main.o` is dropped due to missing __llvm_outline.
# RUN: llvm-cgdata merge %t/merge-2.o %t/merge-1.o -o %t/merge-cgdata-shuffle
# RUN: diff %t/out-cgdata %t/merge-cgdata-shuffle

# We can also generate the merged codegen data from the executable that is not dead-stripped.
# RUN: llvm-objdump -h %t/out| FileCheck %s
CHECK: __llvm_outline
# RUN: llvm-cgdata merge %t/out -o %t/merge-cgdata-exe
# RUN: diff %t/merge-cgdata-exe %t/merge-cgdata

# Dead-strip will remove __llvm_outline sections from the final executable.
# But the codeden data is still correctly produced from the linker.
# RUN: %no-arg-lld -dylib -arch arm64 -platform_version ios 14.0 15.0 -o %t/out-strip \
# RUN: %t/merge-1.o %t/merge-2.o %t/main.o -dead_strip --codegen-data-generate-path=%t/out-cgdata-strip
# RUN: llvm-cgdata merge %t/merge-1.o %t/merge-2.o %t/main.o -o %t/merge-cgdata-strip
# RUN: diff %t/out-cgdata-strip %t/merge-cgdata-strip
# RUN: diff %t/out-cgdata-strip %t/merge-cgdata

# Ensure no __llvm_outline section remains in the executable.
# RUN: llvm-objdump -h %t/out-strip | FileCheck %s --check-prefix=STRIP
STRIP-NOT: __llvm_outline

#--- merge-1.s
# The .data is encoded in a binary form based on the following yaml form. See serialize() in OutlinedHashTreeRecord.cpp
#---
#0:
#  Hash:            0x0
#  Terminals:       0
#  SuccessorIds:    [ 1 ]
#1:
#  Hash:            0x1
#  Terminals:       0
#  SuccessorIds:    [ 2 ]
#2:
#  Hash:            0x2
#  Terminals:       4
#  SuccessorIds:    [  ]
#...
.section __DATA,__llvm_outline
_data:
.byte 0x03,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x02,0x00,0x00,0x00,0x02,0x00,0x00,0x00,0x02,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x04,0x00,0x00,0x00,0x00,0x00,0x00,0x00

#--- merge-2.s
# The .data is encoded in a binary form based on the following yaml form. See serialize() in OutlinedHashTreeRecord.cpp
#---
#0:
#  Hash:            0x0
#  Terminals:       0
#  SuccessorIds:    [ 1 ]
#1:
#  Hash:            0x1
#  Terminals:       0
#  SuccessorIds:    [ 2 ]
#2:
#  Hash:            0x3
#  Terminals:       5
#  SuccessorIds:    [  ]
#...
.section __DATA,__llvm_outline
_data:
.byte 0x03,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x02,0x00,0x00,0x00,0x02,0x00,0x00,0x00,0x03,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x05,0x00,0x00,0x00,0x00,0x00,0x00,0x00

#--- main.s
.globl _main

.text
_main:
  ret
