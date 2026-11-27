# UNSUPPORTED: system-windows
# REQUIRES: aarch64

# RUN: rm -rf %t; split-file %s %t

# Synthesize raw cgdata without the header (32 byte) from the indexed cgdata.
# RUN: llvm-cgdata --convert --format binary %t/raw-1.cgtext -o %t/raw-1.cgdata
# RUN: echo -n "s/<RAW_BYTES>/" > %t/raw-1-sed.txt
# RUN: od -t x1 -j 32 -An %t/raw-1.cgdata | tr -d '\n\r\t' | sed 's/[ ][ ]*/ /g; s/^[ ]*//; s/[ ]*$//; s/[ ]/,0x/g; s/^/0x/' >> %t/raw-1-sed.txt
# RUN: echo "/g" >> %t/raw-1-sed.txt
# RUN: sed -f %t/raw-1-sed.txt %t/merge-template.s > %t/merge-1.s
# RUN: llvm-cgdata --convert --format binary %t/raw-2.cgtext -o %t/raw-2.cgdata
# RUN: echo -n "s/<RAW_BYTES>/" > %t/raw-2-sed.txt
# RUN: od -t x1 -j 32 -An %t/raw-2.cgdata | tr -d '\n\r\t' | sed 's/[ ][ ]*/ /g; s/^[ ]*//; s/[ ]*$//; s/[ ]/,0x/g; s/^/0x/' >> %t/raw-2-sed.txt
# RUN: echo "/g" >> %t/raw-2-sed.txt
# RUN: sed -f %t/raw-2-sed.txt %t/merge-template.s > %t/merge-2.s

# RUN: llvm-mc -filetype obj -triple arm64-apple-darwin %t/merge-1.s -o %t/merge-1.o
# RUN: llvm-mc -filetype obj -triple arm64-apple-darwin %t/merge-2.s -o %t/merge-2.o
# RUN: llvm-mc -filetype obj -triple arm64-apple-darwin %t/main.s -o %t/main.o

# This checks if the codegen data from the linker is identical to the merged codegen data
# from each object file, which is obtained using the llvm-cgdata tool.
# RUN: %no-arg-lld -dylib -arch arm64 -platform_version ios 14.0 15.0 -o %t/out \
# RUN: %t/merge-1.o %t/merge-2.o %t/main.o --codegen-data-generate-path=%t/out-cgdata
# RUN: llvm-cgdata --merge %t/merge-1.o %t/merge-2.o %t/main.o -o %t/merge-cgdata
# RUN: diff %t/out-cgdata %t/merge-cgdata

# Merge order doesn't matter. `main.o` is dropped due to missing __llvm_outline.
# RUN: llvm-cgdata --merge %t/merge-2.o %t/merge-1.o -o %t/merge-cgdata-shuffle
# RUN: diff %t/out-cgdata %t/merge-cgdata-shuffle

# We can also generate the merged codegen data from the executable that is not dead-stripped.
# RUN: llvm-objdump -h %t/out| FileCheck %s
# CHECK: __llvm_outline
# RUN: llvm-cgdata --merge %t/out -o %t/merge-cgdata-exe
# RUN: diff %t/merge-cgdata-exe %t/merge-cgdata

# Dead-strip will remove __llvm_outline sections from the final executable.
# But the codeden data is still correctly produced from the linker.
# RUN: %no-arg-lld -dylib -arch arm64 -platform_version ios 14.0 15.0 -o %t/out-strip \
# RUN: %t/merge-1.o %t/merge-2.o %t/main.o -dead_strip --codegen-data-generate-path=%t/out-cgdata-strip
# RUN: llvm-cgdata --merge %t/merge-1.o %t/merge-2.o %t/main.o -o %t/merge-cgdata-strip
# RUN: diff %t/out-cgdata-strip %t/merge-cgdata-strip
# RUN: diff %t/out-cgdata-strip %t/merge-cgdata

# Ensure no __llvm_outline section remains in the executable.
# RUN: llvm-objdump -h %t/out-strip | FileCheck %s --check-prefix=STRIP
# STRIP-NOT: __llvm_outline

#--- raw-1.cgtext
:outlined_hash_tree
0:
  Hash:            0x0
  Terminals:       0
  SuccessorIds:    [ 1 ]
1:
  Hash:            0x1
  Terminals:       0
  SuccessorIds:    [ 2 ]
2:
  Hash:            0x2
  Terminals:       4
  SuccessorIds:    [  ]
...

#--- raw-2.cgtext
:outlined_hash_tree
0:
  Hash:            0x0
  Terminals:       0
  SuccessorIds:    [ 1 ]
1:
  Hash:            0x1
  Terminals:       0
  SuccessorIds:    [ 2 ]
2:
  Hash:            0x3
  Terminals:       5
  SuccessorIds:    [  ]
...

#--- merge-template.s
.section __DATA,__llvm_outline
_data:
.byte <RAW_BYTES>

#--- main.s
.globl _main

.text
_main:
  ret
