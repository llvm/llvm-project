## Check that llvm-bolt uses 8-byte encodings in .eh_frame_hdr when an offset
## does not fit into 32 bits, and keeps 4-byte encodings otherwise.

# REQUIRES: system-linux

# RUN: split-file %s %t
## Use 8-byte FDE pointers so that the original FDEs can be moved far away from
## their functions.
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown --large-code-model \
# RUN:   %t/main.s -o %t.o
# RUN: ld.lld %t.o -o %t.exe -q --eh-frame-hdr

# RUN: llvm-bolt %t.exe -o %t.bolt | FileCheck %s --check-prefix=CHECK-BOLT4
# RUN: llvm-readelf -u %t.bolt | FileCheck %s --check-prefixes=SDATA4,RELOC

## Place new code more than 2GB above the original functions, which still have
## entries in the header. --use-gnu-stack avoids a multi-gigabyte output file.
# RUN: llvm-bolt %t.exe -o %t.far.bolt --custom-allocation-vma=0x100200000 \
# RUN:   --use-gnu-stack | FileCheck %s --check-prefix=CHECK-BOLT8
# RUN: llvm-readelf -u %t.far.bolt | FileCheck %s --check-prefixes=SDATA8,RELOC

## .eh_frame is more than 2GB above .eh_frame_hdr and .text is not, so only
## eh_frame_ptr and FDE offsets overflow. Without -q, functions are rewritten
## in-place and the new header fits into the original one.
# RUN: ld.lld %t.o -o %t.inplace.exe --eh-frame-hdr -T %t/inplace.lds
# RUN: llvm-bolt %t.inplace.exe -o %t.inplace.bolt --use-gnu-stack \
# RUN:   | FileCheck %s --check-prefixes=CHECK-BOLT,CHECK-BOLT8
# RUN: llvm-readelf -u %t.inplace.bolt \
# RUN:   | FileCheck %s --check-prefixes=SDATA8,INPLACE

# CHECK-BOLT4-NOT: DW_EH_PE_sdata8
# CHECK-BOLT:      rewriting .eh_frame_hdr in-place
# CHECK-BOLT8:     BOLT-INFO: using DW_EH_PE_sdata8 encoding in .eh_frame_hdr

# SDATA4:      eh_frame_ptr_enc: 0x1b
# SDATA4:      table_enc: 0x3b
# SDATA8:      eh_frame_ptr_enc: 0x1c
# SDATA8:      table_enc: 0x3c

# RELOC-NEXT: eh_frame_ptr: [[#%#x,EH_FRAME:]]
# RELOC-NEXT: fde_count: 4
# RELOC:      initial_location: [[#%#x,PC0:]]
# RELOC-NEXT: address: [[#%#x,FDE0:]]
# RELOC:      initial_location: [[#%#x,PC1:]]
# RELOC-NEXT: address: [[#%#x,FDE1:]]
# RELOC:      initial_location: [[#%#x,PC2:]]
# RELOC-NEXT: address: [[#%#x,FDE2:]]
# RELOC:      initial_location: [[#%#x,PC3:]]
# RELOC-NEXT: address: [[#%#x,FDE3:]]
# RELOC:      .eh_frame section at offset {{.*}} address [[#EH_FRAME]]:
# RELOC:      {{\[}}[[#FDE2]]] FDE
# RELOC-NEXT:   initial_location: [[#PC2]]
# RELOC:      {{\[}}[[#FDE3]]] FDE
# RELOC-NEXT:   initial_location: [[#PC3]]
# RELOC:      {{\[}}[[#FDE0]]] FDE
# RELOC-NEXT:   initial_location: [[#PC0]]
# RELOC:      {{\[}}[[#FDE1]]] FDE
# RELOC-NEXT:   initial_location: [[#PC1]]

# INPLACE-NEXT: eh_frame_ptr: [[#%#x,EH_FRAME:]]
# INPLACE-NEXT: fde_count: 2
# INPLACE:      initial_location: [[#%#x,PC0:]]
# INPLACE-NEXT: address: [[#%#x,FDE0:]]
# INPLACE:      initial_location: [[#%#x,PC1:]]
# INPLACE-NEXT: address: [[#%#x,FDE1:]]
# INPLACE:      .eh_frame section at offset {{.*}} address [[#EH_FRAME]]:
# INPLACE:      {{\[}}[[#FDE0]]] FDE
# INPLACE-NEXT:   initial_location: [[#PC0]]
# INPLACE:      {{\[}}[[#FDE1]]] FDE
# INPLACE-NEXT:   initial_location: [[#PC1]]

#--- main.s
  .text
  .globl _start
  .type _start, @function
_start:
  .cfi_startproc
  call foo
  ret
  .cfi_endproc
  .size _start, .-_start

  .globl foo
  .type foo, @function
foo:
  .cfi_startproc
  ret
  .cfi_endproc
  .size foo, .-foo

#--- inplace.lds
SECTIONS {
  . = 0x1000;
  .eh_frame_hdr : {}
  .text 0x40000000 : {}
  .eh_frame 0x80002000 : {}
}
