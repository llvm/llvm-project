# REQUIRES: x86
	
# RUN: llvm-mc -emit-compact-unwind-non-canonical=true -filetype=obj -triple=x86_64-apple-darwin19.0.0 %s -o %t.o
# RUN: not %lld -lSystem -lc++ %t.o -o %t 2>&1 | FileCheck %s --check-prefix=TOO-MANY
# RUN: not %lld -lSystem %t.o -o %t 2>&1 | FileCheck %s --check-prefix=UNDEF
# TOO-MANY: error: too many personalities (4) for compact unwind to encode
# UNDEF: error: undefined symbol: ___gxx_personality_v0

## Tests that we can handle more than 3 personalities with DWARFs
# RUN: llvm-mc -emit-compact-unwind-non-canonical=false -filetype=obj -triple=x86_64-apple-darwin19.0.0 %s -o %t-dwarf.o
# RUN: %lld -lSystem -lc++ %t-dwarf.o -o %t-dwarf
# RUN: llvm-objdump --macho --indirect-symbols --dwarf=frames %t-dwarf | FileCheck %s --check-prefix=DWARF -D#BASE=0x100000000 

# DWARF: Indirect symbols
# DWARF: address            index name
# DWARF: 0x[[#%x,GXX_PERSONALITY:]]     {{.*}} ___gxx_personality_v0
# DWARF: 0x[[#%x,PERSONALITY_1:]]     {{.*}} _personality_1
# DWARF: 0x[[#%x,PERSONALITY_2:]]     {{.*}} _personality_2
# DWARF: 0x[[#%x,PERSONALITY_3:]]     {{.*}} _personality_3
# DWARF: .eh_frame contents:
# DWARF: Personality Address: [[#%.16x,GXX_PERSONALITY]]
# DWARF: Personality Address: [[#%.16x,PERSONALITY_1]]
# DWARF: Personality Address: [[#%.16x,PERSONALITY_2]]
# DWARF: Personality Address: [[#%.16x,PERSONALITY_3]]


.globl _main, _personality_1, _personality_2, _personality_3

.text

_foo:
  .cfi_startproc
  .cfi_personality 155, _personality_1
  .cfi_def_cfa_offset 16
  retq
  .cfi_endproc

_bar:
  .cfi_startproc
  .cfi_personality 155, _personality_2
  .cfi_def_cfa_offset 16
  retq
  .cfi_endproc

_baz:
  .cfi_startproc
  .cfi_personality 155, _personality_3
  .cfi_def_cfa_offset 16
  retq
  .cfi_endproc

_main:
  .cfi_startproc
  .cfi_personality 155, ___gxx_personality_v0
  .cfi_def_cfa_offset 16
  retq
  .cfi_endproc

_personality_1:
  retq
_personality_2:
  retq
_personality_3:
  retq
