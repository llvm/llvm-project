# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld --eh-frame-hdr -shared %t.o -o %t.so
# RUN: llvm-dwarfdump --eh-frame %t.so | FileCheck %s

# CHECK:      00000000 00000014 00000000 CIE
# CHECK:        Augmentation:
# CHECK-SAME:                "zR"

# CHECK:      00000018 00000010 0000001c FDE cie=00000000 {{.*}}

# CHECK:      0000002c 00000018 00000000 CIE
# CHECK:        Augmentation:
# CHECK-SAME:                "zPR"
# CHECK:        Personality Address:

# CHECK:      00000048 00000010 00000020 FDE cie=0000002c {{.*}}

# CHECK:      0000005c 00000018 00000000 CIE
# CHECK:        Augmentation:
# CHECK-SAME:                "zPR"
# CHECK:        Personality Address:

# CHECK:      00000078 00000014 00000020 FDE cie=0000005c {{.*}}

foo0:
.cfi_startproc
.cfi_personality 0x9b, personality0
  ret
.cfi_endproc

## This CIE cannot be merged into the previous one because the Personality is different.
foo1:
.cfi_startproc
.cfi_personality 0x9b, personality1
  ret
.cfi_endproc

bar:
.cfi_startproc
  ret
.cfi_endproc

.globl personality0, personality1
.hidden personality0, personality1
personality0:
  ret

personality1:
  ret
