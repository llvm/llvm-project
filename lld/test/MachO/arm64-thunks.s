# REQUIRES: aarch64

## Check for the following:
## (1) address match between thunk definitions and call destinations
## (2) address match between thunk page+offset computations and function
##     definitions
## (3) a second thunk is created when the first one goes out of range
## (4) early calls to a dylib stub use a thunk, and later calls the stub
##     directly
## (5) Thunks are created for all sections in the text segment with branches.
## (6) Thunks are in the linker map file.
## Notes:
## 0x4000000 = 64 Mi = half the magnitude of the forward-branch range

# RUN: rm -rf %t; mkdir %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t/input.o
# RUN: %lld -arch arm64 -dead_strip -lSystem -U _extern_sym -map %t/thunk.map -o %t/thunk %t/input.o
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t/thunk | FileCheck %s

## Check that the thunks appear in the map file and that everything is sorted by address
# Because of the `.space` instructions, there will end up being a lot of dead symbols in the linker map
# so generate a version of the linker map without dead symbols.
# RUN: awk '/# Dead Stripped Symbols:/ {exit} {print}' %t/thunk.map > %t/thunk_no_dead_syms.map

# RUN: FileCheck %s --input-file %t/thunk_no_dead_syms.map --check-prefix=MAP
 
# MAP:      [[ADDR1:0x[0-9A-Fa-f]+]] {{.*}} _b
# MAP-NEXT: [[ADDR2:0x[0-9A-Fa-f]+]] {{.*}} _c
# MAP-NEXT: [[ADDR3:0x[0-9A-Fa-f]+]] {{.*}} _d.thunk.0
# MAP-NEXT: [[ADDR4:0x[0-9A-Fa-f]+]] {{.*}} _e.thunk.0
# MAP-NEXT: [[ADDR5:0x[0-9A-Fa-f]+]] {{.*}} _f.thunk.0
# MAP-NEXT: [[ADDR6:0x[0-9A-Fa-f]+]] {{.*}} _g.thunk.0
# MAP-NEXT: [[ADDR7:0x[0-9A-Fa-f]+]] {{.*}} _h.thunk.0
# MAP-NEXT: [[ADDR8:0x[0-9A-Fa-f]+]] {{.*}} ___nan.thunk.0
# MAP-NEXT: [[ADDR9:0x[0-9A-Fa-f]+]] {{.*}} _d
# MAP-NEXT: [[ADDR10:0x[0-9A-Fa-f]+]] {{.*}} _e
# MAP-NEXT: [[ADDR11:0x[0-9A-Fa-f]+]] {{.*}} _f
# MAP-NEXT: [[ADDR12:0x[0-9A-Fa-f]+]] {{.*}} _g
# MAP-NEXT: [[ADDR13:0x[0-9A-Fa-f]+]] {{.*}} _a.thunk.0
# MAP-NEXT: [[ADDR14:0x[0-9A-Fa-f]+]] {{.*}} _b.thunk.0
# MAP-NEXT: [[ADDR15:0x[0-9A-Fa-f]+]] {{.*}} _h
# MAP-NEXT: [[ADDR16:0x[0-9A-Fa-f]+]] {{.*}} _main
# MAP-NEXT: [[ADDR17:0x[0-9A-Fa-f]+]] {{.*}} _c.thunk.0
# MAP-NEXT: [[ADDR18:0x[0-9A-Fa-f]+]] {{.*}} _d.thunk.1
# MAP-NEXT: [[ADDR19:0x[0-9A-Fa-f]+]] {{.*}} _e.thunk.1
# MAP-NEXT: [[ADDR20:0x[0-9A-Fa-f]+]] {{.*}} _f.thunk.1
# MAP-NEXT: [[ADDR21:0x[0-9A-Fa-f]+]] {{.*}} _z


# CHECK: Disassembly of section __TEXT,__text:

# CHECK: [[#%.13x, A_PAGE:]][[#%.3x, A_OFFSET:]] <_a>:
# CHECK:  bl 0x[[#%x, A:]] <_a>
# CHECK:  bl 0x[[#%x, B:]] <_b>
# CHECK:  bl 0x[[#%x, C:]] <_c>
# CHECK:  bl 0x[[#%x, D_THUNK_0:]] <_d.thunk.0>
# CHECK:  bl 0x[[#%x, E_THUNK_0:]] <_e.thunk.0>
# CHECK:  bl 0x[[#%x, F_THUNK_0:]] <_f.thunk.0>
# CHECK:  bl 0x[[#%x, G_THUNK_0:]] <_g.thunk.0>
# CHECK:  bl 0x[[#%x, H_THUNK_0:]] <_h.thunk.0>
# CHECK:  bl 0x[[#%x, NAN_THUNK_0:]] <___nan.thunk.0>

# CHECK: [[#%.13x, B_PAGE:]][[#%.3x, B_OFFSET:]] <_b>:
# CHECK:  bl 0x[[#%x, A]] <_a>
# CHECK:  bl 0x[[#%x, B]] <_b>
# CHECK:  bl 0x[[#%x, C]] <_c>
# CHECK:  bl 0x[[#%x, D_THUNK_0]] <_d.thunk.0>
# CHECK:  bl 0x[[#%x, E_THUNK_0]] <_e.thunk.0>
# CHECK:  bl 0x[[#%x, F_THUNK_0]] <_f.thunk.0>
# CHECK:  bl 0x[[#%x, G_THUNK_0]] <_g.thunk.0>
# CHECK:  bl 0x[[#%x, H_THUNK_0]] <_h.thunk.0>
# CHECK:  bl 0x[[#%x, NAN_THUNK_0]] <___nan.thunk.0>

# CHECK: [[#%.13x, C_PAGE:]][[#%.3x, C_OFFSET:]] <_c>:
# CHECK:  bl 0x[[#%x, A]] <_a>
# CHECK:  bl 0x[[#%x, B]] <_b>
# CHECK:  bl 0x[[#%x, C]] <_c>
# CHECK:  bl 0x[[#%x, D:]] <_d>
# CHECK:  bl 0x[[#%x, E:]] <_e>
# CHECK:  bl 0x[[#%x, F_THUNK_0]] <_f.thunk.0>
# CHECK:  bl 0x[[#%x, G_THUNK_0]] <_g.thunk.0>
# CHECK:  bl 0x[[#%x, H_THUNK_0]] <_h.thunk.0>
# CHECK:  bl 0x[[#%x, NAN_THUNK_0]] <___nan.thunk.0>

# CHECK: [[#%x, D_THUNK_0]] <_d.thunk.0>:
# CHECK:  adrp x16, 0x[[#%x, D_PAGE:]]
# CHECK:  add  x16, x16, #[[#D_OFFSET:]]

# CHECK: [[#%x, E_THUNK_0]] <_e.thunk.0>:
# CHECK:  adrp x16, 0x[[#%x, E_PAGE:]]
# CHECK:  add  x16, x16, #[[#E_OFFSET:]]

# CHECK: [[#%x, F_THUNK_0]] <_f.thunk.0>:
# CHECK:  adrp x16, 0x[[#%x, F_PAGE:]]
# CHECK:  add  x16, x16, #[[#F_OFFSET:]]

# CHECK: [[#%x, G_THUNK_0]] <_g.thunk.0>:
# CHECK:  adrp x16, 0x[[#%x, G_PAGE:]]
# CHECK:  add  x16, x16, #[[#G_OFFSET:]]

# CHECK: [[#%x, H_THUNK_0]] <_h.thunk.0>:
# CHECK:  adrp x16, 0x[[#%x, H_PAGE:]]
# CHECK:  add  x16, x16, #[[#H_OFFSET:]]

# CHECK: [[#%x, NAN_THUNK_0]] <___nan.thunk.0>:
# CHECK:  adrp x16, 0x[[#%x, NAN_PAGE:]]
# CHECK:  add  x16, x16, #[[#NAN_OFFSET:]]

# CHECK: [[#%x, D_PAGE + D_OFFSET]] <_d>:
# CHECK:  bl 0x[[#%x, A]] <_a>
# CHECK:  bl 0x[[#%x, B]] <_b>
# CHECK:  bl 0x[[#%x, C]] <_c>
# CHECK:  bl 0x[[#%x, D]] <_d>
# CHECK:  bl 0x[[#%x, E]] <_e>
# CHECK:  bl 0x[[#%x, F_THUNK_0]] <_f.thunk.0>
# CHECK:  bl 0x[[#%x, G_THUNK_0]] <_g.thunk.0>
# CHECK:  bl 0x[[#%x, H_THUNK_0]] <_h.thunk.0>
# CHECK:  bl 0x[[#%x, NAN_THUNK_0]] <___nan.thunk.0>

# CHECK: [[#%x, E_PAGE + E_OFFSET]] <_e>:
# CHECK:  bl 0x[[#%x, A_THUNK_0:]] <_a.thunk.0>
# CHECK:  bl 0x[[#%x, B_THUNK_0:]] <_b.thunk.0>
# CHECK:  bl 0x[[#%x, C]] <_c>
# CHECK:  bl 0x[[#%x, D]] <_d>
# CHECK:  bl 0x[[#%x, E]] <_e>
# CHECK:  bl 0x[[#%x, F:]] <_f>
# CHECK:  bl 0x[[#%x, G:]] <_g>
# CHECK:  bl 0x[[#%x, H_THUNK_0]] <_h.thunk.0>
# CHECK:  bl 0x[[#%x, NAN_THUNK_0]] <___nan.thunk.0>

# CHECK: [[#%x, F_PAGE + F_OFFSET]] <_f>:
# CHECK:  bl 0x[[#%x, A_THUNK_0]] <_a.thunk.0>
# CHECK:  bl 0x[[#%x, B_THUNK_0]] <_b.thunk.0>
# CHECK:  bl 0x[[#%x, C]] <_c>
# CHECK:  bl 0x[[#%x, D]] <_d>
# CHECK:  bl 0x[[#%x, E]] <_e>
# CHECK:  bl 0x[[#%x, F]] <_f>
# CHECK:  bl 0x[[#%x, G]] <_g>
# CHECK:  bl 0x[[#%x, H_THUNK_0]] <_h.thunk.0>
# CHECK:  bl 0x[[#%x, NAN_THUNK_0]] <___nan.thunk.0>

# CHECK: [[#%x, G_PAGE + G_OFFSET]] <_g>:
# CHECK:  bl 0x[[#%x, A_THUNK_0]] <_a.thunk.0>
# CHECK:  bl 0x[[#%x, B_THUNK_0]] <_b.thunk.0>
# CHECK:  bl 0x[[#%x, C_THUNK_0:]] <_c.thunk.0>
# CHECK:  bl 0x[[#%x, D_THUNK_1:]] <_d.thunk.1>
# CHECK:  bl 0x[[#%x, E]] <_e>
# CHECK:  bl 0x[[#%x, F]] <_f>
# CHECK:  bl 0x[[#%x, G]] <_g>
# CHECK:  bl 0x[[#%x, H:]] <_h>
# CHECK:  bl 0x[[#%x, STUBS:]]

# CHECK: [[#%x, A_THUNK_0]] <_a.thunk.0>:
# CHECK:  adrp x16, 0x[[#%x, A_PAGE]]000
# CHECK:  add  x16, x16, #[[#%d, A_OFFSET]]

# CHECK: [[#%x, B_THUNK_0]] <_b.thunk.0>:
# CHECK:  adrp x16, 0x[[#%x, B_PAGE]]000
# CHECK:  add  x16, x16, #[[#%d, B_OFFSET]]

# CHECK: [[#%x, H_PAGE + H_OFFSET]] <_h>:
# CHECK:  bl 0x[[#%x, A_THUNK_0]] <_a.thunk.0>
# CHECK:  bl 0x[[#%x, B_THUNK_0]] <_b.thunk.0>
# CHECK:  bl 0x[[#%x, C_THUNK_0]] <_c.thunk.0>
# CHECK:  bl 0x[[#%x, D_THUNK_1]] <_d.thunk.1>
# CHECK:  bl 0x[[#%x, E]] <_e>
# CHECK:  bl 0x[[#%x, F]] <_f>
# CHECK:  bl 0x[[#%x, G]] <_g>
# CHECK:  bl 0x[[#%x, H]] <_h>
# CHECK:  bl 0x[[#%x, STUBS]]

# CHECK: <_main>:
# CHECK:  bl 0x[[#%x, A_THUNK_0]] <_a.thunk.0>
# CHECK:  bl 0x[[#%x, B_THUNK_0]] <_b.thunk.0>
# CHECK:  bl 0x[[#%x, C_THUNK_0]] <_c.thunk.0>
# CHECK:  bl 0x[[#%x, D_THUNK_1]] <_d.thunk.1>
# CHECK:  bl 0x[[#%x, E_THUNK_1:]] <_e.thunk.1>
# CHECK:  bl 0x[[#%x, F_THUNK_1:]] <_f.thunk.1>
# CHECK:  bl 0x[[#%x, G]] <_g>
# CHECK:  bl 0x[[#%x, H]] <_h>
# CHECK:  bl 0x[[#%x, STUBS]]

# CHECK: [[#%x, C_THUNK_0]] <_c.thunk.0>:
# CHECK:  adrp x16, 0x[[#%x, C_PAGE]]000
# CHECK:  add  x16, x16, #[[#%d, C_OFFSET]]

# CHECK: [[#%x, D_THUNK_1]] <_d.thunk.1>:
# CHECK:  adrp x16, 0x[[#%x, D_PAGE]]
# CHECK:  add  x16, x16, #[[#D_OFFSET]]

# CHECK: [[#%x, E_THUNK_1]] <_e.thunk.1>:
# CHECK:  adrp x16, 0x[[#%x, E_PAGE]]
# CHECK:  add  x16, x16, #[[#E_OFFSET]]

# CHECK: [[#%x, F_THUNK_1]] <_f.thunk.1>:
# CHECK:  adrp x16, 0x[[#%x, F_PAGE]]
# CHECK:  add  x16, x16, #[[#F_OFFSET]]

# CHECK: Disassembly of section __TEXT,__lcxx_override:
# CHECK: <_z>:
# CHECK:  bl 0x[[#%x, A_THUNK_0]] <_a.thunk.0>

# CHECK: Disassembly of section __TEXT,__stubs:

# CHECK: [[#%x, NAN_PAGE + NAN_OFFSET]] <__stubs>:

.subsections_via_symbols
.text

.globl _a
.p2align 2
_a:
  bl _a
  bl _b
  bl _c
  bl _d
  bl _e
  bl _f
  bl _g
  bl _h
  bl ___nan
  ret

.globl _b
.p2align 2
_b:
  bl _a
  bl _b
  bl _c
  bl _d
  bl _e
  bl _f
  bl _g
  bl _h
  bl ___nan
  .space 0x4000000-0x3c
  ret

.globl _c
.p2align 2
_c:
  bl _a
  bl _b
  bl _c
  bl _d
  bl _e
  bl _f
  bl _g
  bl _h
  bl ___nan
  ret

.globl _d
.p2align 2
_d:
  bl _a
  bl _b
  bl _c
  bl _d
  bl _e
  bl _f
  bl _g
  bl _h
  bl ___nan
  .space 0x4000000-0x38
  ret

.globl _e
.p2align 2
_e:
  bl _a
  bl _b
  bl _c
  bl _d
  bl _e
  bl _f
  bl _g
  bl _h
  bl ___nan
  ret

.globl _f
.p2align 2
_f:
  bl _a
  bl _b
  bl _c
  bl _d
  bl _e
  bl _f
  bl _g
  bl _h
  bl ___nan
  .space 0x4000000-0x34
  ret

.globl _g
.p2align 2
_g:
  bl _a
  bl _b
  bl _c
  bl _d
  bl _e
  bl _f
  bl _g
  bl _h
  bl ___nan
  ret

.globl _h
.p2align 2
_h:
  bl _a
  bl _b
  bl _c
  bl _d
  bl _e
  bl _f
  bl _g
  bl _h
  bl ___nan
  .space 0x4000000-0x30
  ret

.globl _main
.p2align 2
_main:
  bl _a
  bl _b
  bl _c
  bl _d
  bl _e
  bl _f
  bl _g
  bl _h
  bl ___nan
  ret

.section __TEXT,__cstring
  .space 0x4000000

.section __TEXT,__lcxx_override,regular,pure_instructions

.globl _z
.no_dead_strip _z
.p2align 2
_z:
  bl _a
  ## Ensure calling into stubs works
  bl _extern_sym
  ret
