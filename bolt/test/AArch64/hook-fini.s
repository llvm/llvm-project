## Test the different ways of hooking the fini function for instrumentation (via
## DT_FINI and via DT_FINI_ARRAY). We test the latter for both PIE and non-PIE
## binaries because of the different ways of handling relocations (static or
## dynamic).
## All tests perform the following steps:
## - Compile and link for the case to be tested
## - Some sanity-checks on the dynamic section and relocations in the binary to
##   verify it has the shape we want for testing:
##   - DT_FINI or DT_FINI_ARRAY in dynamic section
##   - No relative relocations for non-PIE
## - Instrument
## - Verify generated binary
# REQUIRES: system-linux,bolt-runtime,target=aarch64{{.*}}

# RUN: %clang %cflags -pie %s -Wl,-q -o %t.exe
# RUN: llvm-readelf -d %t.exe | FileCheck --check-prefix=DYN-FINI %s
# RUN: llvm-readelf -r %t.exe | FileCheck --check-prefix=RELOC-PIE %s
# RUN: llvm-bolt %t.exe -o %t --instrument
# RUN: llvm-readelf -drs %t | FileCheck --check-prefix=CHECK-FINI %s

# RUN: %clang %cflags -pie %s -Wl,-q,-fini=0 -o %t-no-fini.exe
# RUN: llvm-readelf -d %t-no-fini.exe | FileCheck --check-prefix=DYN-NO-FINI %s
# RUN: llvm-readelf -r %t-no-fini.exe | FileCheck --check-prefix=RELOC-PIE %s
# RUN: llvm-bolt %t-no-fini.exe -o %t-no-fini --instrument
# RUN: llvm-readelf -drs %t-no-fini | FileCheck --check-prefix=CHECK-NO-FINI %s
# RUN: llvm-readelf -ds -x .fini_array %t-no-fini | FileCheck --check-prefix=CHECK-NO-FINI-RELOC %s

## Create a dummy shared library to link against to force creation of the dynamic section.
# RUN: %clang %cflags %p/../Inputs/stub.c -fPIC -shared -o %t-stub.so
# RUN: %clang %cflags %s -no-pie -Wl,-q,-fini=0 %t-stub.so -o %t-no-pie-no-fini.exe
# RUN: llvm-readelf -r %t-no-pie-no-fini.exe | FileCheck --check-prefix=RELOC-NO-PIE %s
# RUN: llvm-bolt %t-no-pie-no-fini.exe -o %t-no-pie-no-fini --instrument
# RUN: llvm-readelf -ds -x .fini_array %t-no-pie-no-fini | FileCheck --check-prefix=CHECK-NO-PIE-NO-FINI %s

## With fini: dynamic section should contain DT_FINI
# DYN-FINI: (FINI)

## Without fini: dynamic section should only contain DT_FINI_ARRAY
# DYN-NO-FINI-NOT: (FINI)
# DYN-NO-FINI:     (FINI_ARRAY)
# DYN-NO-FINI:     (FINI_ARRAYSZ)

## With PIE: binary should have relative relocations
# RELOC-PIE: R_AARCH64_RELATIVE

## Without PIE: binary should not have relative relocations
# RELOC-NO-PIE-NOT: R_AARCH64_RELATIVE

## Check that DT_FINI is set to __bolt_runtime_fini
# CHECK-FINI:     Dynamic section at offset {{.*}} contains {{.*}} entries:
# CHECK-FINI-DAG: (FINI) 0x[[FINI:[[:xdigit:]]+]]
# CHECK-FINI-DAG: (FINI_ARRAY) 0x[[FINI_ARRAY:[[:xdigit:]]+]]
## Check that the dynamic relocation at .fini_array was not patched
# CHECK-FINI:     Relocation section '.rela.dyn' at offset {{.*}} contains {{.*}} entries
# CHECK-FINI-NOT: {{0+}}[[FINI_ARRAY]] {{.*}} R_AARCH64_RELATIVE [[FINI]]
# CHECK-FINI:     Symbol table '.symtab' contains {{.*}} entries:
# CHECK-FINI:     {{0+}}[[FINI]] {{.*}} __bolt_runtime_fini

## Check that DT_FINI_ARRAY has a dynamic relocation for __bolt_runtime_fini
# CHECK-NO-FINI:     Dynamic section at offset {{.*}} contains {{.*}} entries:
# CHECK-NO-FINI-NOT: (FINI)
# CHECK-NO-FINI:     (FINI_ARRAY) 0x[[FINI_ARRAY:[[:xdigit:]]+]]
# CHECK-NO-FINI:     Relocation section '.rela.dyn' at offset {{.*}} contains {{.*}} entries
# CHECK-NO-FINI:     {{0+}}[[FINI_ARRAY]] {{.*}} R_AARCH64_RELATIVE [[FINI_ADDR:[[:xdigit:]]+]]
# CHECK-NO-FINI:     Symbol table '.symtab' contains {{.*}} entries:
# CHECK-NO-FINI:     {{0+}}[[FINI_ADDR]] {{.*}} __bolt_runtime_fini

## Check that the static relocation in .fini_array is patched even for PIE
# CHECK-NO-FINI-RELOC: Dynamic section at offset {{.*}} contains {{.*}} entries:
# CHECK-NO-FINI-RELOC: (FINI_ARRAY) 0x[[FINI_ARRAY:[[:xdigit:]]+]]
# CHECK-NO-FINI-RELOC: Symbol table '.symtab' contains {{.*}} entries:
## Read  bytes separately so we can reverse them later
# CHECK-NO-FINI-RELOC: {{0+}}[[FINI_ADDR_B0:[[:xdigit:]]{2}]][[FINI_ADDR_B1:[[:xdigit:]]{2}]][[FINI_ADDR_B2:[[:xdigit:]]{2}]][[FINI_ADDR_B3:[[:xdigit:]]{2}]] {{.*}} __bolt_runtime_fini
# CHECK-NO-FINI-RELOC: Hex dump of section '.fini_array':
# CHECK-NO-FINI-RELOC: 0x{{0+}}[[FINI_ARRAY]] [[FINI_ADDR_B3]][[FINI_ADDR_B2]][[FINI_ADDR_B1]][[FINI_ADDR_B0]] 00000000

## Check that DT_FINI_ARRAY has static relocation applied for __bolt_runtime_fini
# CHECK-NO-PIE-NO-FINI:     Dynamic section at offset {{.*}} contains {{.*}} entries:
# CHECK-NO-PIE-NO-FINI-NOT: (FINI)
# CHECK-NO-PIE-NO-FINI:     (FINI_ARRAY) 0x[[FINI_ARRAY:[a-f0-9]+]]
# CHECK-NO-PIE-NO-FINI:     Symbol table '.symtab' contains {{.*}} entries:
## Read address bytes separately so we can reverse them later
# CHECK-NO-PIE-NO-FINI:     {{0+}}[[FINI_ADDR_B0:[[:xdigit:]]{2}]][[FINI_ADDR_B1:[[:xdigit:]]{2}]][[FINI_ADDR_B2:[[:xdigit:]]{2}]][[FINI_ADDR_B3:[[:xdigit:]]{2}]] {{.*}} __bolt_runtime_fini
# CHECK-NO-PIE-NO-FINI:     Hex dump of section '.fini_array':
# CHECK-NO-PIE-NO-FINI:     0x{{0+}}[[FINI_ARRAY]] [[FINI_ADDR_B3]][[FINI_ADDR_B2]][[FINI_ADDR_B1]][[FINI_ADDR_B0]] 00000000

  .globl _start
  .type _start, %function
_start:
  # Dummy relocation to force relocation mode.
  .reloc 0, R_AARCH64_NONE
  ret
.size _start, .-_start

  .globl _fini
  .type _fini, %function
_fini:
  ret
  .size _fini, .-_fini

  .section .fini_array,"aw"
  .align 3
  .dword _fini
