## Test the different ways of handling entry point for instrumentation.
## Bolt is hooking its runtime function via Elf entry, DT_INIT or DT_INIT_ARRAYS.
## Bolt uses Elf e_entry address for ELF executable, and DT_INIT address
## for ELF shared object to determine the start address.
## The Test is checking the following cases:
## - For executable, check ELF e_entry is pathced.
## - For shared object:
##   - Bolt use DT_INIT for hooking runtime start function if that exists.
##   - If it doesn't exists, DT_INIT_ARRAY takes its place.
# REQUIRES: system-linux,bolt-runtime,target=aarch64{{.*}}

## Check e_entry address is updated with ELF PIE executable.
# RUN: %clang %cflags -pie %s -Wl,-q -o %t.exe
# RUN: llvm-readelf -l %t.exe | FileCheck --check-prefix=CHECK-INTERP %s
# RUN: llvm-readelf -r %t.exe | FileCheck --check-prefix=RELOC-PIE %s
# RUN: llvm-readelf -hs %t.exe | FileCheck --check-prefix=CHECK-START %s
# RUN: llvm-bolt %t.exe -o %t --instrument
# RUN: llvm-readelf -dhs %t | FileCheck --check-prefix=CHECK-ENTRY %s

## Create a shared library to use DT_INIT for the instrumentation.
# RUN: %clang %cflags -fPIC -shared %s -Wl,-q -o %t-init.so
# RUN: llvm-bolt %t-init.so -o %t-init --instrument
# RUN: llvm-readelf -drs %t-init | FileCheck --check-prefix=CHECK-INIT %s

# Create a shared library with no init to use DT_INIT_ARRAY for the instrumentation.
# RUN: %clang %cflags -shared %s -Wl,-q,-init=0 -o %t-no-init.so
# RUN: llvm-bolt %t-no-init.so -o %t-no-init --instrument
# RUN: llvm-readelf -drs %t-no-init | FileCheck --check-prefix=CHECK-NO-INIT %s

## Check the binary has InterP header
# CHECK-INTERP: Program Headers:
# CHECK-INTERP: INTERP

## With PIE: binary should have relative relocations
# RELOC-PIE: R_AARCH64_RELATIVE

## ELF excecutable where e_entry is set to __bolt_runtime_start (PIE).
## Check the input that e_entry points to _start by default.
# CHECK-START: ELF Header:
# CHECK-START-DAG:   Entry point address: 0x[[ENTRY:[[:xdigit:]]+]]
# CHECK-START: Symbol table '.symtab' contains {{.*}} entries:
# CHECK-START-DAG: {{0+}}[[ENTRY]] {{.*}} _start
## Check that e_entry is set to __bolt_runtime_start after the instrumentation.
# CHECK-ENTRY: ELF Header:
# CHECK-ENTRY-DAG:   Entry point address: 0x[[ENTRY:[[:xdigit:]]+]]
# CHECK-ENTRY: Symbol table '.symtab' contains {{.*}} entries:
# CHECK-ENTRY-DAG: {{0+}}[[ENTRY]] {{.*}} __bolt_runtime_start

## Check that DT_INIT is set to __bolt_runtime_start.
# CHECK-INIT:     Dynamic section at offset {{.*}} contains {{.*}} entries:
# CHECK-INIT-DAG: (INIT) 0x[[INIT:[[:xdigit:]]+]]
# CHECK-INIT-DAG: (INIT_ARRAY) 0x[[INIT_ARRAY:[[:xdigit:]]+]]
## Check that the dynamic relocation at .init_array was not patched
# CHECK-INIT:     Relocation section '.rela.dyn' at offset {{.*}} contains {{.*}} entries
# CHECK-INIT:     {{0+}}[[INIT_ARRAY]] {{.*}} R_AARCH64_RELATIVE [[MYINIT_ADDR:[[:xdigit:]]+]
]
# CHECK-INIT:     Symbol table '.symtab' contains {{.*}} entries:
# CHECK-INIT-DAG: {{0+}}[[MYINIT_ADDR]] {{.*}} _myinit

## Check that DT_INIT_ARRAY is set to __bolt_runtime_start.
# CHECK-NO-INIT:     Dynamic section at offset {{.*}} contains {{.*}} entries:
# CHECK-NO-INIT-NOT: (INIT)
# CHECK-NO-INIT:     (INIT_ARRAY) 0x[[INIT_ARRAY:[a-f0-9]+]]
# CHECK-NO-INIT:     Relocation section '.rela.dyn' at offset {{.*}} contains {{.*}} entries
# CHECK-NO-INIT:     {{0+}}[[INIT_ARRAY]] {{.*}} R_AARCH64_RELATIVE [[INIT_ADDR:[[:xdigit:]]+]]
# CHECK-NO-INIT:     Symbol table '.symtab' contains {{.*}} entries:
# CHECK-NO-INIT-DAG: {{0+}}[[INIT_ADDR]] {{.*}} __bolt_runtime_start

  .globl _start
  .type _start, %function
_start:
  # Dummy relocation to force relocation mode.
  .reloc 0, R_AARCH64_NONE
  ret
.size _start, .-_start

  .globl _init
  .type _init, %function
_init:
  ret
  .size _init, .-_init

  .globl _fini
  .type _fini, %function
_fini:
  ret
  .size _fini, .-_fini

  .section .text
_myinit:
  ret
  .size _myinit, .-_myinit

  .section .init_array,"aw"
  .align 3
  .dword _myinit # For relative relocation
