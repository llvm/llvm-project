## Test the different ways of hooking the init function for instrumentation (via
## entry point, DT_INIT and via DT_INIT_ARRAY). We test the latter for both PIE
## and non-PIE binaries because of the different ways of handling relocations
## (static or dynamic), executable and shared library.
## All tests perform the following steps:
## - Compile and link for the case to be tested
## - Some sanity-checks on the dynamic section and relocations in the binary to
##   verify it has the shape we want for testing:
##   - INTERP in Program Headers
##   - DT_INIT or DT_INIT_ARRAY in dynamic section
##   - No relative relocations for non-PIE
## - Instrument (with extra --runtime-lib-init-hook=init/init_array options
##   in some cases)
## - Verify generated binary
# REQUIRES: system-linux,bolt-runtime,target=aarch64{{.*}}

# RUN: %clang %cflags -pie %s -Wl,-q -o %t.exe
# RUN: llvm-readelf -d %t.exe | FileCheck --check-prefix=DYN-INIT %s
# RUN: llvm-readelf -l %t.exe | FileCheck --check-prefix=PH-INTERP %s
# RUN: llvm-readelf -r %t.exe | FileCheck --check-prefix=RELOC-PIE %s
# RUN: llvm-bolt %t.exe -o %t --instrument | FileCheck --check-prefix=CHECK-BOLT-RT-EP %s
# RUN: llvm-readelf -hdrs %t | FileCheck --check-prefix=CHECK-INIT-EP %s
# RUN: llvm-bolt %t.exe -o %t-no-ep --instrument --runtime-lib-init-hook=init | FileCheck --check-prefix=CHECK-BOLT-RT-INIT %s
# RUN: llvm-readelf -hdrs %t-no-ep | FileCheck --check-prefix=CHECK-INIT-NO-EP %s
# RUN: llvm-bolt %t.exe -o %t-no-ep --instrument --runtime-lib-init-hook=init_array | FileCheck --check-prefix=CHECK-BOLT-RT-INIT-ARRAY %s
# RUN: llvm-readelf -hdrs %t-no-ep | FileCheck --check-prefix=CHECK-INIT-ARRAY-NO-EP %s

# RUN: %clang -shared %cflags -pie %s -Wl,-q -o %t-shared.exe
# RUN: llvm-readelf -d %t-shared.exe | FileCheck --check-prefix=DYN-INIT %s
# RUN: llvm-readelf -l %t-shared.exe | FileCheck --check-prefix=PH-INTERP-SHARED %s
# RUN: llvm-readelf -r %t-shared.exe | FileCheck --check-prefix=RELOC-SHARED-PIE %s
# RUN: llvm-bolt %t-shared.exe -o %t-shared --instrument | FileCheck --check-prefix=CHECK-BOLT-RT-INIT %s
# RUN: llvm-readelf -hdrs %t-shared | FileCheck --check-prefix=CHECK-SHARED-INIT %s

# RUN: %clang %cflags -pie %s -Wl,-q,-init=0 -o %t-no-init.exe
# RUN: llvm-readelf -d %t-no-init.exe | FileCheck --check-prefix=DYN-NO-INIT %s
# RUN: llvm-readelf -l %t-no-init.exe | FileCheck --check-prefix=PH-INTERP %s
# RUN: llvm-readelf -r %t-no-init.exe | FileCheck --check-prefix=RELOC-PIE %s
# RUN: llvm-bolt %t-no-init.exe -o %t-no-init --instrument | FileCheck --check-prefix=CHECK-BOLT-RT-EP %s
# RUN: llvm-readelf -hdrs %t-no-init | FileCheck --check-prefix=CHECK-NO-INIT-EP %s
# RUN: llvm-bolt %t-no-init.exe -o %t-no-init-no-ep --instrument --runtime-lib-init-hook=init | FileCheck --check-prefix=CHECK-BOLT-RT-INIT-ARRAY %s
# RUN: llvm-readelf -hdrs %t-no-init-no-ep | FileCheck --check-prefix=CHECK-NO-INIT-NO-EP %s

# RUN: %clang -shared %cflags -pie %s -Wl,-q,-init=0 -o %t-shared-no-init.exe
# RUN: llvm-readelf -d %t-shared-no-init.exe | FileCheck --check-prefix=DYN-NO-INIT %s
# RUN: llvm-readelf -l %t-shared-no-init.exe | FileCheck --check-prefix=PH-INTERP-SHARED %s
# RUN: llvm-readelf -r %t-shared-no-init.exe | FileCheck --check-prefix=RELOC-SHARED-PIE %s
# RUN: llvm-bolt %t-shared-no-init.exe -o %t-shared-no-init --instrument | FileCheck --check-prefix=CHECK-BOLT-RT-INIT-ARRAY %s
# RUN: llvm-readelf -drs %t-shared-no-init | FileCheck --check-prefix=CHECK-SHARED-NO-INIT %s

## Create a dummy shared library to link against to force creation of the dynamic section.
# RUN: %clang %cflags %p/../Inputs/stub.c -fPIC -shared -o %t-stub.so
# RUN: %clang %cflags %s -no-pie -Wl,-q,-init=0 %t-stub.so -o %t-no-pie-no-init.exe
# RUN: llvm-readelf -r %t-no-pie-no-init.exe | FileCheck --check-prefix=RELOC-NO-PIE %s
# RUN: llvm-bolt %t-no-pie-no-init.exe -o %t-no-pie-no-init --instrument | FileCheck --check-prefix=CHECK-BOLT-RT-EP %s
# RUN: llvm-readelf -hds %t-no-pie-no-init | FileCheck --check-prefix=CHECK-NO-PIE-NO-INIT-EP %s

## With init: dynamic section should contain DT_INIT
# DYN-INIT: (INIT)

## Without init: dynamic section should only contain DT_INIT_ARRAY
# DYN-NO-INIT-NOT: (INIT)
# DYN-NO-INIT:     (INIT_ARRAY)
# DYN-NO-INIT:     (INIT_ARRAYSZ)

## With interp program header (executable)
# PH-INTERP: Program Headers:
# PH-INTERP: INTERP

## Without interp program header (shared library)
# PH-INTERP-SHARED:     Program Headers:
# PH-INTERP-SHARED-NOT: INTERP

## With PIE: binary should have relative relocations
# RELOC-PIE: R_AARCH64_RELATIVE

## With PIE: binary should have relative relocations
# RELOC-SHARED-PIE: R_AARCH64_ABS64

## Without PIE: binary should not have relative relocations
# RELOC-NO-PIE-NOT: R_AARCH64_RELATIVE

## Check BOLT output output initialization hook (ELF Header Entry Point)
# CHECK-BOLT-RT-EP: runtime library initialization was hooked via ELF Header Entry Point
# CHECK-BOLT-RT-EP-NOT: runtime library initialization was hooked via DT_INIT
# CHECK-BOLT-RT-EP-NOT: runtime library initialization was hooked via .init_array entry

## Check BOLT output output initialization hook (DT_INIT)
# CHECK-BOLT-RT-INIT-NOT: runtime library initialization was hooked via ELF Header Entry Point
# CHECK-BOLT-RT-INIT: runtime library initialization was hooked via DT_INIT
# CHECK-BOLT-RT-INIT-NOT: runtime library initialization was hooked via .init_array entry

## Check BOLT output output initialization hook (.init_array entry)
# CHECK-BOLT-RT-INIT-ARRAY-NOT: runtime library initialization was hooked via ELF Header Entry Point
# CHECK-BOLT-RT-INIT-ARRAY-NOT: runtime library initialization was hooked via DT_INIT
# CHECK-BOLT-RT-INIT-ARRAY: runtime library initialization was hooked via .init_array entry

## Check that entry point address is set to __bolt_runtime_start for PIE executable with DT_INIT
# CHECK-INIT-EP:               ELF Header:
# CHECK-INIT-EP:               Entry point address: 0x[[#%x,EP_ADDR:]]
## Check that the dynamic relocation at .init and .init_array were not patched
# CHECK-INIT-EP:               Dynamic section at offset {{.*}} contains {{.*}} entries:
# CHECK-INIT-EP-NOT:           (INIT) 0x[[#%x, EP_ADDR]]
# CHECK-INIT-EP-NOT:           (INIT_ARRAY) 0x[[#%x, EP_ADDR]]
## Check that the new entry point address points to __bolt_runtime_start
# CHECK-INIT-EP:               Symbol table '.symtab' contains {{.*}} entries:
# CHECK-INIT-EP:               {{0+}}[[#%x, EP_ADDR]] {{.*}} __bolt_runtime_start

## Check that DT_INIT address is set to __bolt_runtime_start for PIE executable with DT_INIT
# CHECK-INIT-NO-EP:            ELF Header:
# CHECK-INIT-NO-EP:            Entry point address: 0x[[#%x,EP_ADDR:]]
## Read Dynamic section DT_INIT and DT_INIT_ARRAY entries
# CHECK-INIT-NO-EP:            Dynamic section at offset {{.*}} contains {{.*}} entries:
# CHECK-INIT-NO-EP-DAG:        (INIT) 0x[[#%x,INIT:]]
# CHECK-INIT-NO-EP-DAG:        (INIT_ARRAY) 0x[[#%x,INIT_ARRAY:]]
## Check if ELF entry point address points to _start symbol and new DT_INIT entry points to __bolt_runtime_start
# CHECK-INIT-NO-EP:            Symbol table '.symtab' contains {{.*}} entries:
# CHECK-INIT-NO-EP-DAG:        {{0+}}[[#%x, EP_ADDR]] {{.*}} _start
# CHECK-INIT-NO-EP-DAG:        {{0+}}[[#%x, INIT]] {{.*}} __bolt_runtime_start

## Check that 1st entry of DT_INIT_ARRAY is set to __bolt_runtime_start and DT_INIT was not changed
# CHECK-INIT-ARRAY-NO-EP:      ELF Header:
# CHECK-INIT-ARRAY-NO-EP:      Entry point address: 0x[[#%x,EP_ADDR:]]
## Read Dynamic section DT_INIT and DT_INIT_ARRAY entries
# CHECK-INIT-ARRAY-NO-EP:      Dynamic section at offset {{.*}} contains {{.*}} entries:
# CHECK-INIT-ARRAY-NO-EP-DAG:  (INIT) 0x[[#%x,INIT:]]
# CHECK-INIT-ARRAY-NO-EP-DAG:  (INIT_ARRAY) 0x[[#%x,INIT_ARRAY:]]
## Read the dynamic relocation from 1st entry of .init_array
# CHECK-INIT-ARRAY-NO-EP:      Relocation section '.rela.dyn' at offset {{.*}} contains {{.*}} entries
# CHECK-INIT-ARRAY-NO-EP:      {{0+}}[[#%x,INIT_ARRAY]] {{.*}} R_AARCH64_RELATIVE [[#%x,INIT_ADDR:]]
# CHECK-INIT-ARRAY-NO-EP-NOT:  {{0+}}[[#%x,INIT_ARRAY]] {{.*}} R_AARCH64_RELATIVE [[#%x,INIT]]
## Check that 1st entry of .init_array points to __bolt_runtime_start
# CHECK-INIT-ARRAY-NO-EP:      Symbol table '.symtab' contains {{.*}} entries:
# CHECK-INIT-ARRAY-NO-EP-DAG:  {{0+}}[[#%x, EP_ADDR]] {{.*}} _start
# CHECK-INIT-ARRAY-NO-EP-DAG:  {{[0-9]]*}}: {{0+}}[[#%x, INIT_ADDR]] {{.*}} __bolt_runtime_start

## Check that entry point address is set to __bolt_runtime_start for PIE executable without DT_INIT
# CHECK-NO-INIT-EP:            ELF Header:
# CHECK-NO-INIT-EP:            Entry point address: 0x[[#%x,EP_ADDR:]]
## Check that the dynamic relocation at .init and .init_array were not patched
# CHECK-NO-INIT-EP:            Dynamic section at offset {{.*}} contains {{.*}} entries:
# CHECK-NO-INIT-EP-NOT:        (INIT) 0x[[#%x, EP_ADDR]]
# CHECK-NO-INIT-EP-NOT:        (INIT_ARRAY) 0x[[#%x, EP_ADDR]]
## Check that the new entry point address points to __bolt_runtime_start
# CHECK-NO-INIT-EP:            Symbol table '.symtab' contains {{.*}} entries:
# CHECK-NO-INIT-EP:            {{0+}}[[#%x, EP_ADDR]] {{.*}} __bolt_runtime_start

## Check that DT_INIT is set to __bolt_runtime_start for shared library with DT_INIT
# CHECK-SHARED-INIT:           Dynamic section at offset {{.*}} contains {{.*}} entries:
# CHECK-SHARED-INIT-DAG:       (INIT) 0x[[#%x, INIT:]]
# CHECK-SHARED-INIT-DAG:       (INIT_ARRAY) 0x[[#%x, INIT_ARRAY:]]
## Check that the dynamic relocation at .init_array was not patched
# CHECK-SHARED-INIT:           Relocation section '.rela.dyn' at offset {{.*}} contains {{.*}} entries
# CHECK-SHARED-INIT-NOT:       {{0+}}[[#%x, INIT_ARRAY]] {{.*}} R_AARCH64_ABS64 {{0+}}[[#%x, INIT]]
## Check that dynamic section DT_INIT points to __bolt_runtime_start
# CHECK-SHARED-INIT:           Symbol table '.symtab' contains {{.*}} entries:
# CHECK-SHARED-INIT:           {{0+}}[[#%x, INIT]] {{.*}} __bolt_runtime_start

## Check that entry point address is set to __bolt_runtime_start for PIE executable without DT_INIT
# CHECK-NO-INIT-NO-EP:         ELF Header:
# CHECK-NO-INIT-NO-EP:         Entry point address: 0x[[#%x,EP_ADDR:]]
# CHECK-NO-INIT-NO-EP:         Dynamic section at offset {{.*}} contains {{.*}} entries:
# CHECK-NO-INIT-NO-EP-NOT:     (INIT)
# CHECK-NO-INIT-NO-EP:         (INIT_ARRAY) 0x[[#%x,INIT_ARRAY:]]
## Read the dynamic relocation from 1st entry of .init_array
# CHECK-NO-INIT-NO-EP:         Relocation section '.rela.dyn' at offset {{.*}} contains {{.*}} entries
# CHECK-NO-INIT-NO-EP:         {{0+}}[[#%x,INIT_ARRAY]] {{.*}} R_AARCH64_RELATIVE [[#%x,INIT_ADDR:]]
## Check that 1st entry of .init_array points to __bolt_runtime_start
# CHECK-NO-INIT-NO-EP:         Symbol table '.symtab' contains {{.*}} entries:
# CHECK-NO-INIT-NO-EP-DAG:     {{0+}}[[#%x, EP_ADDR]] {{.*}} _start
# CHECK-NO-INIT-NO-EP-DAG:     {{[0-9]]*}}: {{0+}}[[#%x, INIT_ADDR]] {{.*}} __bolt_runtime_start

## Check that entry point address is set to __bolt_runtime_start for shared library without DT_INIT
# CHECK-SHARED-NO-INIT:        Dynamic section at offset {{.*}} contains {{.*}} entries:
# CHECK-SHARED-NO-INIT-NOT:    (INIT)
# CHECK-SHARED-NO-INIT:        (INIT_ARRAY) 0x[[#%x,INIT_ARRAY:]]
## Read the dynamic relocation from 1st entry of .init_array
# CHECK-SHARED-NO-INIT:        Relocation section '.rela.dyn' at offset {{.*}} contains {{.*}} entries
# CHECK-SHARED-NO-INIT:        {{0+}}[[#%x, INIT_ARRAY]] {{.*}} R_AARCH64_ABS64 [[#%x,INIT_ADDR:]]
## Check that 1st entry of .init_array points to __bolt_runtime_start
# CHECK-SHARED-NO-INIT:        Symbol table '.symtab' contains {{.*}} entries:
# CHECK-SHARED-NO-INIT:        {{[0-9]]*}}: {{0+}}[[#%x, INIT_ADDR]] {{.*}} __bolt_runtime_start

## Check that entry point address is set to __bolt_runtime_start for non-PIE executable with DT_INIT
# CHECK-NO-PIE-NO-INIT-EP:     ELF Header:
# CHECK-NO-PIE-NO-INIT-EP:     Entry point address: 0x[[#%x,EP_ADDR:]]
## Check that the dynamic relocation at .init and .init_array were not patched
# CHECK-NO-PIE-NO-INIT-EP:     Dynamic section at offset {{.*}} contains {{.*}} entries:
# CHECK-NO-PIE-NO-INIT-EP-NOT: (INIT) 0x[[#%x, EP_ADDR]]
# CHECK-NO-PIE-NO-INIT-EP-NOT: (INIT_ARRAY) 0x[[#%x, EP_ADDR]]
## Check that the new entry point address points to __bolt_runtime_start
# CHECK-NO-PIE-NO-INIT-EP:     Symbol table '.symtab' contains {{.*}} entries:
# CHECK-NO-PIE-NO-INIT-EP:     {{0+}}[[#%x, EP_ADDR]] {{.*}} __bolt_runtime_start

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

  .section .init_array,"aw"
  .align 3
  .dword _init

  .section .fini_array,"aw"
  .align 3
  .dword _fini
