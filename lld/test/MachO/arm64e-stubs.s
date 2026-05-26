# REQUIRES: aarch64

## Test arm64e authenticated stubs use braa with x17 context.
## ARM64e stubs are 16 bytes (4 instructions), not 12 like arm64,
## because they compute the GOT address in x17 for use as the
## authentication context in the braa instruction.
##
## With chained fixups on arm64e, the stubs section is called
## __auth_stubs and references the __auth_got section.

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/foo.o %t/foo.s
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/test.o %t/test.s
# RUN: %no-arg-lld -arch arm64e -platform_version macos 13.0 13.0 \
# RUN:   -syslibroot %S/Inputs/MacOSX.sdk -lSystem \
# RUN:   -dylib -install_name @executable_path/libfoo.dylib %t/foo.o -o %t/libfoo.dylib
# RUN: %no-arg-lld -arch arm64e -platform_version macos 13.0 13.0 \
# RUN:   -syslibroot %S/Inputs/MacOSX.sdk -lSystem \
# RUN:   %t/libfoo.dylib %t/test.o -o %t/test
# RUN: llvm-objdump --no-print-imm-hex --macho -d --no-show-raw-insn \
# RUN:   --section="__TEXT,__auth_stubs" %t/test | FileCheck %s

## Verify the main function calls through a stub.
# CHECK-LABEL: _main:
# CHECK:       bl {{.*}} ; symbol stub for: _foo

## Verify the stub uses the arm64e 4-instruction sequence with braa.
# CHECK-LABEL: Contents of (__TEXT,__auth_stubs) section
# CHECK-NEXT:  {{[0-9a-f]+}}: adrp x17
# CHECK-NEXT:                 add  x17, x17, {{.*}} ; literal pool symbol address: _foo
# CHECK-NEXT:                 ldr  x16, [x17]
# CHECK-NEXT:                 braa x16, x17

## Verify that the __auth_got section exists in __DATA_CONST.
# RUN: llvm-objdump --macho --all-headers %t/test | FileCheck %s --check-prefix=HEADERS
# HEADERS:      sectname __auth_got
# HEADERS-NEXT: segname __DATA_CONST

#--- foo.s
.globl _foo
_foo:
  ret

#--- test.s
.text
.globl _main

.p2align 2
_main:
  bl _foo
  ret
