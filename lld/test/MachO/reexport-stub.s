# REQUIRES: x86
# RUN: mkdir -p %t

## This test verifies that a non-TBD dylib can re-export a TBD library.

# RUN: echo "" | llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/reexporter.o
# RUN: %lld -dylib -lc++ -sub_library libc++ \
# RUN:   %t/reexporter.o -o %t/libreexporter.dylib
# RUN: llvm-objdump --macho --all-headers %t/libreexporter.dylib | FileCheck %s --check-prefix=DYLIB-HEADERS
# DYLIB-HEADERS:     cmd     LC_REEXPORT_DYLIB
# DYLIB-HEADERS-NOT: Load command
# DYLIB-HEADERS:     name    /usr/lib/libc++.dylib

# RUN: llvm-mc -filetype obj -triple x86_64-apple-darwin %s -o %t/test.o
# RUN: %lld -o %t/test -lSystem -L%t -lreexporter %t/test.o
# RUN: llvm-objdump --bind --no-show-raw-insn -d %t/test | FileCheck %s

# CHECK: Bind table:
# CHECK-DAG: __DATA __data {{.*}} pointer 0 libreexporter ___gxx_personality_v0

.text
.globl _main

_main:
  ret

.data
  .quad ___gxx_personality_v0
