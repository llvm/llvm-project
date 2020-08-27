# REQUIRES: x86
# RUN: mkdir -p %t

## This test verifies that a non-TBD dylib can re-export a TBD library.

# RUN: echo "" | llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/reexporter.o
# RUN: lld -flavor darwinnew -dylib -syslibroot %S/Inputs/MacOSX.sdk -lc++ -sub_library libc++ \
# RUN:   %t/reexporter.o -o %t/libreexporter.dylib
# RUN: llvm-objdump --macho --all-headers %t/libreexporter.dylib | FileCheck %s --check-prefix=DYLIB-HEADERS
# DYLIB-HEADERS:     cmd     LC_REEXPORT_DYLIB
# DYLIB-HEADERS-NOT: Load command
# DYLIB-HEADERS:     name    /usr/lib/libc++.dylib
