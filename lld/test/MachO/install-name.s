# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t.o %s

# RUN: %no-fatal-warnings-lld --warn-dylib-install-name -o %t.exec %t.o \
# RUN:     -install_name foo 2>&1 | FileCheck --check-prefix=WARN %s
# RUN: llvm-objdump --macho --all-headers %t.exec \
# RUN:     | FileCheck --check-prefix=NO-ID %s

# RUN: %no-fatal-warnings-lld --warn-dylib-install-name -bundle -o %t.bundle %t.o \
# RUN:     -install_name foo 2>&1 | FileCheck --check-prefix=WARN %s
# RUN: llvm-objdump --macho --all-headers %t.bundle \
# RUN:     | FileCheck --check-prefix=NO-ID %s

# RUN: %lld -bundle -o %t.bundle %t.o -install_name foo 2>&1
# RUN: llvm-objdump --macho --all-headers %t.bundle \
# RUN:     | FileCheck --check-prefix=NO-ID %s

# RUN: %lld -bundle --warn-dylib-install-name --no-warn-dylib-install-name \
# RUN:     -o %t.bundle %t.o -install_name foo 2>&1
# RUN: llvm-objdump --macho --all-headers %t.bundle \
# RUN:     | FileCheck --check-prefix=NO-ID %s

# RUN: %lld -dylib -o %t.dylib %t.o -install_name foo 2>&1
# RUN: llvm-objdump --macho --all-headers %t.dylib \
# RUN:     | FileCheck --check-prefix=ID %s

# WARN: warning: -install_name foo: ignored, only has effect with -dylib [--warn-dylib-install-name]

# NO-ID-NOT: LC_ID_DYLIB

# ID:          cmd LC_ID_DYLIB
# ID-NEXT: cmdsize
# ID-NEXT:   name foo

.globl _main
_main:
  ret
