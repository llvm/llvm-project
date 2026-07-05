# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t.o %s

# RUN: %no-fatal-warnings-lld -o %t.exec %t.o -mark_dead_strippable_dylib 2>&1 \
# RUN:     | FileCheck --check-prefix=WARN %s
# RUN: llvm-objdump --macho --private-header %t.exec \
# RUN:     | FileCheck --check-prefix=NO-DS %s

# RUN: %no-fatal-warnings-lld -bundle -o %t.bundle %t.o \
# RUN:     -mark_dead_strippable_dylib 2>&1 \
# RUN:     | FileCheck --check-prefix=WARN %s
# RUN: llvm-objdump --macho --private-header %t.bundle \
# RUN:     | FileCheck --check-prefix=NO-DS %s

# RUN: %lld -dylib -o %t.dylib %t.o -mark_dead_strippable_dylib 2>&1
# RUN: llvm-objdump --macho --private-header %t.dylib \
# RUN:     | FileCheck --check-prefix=DS %s

# WARN: warning: -mark_dead_strippable_dylib: ignored, only has effect with -dylib

# NO-DS-NOT: DEAD_STRIPPABLE_DYLIB
# DS:        DEAD_STRIPPABLE_DYLIB

.globl _main
_main:
  ret
