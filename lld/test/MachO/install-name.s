# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t.o %s

# RUN: %no_fatal_warnings_lld -o %t.exec %t.o -install_name foo 2>&1 \
# RUN:     | FileCheck --check-prefix=WARN %s
# RUN: llvm-objdump --macho --all-headers %t.exec \
# RUN:     | FileCheck --check-prefix=NO-ID %s

# RUN: %no_fatal_warnings_lld -bundle -o %t.bundle %t.o -install_name foo 2>&1 \
# RUN:     | FileCheck --check-prefix=WARN %s
# RUN: llvm-objdump --macho --all-headers %t.bundle \
# RUN:     | FileCheck --check-prefix=NO-ID %s

# RUN: %lld -dylib -o %t.dylib %t.o -install_name foo 2>&1
# RUN: llvm-objdump --macho --all-headers %t.dylib \
# RUN:     | FileCheck --check-prefix=ID %s

# WARN: warning: -install_name foo: ignored, only has effect with -dylib

# NO-ID-NOT: LC_ID_DYLIB

# ID:          cmd LC_ID_DYLIB
# ID-NEXT: cmdsize
# LID-NEXT:   name foo

.globl _main
_main:
  ret
