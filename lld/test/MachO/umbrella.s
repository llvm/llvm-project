# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t.o %s
# RUN: %lld -dylib -o %t.dylib -umbrella umbrella.dylib %t.o
# RUN: llvm-otool -lv %t.dylib | FileCheck %s

# RUN: %no-fatal-warnings-lld -bundle -o %t.so -umbrella umbrella.dylib %t.o \
# RUN:   2>&1 | FileCheck --check-prefix=WARN %s
# WARN: warning: -umbrella used, but not creating dylib
# RUN: llvm-otool -lv %t.so | FileCheck %s

# CHECK:           cmd LC_SUB_FRAMEWORK
# CHECK-NEXT:  cmdsize 32
# CHECK-NEXT: umbrella umbrella.dylib (offset 12)

.globl __Z3foo
__Z3foo:
  ret
