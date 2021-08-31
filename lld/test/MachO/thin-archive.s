# REQUIRES: x86

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/main.o %t/main.s

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/lib.o \
# RUN:     %t/mangled-symbol.s
# RUN: llvm-ar csr  %t/lib.a      %t/lib.o
# RUN: llvm-ar csrT %t/lib_thin.a %t/lib.o

# RUN: %lld %t/main.o %t/lib.a -o %t/out
# RUN: llvm-nm %t/out
# RUN: %lld %t/main.o %t/lib_thin.a -o %t/out
# RUN: llvm-nm %t/out
# RUN: %lld %t/main.o -force_load %t/lib_thin.a -o %t/out
# RUN: llvm-nm %t/out

# RUN: rm %t/lib.o
# RUN: %lld %t/main.o %t/lib.a -o %t/out
# RUN: llvm-nm %t/out
# RUN: not %lld %t/main.o %t/lib_thin.a -demangle -o %t/out 2>&1 | \
# RUN:     FileCheck --check-prefix=NOOBJ %s
# RUN: not %lld %t/main.o %t/lib_thin.a -o %t/out 2>&1 | \
# RUN:     FileCheck --check-prefix=NOOBJNODEMANGLE %s

# CHECK: __Z1fv
# CHECK: _main
# NOOBJ: error: {{.*}}lib_thin.a: could not get the member defining symbol f(): '{{.*}}lib.o': {{[N|n]}}o such file or directory
# NOOBJNODEMANGLE: error: {{.*}}lib_thin.a: could not get the member defining symbol __Z1fv: '{{.*}}lib.o': {{[N|n]}}o such file or directory

#--- mangled-symbol.s
.globl  __Z1fv
__Z1fv:
  retq

#--- main.s
.global _main
_main:
  callq __Z1fv
  mov $0, %rax
  retq
