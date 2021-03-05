# REQUIRES: x86, aarch64

# RUN: rm -rf %t
# RUN: split-file %s %t


# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos -o %t/main-arm64.o %t/main.s
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/main-x86_64.o %t/main.s
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos -o %t/foo-arm64.o %t/foo.s
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/foo-x86_64.o %t/foo.s

# Exhaustive test for:
# (x86_64, arm64) x (default, -adhoc_codesign, -no_adhoc-codesign) x (execute, dylib, bundle)

# RUN: %lld -lSystem -arch x86_64 -execute -o %t/out %t/main-x86_64.o
# RUN: llvm-objdump --macho --all-headers %t/out | FileCheck --check-prefix=NO-ADHOC %s
# RUN: %lld -arch x86_64 -dylib   -o %t/out %t/foo-x86_64.o
# RUN: llvm-objdump --macho --all-headers  %t/out| FileCheck --check-prefix=NO-ADHOC %s
# RUN: %lld -arch x86_64 -bundle  -o %t/out %t/foo-x86_64.o
# RUN: llvm-objdump --macho --all-headers  %t/out| FileCheck --check-prefix=NO-ADHOC %s

# RUN: %lld -lSystem -arch x86_64 -execute -adhoc_codesign -o %t/out %t/main-x86_64.o
# RUN: llvm-objdump --macho --all-headers %t/out| FileCheck --check-prefix=ADHOC %s
# RUN: %lld -arch x86_64 -dylib   -adhoc_codesign -o %t/out %t/foo-x86_64.o
# RUN: llvm-objdump --macho --all-headers %t/out| FileCheck --check-prefix=ADHOC %s
# RUN: %lld -arch x86_64 -bundle  -adhoc_codesign -o %t/out %t/foo-x86_64.o
# RUN: llvm-objdump --macho --all-headers %t/out| FileCheck --check-prefix=ADHOC %s

# RUN: %lld -lSystem -arch x86_64 -execute -no_adhoc_codesign -o %t/out %t/main-x86_64.o
# RUN: llvm-objdump --macho --all-headers %t/out| FileCheck --check-prefix=NO-ADHOC %s
# RUN: %lld -arch x86_64 -dylib   -no_adhoc_codesign -o %t/out %t/foo-x86_64.o
# RUN: llvm-objdump --macho --all-headers %t/out| FileCheck --check-prefix=NO-ADHOC %s
# RUN: %lld -arch x86_64 -bundle  -no_adhoc_codesign -o %t/out %t/foo-x86_64.o
# RUN: llvm-objdump --macho --all-headers %t/out| FileCheck --check-prefix=NO-ADHOC %s


# RUN: %lld -lSystem -arch arm64 -execute -o %t/out %t/main-arm64.o
# RUN: llvm-objdump --macho --all-headers %t/out | FileCheck --check-prefix=ADHOC %s
# RUN: %lld -arch arm64 -dylib   -o %t/out %t/foo-arm64.o
# RUN: llvm-objdump --macho --all-headers  %t/out| FileCheck --check-prefix=ADHOC %s
# RUN: %lld -arch arm64 -bundle  -o %t/out %t/foo-arm64.o
# RUN: llvm-objdump --macho --all-headers  %t/out| FileCheck --check-prefix=ADHOC %s

# RUN: %lld -lSystem -arch arm64 -execute -adhoc_codesign -o %t/out %t/main-arm64.o
# RUN: llvm-objdump --macho --all-headers %t/out| FileCheck --check-prefix=ADHOC %s
# RUN: %lld -arch arm64 -dylib   -adhoc_codesign -o %t/out %t/foo-arm64.o
# RUN: llvm-objdump --macho --all-headers %t/out| FileCheck --check-prefix=ADHOC %s
# RUN: %lld -arch arm64 -bundle  -adhoc_codesign -o %t/out %t/foo-arm64.o
# RUN: llvm-objdump --macho --all-headers %t/out| FileCheck --check-prefix=ADHOC %s

# RUN: %lld -lSystem -arch arm64 -execute -no_adhoc_codesign -o %t/out %t/main-arm64.o
# RUN: llvm-objdump --macho --all-headers %t/out| FileCheck --check-prefix=NO-ADHOC %s
# RUN: %lld -arch arm64 -dylib   -no_adhoc_codesign -o %t/out %t/foo-arm64.o
# RUN: llvm-objdump --macho --all-headers %t/out| FileCheck --check-prefix=NO-ADHOC %s
# RUN: %lld -arch arm64 -bundle  -no_adhoc_codesign -o %t/out %t/foo-arm64.o
# RUN: llvm-objdump --macho --all-headers %t/out| FileCheck --check-prefix=NO-ADHOC %s

# ADHOC:          cmd LC_CODE_SIGNATURE
# ADHOC-NEXT: cmdsize 16

# NO-ADHOC-NOT:          cmd LC_CODE_SIGNATURE

#--- foo.s
.globl _foo
_foo:
  ret

#--- main.s
.globl _main
_main:
  ret
