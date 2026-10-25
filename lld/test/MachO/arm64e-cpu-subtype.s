# REQUIRES: aarch64
## As with ld64, the arm64e CPU subtype carries ptrauth ABI version 1 for macOS
## and Mac Catalyst executables, and version 0 for everything else.

# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t.o %s
# DEFINE: %{link} = %no-arg-lld -arch arm64e -fatal_warnings %t.o -o %t.out
# DEFINE: %{subtype} = llvm-objdump --macho --private-header %t.out

# RUN: %{link} -platform_version macos 13.0 13.0 -execute
# RUN: %{subtype} | FileCheck %s --check-prefixes=PAC01,EXECUTE
# RUN: %{link} -platform_version mac-catalyst 16.0 16.0 -execute
# RUN: %{subtype} | FileCheck %s --check-prefixes=PAC01,EXECUTE
# RUN: %{link} -platform_version ios 16.0 16.0 -execute
# RUN: %{subtype} | FileCheck %s --check-prefixes=PAC00,EXECUTE

# RUN: %{link} -platform_version macos 13.0 13.0 -dylib
# RUN: %{subtype} | FileCheck %s --check-prefixes=PAC00,DYLIB
# RUN: %{link} -platform_version mac-catalyst 16.0 16.0 -dylib
# RUN: %{subtype} | FileCheck %s --check-prefixes=PAC00,DYLIB
# RUN: %{link} -platform_version ios 16.0 16.0 -dylib
# RUN: %{subtype} | FileCheck %s --check-prefixes=PAC00,DYLIB

# RUN: %{link} -platform_version macos 13.0 13.0 -bundle
# RUN: %{subtype} | FileCheck %s --check-prefixes=PAC00,BUNDLE
# RUN: %{link} -platform_version ios 16.0 16.0 -bundle
# RUN: %{subtype} | FileCheck %s --check-prefixes=PAC00,BUNDLE

# PAC00:   MH_MAGIC_64 ARM64 E PAC00
# PAC01:   MH_MAGIC_64 ARM64 E PAC01
# EXECUTE-SAME: EXECUTE
# DYLIB-SAME:   DYLIB
# BUNDLE-SAME:  BUNDLE

.globl _main
.p2align 2
_main:
  ret
