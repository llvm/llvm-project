# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: lld -flavor darwinnew -Z -platform_version macos 10.14.1 10.15 -o %t %t.o
# RUN: llvm-objdump --macho --all-headers %t | FileCheck %s

# CHECK: cmd LC_BUILD_VERSION
# CHECK-NEXT: cmdsize 32
# CHECK-NEXT: platform macos
# CHECK-NEXT: sdk 10.15
# CHECK-NEXT: minos 10.14.1
# CHECK-NEXT: ntools 1
# CHECK-NEXT: tool ld
# CHECK-NEXT: version {{[0-9\.]+}}
# CHECK-NEXT: Load command [[#]]

.text
.global _main
_main:
  mov $0, %eax
  ret
