# REQUIRES: system-linux

## Check that BOLT correctly detects the Linux kernel version

# RUN: %clang -DA -target x86_64-unknown-unknown \
# RUN:   %cflags -nostdlib %s -o %t.exe \
# RUN:   -Wl,--image-base=0xffffffff80000000,--no-dynamic-linker,--no-eh-frame-hdr
# RUN: llvm-bolt %t.exe -o %t.out 2>&1 | FileCheck --check-prefix=CHECK-A %s

# RUN: %clang -DB -target x86_64-unknown-unknown \
# RUN:   %cflags -nostdlib %s -o %t.exe \
# RUN:   -Wl,--image-base=0xffffffff80000000,--no-dynamic-linker,--no-eh-frame-hdr
# RUN: llvm-bolt %t.exe -o %t.out 2>&1 | FileCheck --check-prefix=CHECK-B %s

# RUN: %clang -DC -target x86_64-unknown-unknown \
# RUN:   %cflags -nostdlib %s -o %t.exe \
# RUN:   -Wl,--image-base=0xffffffff80000000,--no-dynamic-linker,--no-eh-frame-hdr
# RUN: llvm-bolt %t.exe -o %t.out 2>&1 | FileCheck --check-prefix=CHECK-C %s

  .text
  .globl foo
  .type foo, %function
foo:
  ret
  .size foo, .-foo

## Linux kernel version
  .rodata
  .align 16
  .globl linux_banner
  .type  linux_banner, @object
linux_banner:

#ifdef A
  .string  "Linux version 6.6.61\n"
#endif
# CHECK-A: BOLT-INFO: Linux kernel version is 6.6.61

#ifdef B
  .string  "Linux version 6.6.50-rc4\n"
#endif
# CHECK-B: BOLT-INFO: Linux kernel version is 6.6.50

#ifdef C
  .string  "Linux version 6.6\n"
#endif
# CHECK-C: BOLT-INFO: Linux kernel version is 6.6

  .size  linux_banner, . - linux_banner

## Fake Linux Kernel sections.
  .section __ksymtab,"a",@progbits
  .section __ksymtab_gpl,"a",@progbits
