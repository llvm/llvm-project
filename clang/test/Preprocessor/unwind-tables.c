// RUN: %clang_cc1 %s -verify -munwind-tables -triple x86_64-windows
// RUN: %clang_cc1 %s -verify -triple x86_64-unknown-elf

// RUN: %clang_cc1 %s -verify -munwind-tables -DCFI_ASM -triple x86_64-unknown-elf
// RUN: %clang_cc1 %s -verify -munwind-tables -DCFI_ASM -triple aarch64-apple-darwin
// RUN: %clang_cc1 %s -verify -debug-info-kind=line-tables-only -DCFI_ASM -triple x86_64-unknown-elf
// RUN: %clang_cc1 %s -verify -fexceptions -DCFI_ASM -triple x86_64-unknown-elf

// expected-no-diagnostics

#ifdef CFI_ASM
  #if __GCC_HAVE_DWARF2_CFI_ASM != 1
  #error "__GCC_HAVE_DWARF2_CFI_ASM not defined"
  #endif
#else
  #ifdef __GCC_HAVE_DWARF2_CFI_ASM
  #error "__GCC_HAVE_DWARF2_CFI_ASM defined"
  #endif
#endif
