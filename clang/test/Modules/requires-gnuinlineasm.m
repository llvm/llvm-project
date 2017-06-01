// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules \
// RUN:     -fimplicit-module-maps -F %S/Inputs/GNUAsm %s \
// RUN:     -fno-gnu-inline-asm -DNO_ASM_INLINE -verify
// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules \
// RUN:     -fimplicit-module-maps -F %S/Inputs/GNUAsm %s \
// RUN:     -DASM_INLINE -verify

#ifdef NO_ASM_INLINE
@import NeedsGNUInlineAsm.Asm; // expected-error{{module 'NeedsGNUInlineAsm.Asm' requires feature 'gnuinlineasm'}}
#endif

#ifdef ASM_INLINE
@import NeedsGNUInlineAsm.Asm; // expected-no-diagnostics
#endif
