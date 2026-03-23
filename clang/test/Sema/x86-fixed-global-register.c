// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -DX64_NORESERVE -verify=common,x64_noreserve -fsyntax-only
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -DX64_RESERVE -target-feature +reserve-r11 -verify=common,x64_reserve -fsyntax-only
// RUN: %clang_cc1 -triple i386-unknown-linux-gnu %s -DX86_NORESERVE -verify=common,x86_noreserve -fsyntax-only
// RUN: %clang_cc1 -triple i386-unknown-linux-gnu %s -DX86_RESERVE -target-feature +reserve-edi -verify=common,x86_reserve -fsyntax-only

#if defined(X64_NORESERVE) || defined(X64_RESERVE)
register long long x64_rsp_ok __asm__("rsp");
register int x64_rsp_bad_size __asm__("rsp"); // common-error {{size of register 'rsp' does not match variable size}}
register long long x64_rbp_ok __asm__("rbp");
#endif

#ifdef X64_NORESERVE
register long long x64_r11_noreserve __asm__("r11"); // x64_noreserve-error {{register 'r11' unsuitable for global register variables on this target}}
#endif

#ifdef X64_RESERVE
register long long x64_r11_ok __asm__("r11");
register int x64_r11d_ok __asm__("r11d");
register short x64_r11w_ok __asm__("r11w");
register char x64_r11b_ok __asm__("r11b");
#endif

#if defined(X86_NORESERVE) || defined(X86_RESERVE)
register int x86_esp_ok __asm__("esp");
register long long x86_esp_bad_size __asm__("esp"); // common-error {{size of register 'esp' does not match variable size}}
register int x86_ebp_ok __asm__("ebp");
#endif

#ifdef X86_NORESERVE
register int x86_edi_noreserve __asm__("edi"); // x86_noreserve-error {{register 'edi' unsuitable for global register variables on this target}}
#endif

#ifdef X86_RESERVE
register int x86_edi_ok __asm__("edi");
register char x86_edi_bad_size __asm__("edi"); // x86_reserve-error {{size of register 'edi' does not match variable size}}
#endif
