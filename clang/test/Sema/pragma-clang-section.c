// RUN: %clang_cc1 -fsyntax-only -verify %s -triple arm-none-eabi
#pragma clang section bss="mybss.1" data="mydata.1" rodata="myrodata.1" text="mytext.1"
#pragma clang section bss="" data="" rodata="" text=""
#pragma clang section

#pragma clang section dss="mybss.2" // expected-error {{expected one of [bss|data|rodata|text|relro] section kind in '#pragma clang section'}}
#pragma clang section deta="mydata.2" // expected-error {{expected one of [bss|data|rodata|text|relro] section kind in '#pragma clang section'}}
#pragma clang section rodeta="rodata.2" // expected-error {{expected one of [bss|data|rodata|text|relro] section kind in '#pragma clang section'}}
#pragma clang section taxt="text.2" // expected-error {{expected one of [bss|data|rodata|text|relro] section kind in '#pragma clang section'}}

#pragma clang section section bss="mybss.2"  // expected-error {{expected one of [bss|data|rodata|text|relro] section kind in '#pragma clang section'}}

#pragma clang section bss "mybss.2"   // expected-error {{expected '=' following '#pragma clang section bss'}}
#pragma clang section data "mydata.2"   // expected-error {{expected '=' following '#pragma clang section data'}}
#pragma clang section rodata "myrodata.2"   // expected-error {{expected '=' following '#pragma clang section rodata'}}
#pragma clang section text "text.2"   // expected-error {{expected '=' following '#pragma clang section text'}}
#pragma clang section relro "relro.2"   // expected-error {{expected '=' following '#pragma clang section relro'}}
#pragma clang section bss="" data="" rodata="" text="" more //expected-error {{expected one of [bss|data|rodata|text|relro] section kind in '#pragma clang section'}}
int a;
