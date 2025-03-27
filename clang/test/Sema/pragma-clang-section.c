// RUN: %clang_cc1 -fsyntax-only -verify %s -triple arm-none-eabi
// RUN: %clang_cc1 -fsyntax-only -verify %s -triple arm64-windows-msvc
#pragma clang section bss = "mybss.1" data = "mydata.1" rodata = "myrodata.1" text = "mytext.1" // expected-note 2 {{#pragma entered here}}
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

#pragma clang section bss = "mybss.3" data = "mybss.3" // expected-error {{this causes a section type conflict with a prior #pragma section}} expected-note {{#pragma entered here}} expected-note {{#pragma entered here}}
#pragma clang section rodata = "mydata.1"              // expected-error {{this causes a section type conflict with a prior #pragma section}}
#pragma clang section bss = "myrodata.1"               // expected-error {{this causes a section type conflict with a prior #pragma section}}
#pragma clang section text = "mybss.3"                 // expected-error {{this causes a section type conflict with a prior #pragma section}}

#pragma clang section rodata = "myrodata.4"              // expected-note {{#pragma entered here}}
int x __attribute__((section("myrodata.4")));            // expected-error {{'x' causes a section type conflict with a prior #pragma section}}
const int y __attribute__((section("myrodata.5"))) = 10; // expected-note {{declared here}}
#pragma clang section data = "myrodata.5"                // expected-error {{this causes a section type conflict with 'y'}}
const int z __attribute__((section("myrodata.6"))) = 11;

int a;
