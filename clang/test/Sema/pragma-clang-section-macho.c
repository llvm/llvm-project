// RUN: %clang_cc1 -fsyntax-only -verify %s -triple arm64-apple-ios

#pragma clang section bss = "" data = "" rodata = "" text = ""
#pragma clang section bss = "" data = "" rodata = "" text = "__TEXT,__text"
#pragma clang section bss = "" data = "" rodata = "" text = "badname"                       // expected-error {{argument to #pragma section is not valid for this target: mach-o section specifier requires a segment and section separated by a comma}}
#pragma clang section bss = "" data = "" rodata = "" text = "__TEXT,__namethatiswaytoolong" // expected-error {{argument to #pragma section is not valid for this target: mach-o section specifier requires a section whose length is between 1 and 16 characters}}
#pragma clang section

int a;
