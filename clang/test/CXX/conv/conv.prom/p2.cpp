// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x -triple x86_64-pc-linux-gnu -ffreestanding %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++0x -triple x86_64-pc-linux-gnu -ffreestanding -fwchar-type=short -fno-signed-wchar %s
// expected-no-diagnostics

#include <stdint.h>

// In theory, the promoted types vary by platform; however, in reality they
// are quite consistent across all platforms where clang runs.

extern int promoted_wchar;
extern decltype(+L'a') promoted_wchar;

extern int promoted_char16;
extern decltype(+u'a') promoted_char16;

extern unsigned promoted_char32;
extern decltype(+U'a') promoted_char32;
