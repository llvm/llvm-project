// RUN: %clang_cc1 -fsyntax-only -triple x86_64-pc-windows-msvc -fms-compatibility-version=17.00 -fno-wchar %s
// MSVC defines wchar_t instead of using the builtin if /Zc:wchar_t- is passed

#include <stddef.h>

wchar_t c;
