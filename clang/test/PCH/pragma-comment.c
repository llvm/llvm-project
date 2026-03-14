// Test this without pch.
// RUN: %clang_cc1 %s -Wunknown-pragmas -Werror -triple thumbv7-windows -fms-extensions -emit-llvm -include %s -o - | FileCheck %s
// RUN: %clang_cc1 %s -Wunknown-pragmas -Werror -triple x86_64-pc-win32 -fms-extensions -emit-llvm -include %s -o - | FileCheck %s

// Test with pch.
// RUN: %clang_cc1 %s -Wunknown-pragmas -Werror -triple thumbv7-windows -fms-extensions -emit-pch -o %t
// RUN: %clang_cc1 %s -Wunknown-pragmas -Werror -triple thumbv7-windows -fms-extensions -emit-llvm -include-pch %t -o - | FileCheck %s
// RUN: %clang_cc1 %s -Wunknown-pragmas -Werror -triple x86_64-pc-win32 -fms-extensions -emit-pch -o %t
// RUN: %clang_cc1 %s -Wunknown-pragmas -Werror -triple x86_64-pc-win32 -fms-extensions -emit-llvm -include-pch %t -o - | FileCheck %s

// The first run line creates a pch, and since at that point HEADER is not
// defined, the only thing contained in the pch is the pragma. The second line
// then includes that pch, so HEADER is defined and the actual code is compiled.
// The check then makes sure that the pragma is in effect in the file that
// includes the pch.

#ifndef HEADER
#define HEADER
#pragma comment(lib, "foo.lib")

#else

// CHECK: "/DEFAULTLIB:foo.lib"

#endif
