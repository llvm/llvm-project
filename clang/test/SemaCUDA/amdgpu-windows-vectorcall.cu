// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -aux-triple x86_64-pc-windows-msvc -fms-compatibility -fcuda-is-device -fsyntax-only -verify %s

__cdecl void hostf1();
__vectorcall void (*hostf2)() = hostf1; // expected-error {{cannot initialize a variable of type 'void ((*))() __attribute__((vectorcall))' with an lvalue of type 'void () __attribute__((cdecl))'}}
