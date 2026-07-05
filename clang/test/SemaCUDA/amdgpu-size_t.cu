// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -aux-triple x86_64-pc-windows-msvc -fms-compatibility -fcuda-is-device -fsyntax-only -verify %s

// expected-no-diagnostics
typedef unsigned __int64 size_t;
typedef __int64 intptr_t;
typedef unsigned __int64 uintptr_t;

